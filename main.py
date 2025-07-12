from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import fitz  # PyMuPDF
import pandas as pd
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re
import logging
import asyncio
from dotenv import load_dotenv

# --- Initial Setup ---
load_dotenv()

# Logging Configuration
LOG_FILE = "app.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI(title="Gemini PDF Extractor v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise ValueError("GEMINI_API_KEY is required")
genai.configure(api_key=GEMINI_API_KEY)

# Global variable to hold the progress stream queue
progress_queues: Dict[str, asyncio.Queue] = {}

# --- Pydantic Models ---
class AnalysisPayload(BaseModel):
    file_path: str

class ExtractionPayload(BaseModel):
    file_path: str
    columns: Dict[str, str]
    record_separator: str

class ColumnAnalysisResponse(BaseModel):
    success: bool
    columns: Dict[str, str]
    record_separator: str
    sample_data: List[Dict[str, Any]]
    message: str
    file_path: str

class DataExtractionResponse(BaseModel):
    success: bool
    file_path: str
    total_records: int
    message: str

# --- PDF Processor Class ---
class PDFProcessor:
    def __init__(self, task_id: str):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.upload_dir = "uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.task_id = task_id
        progress_queues[self.task_id] = asyncio.Queue()

    async def _update_progress(self, status: str, percentage: int):
        await progress_queues[self.task_id].put({"status": status, "percentage": percentage})

    def _get_layout_aware_text(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        logger.info(f"Extracting layout-aware text from '{pdf_path}'.")
        full_text = ""
        try:
            with fitz.open(pdf_path) as doc:
                pages_to_process = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
                for page_num in range(pages_to_process):
                    page = doc.load_page(page_num)
                    # Get text as blocks to preserve layout, then sort by vertical position
                    blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
                    full_text += f"--- Page {page_num + 1} ---\n"
                    for b in blocks:
                        full_text += b[4] # The text content of the block
                    full_text += "\n\n"
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error in _get_layout_aware_text: {e}")
            raise

    async def analyze_document_structure(self, sample_text: str) -> Dict[str, Any]:
        await self._update_progress("Analyzing document with Gemini AI...", 25)
        analysis_prompt = f"""
        You are an expert data extraction AI. Analyze the following document sample to identify data fields, generate regex for them, and determine how records are separated.

        Document Sample:
        {sample_text[:8000]}

        Your Task:
        1.  Identify the main data fields/columns.
        2.  For each field, create a specific Python-compatible regex pattern to capture the data. Use capturing groups `()`.
        3.  Determine the regex pattern that separates individual records (e.g., a newline separating table rows, or a specific phrase). This is the `record_separator`.
        4.  Provide a few examples of extracted data.

        Return your analysis in a clean JSON format with these keys:
        -   `"columns"`: A dictionary of `{{"Column Name": "Regex Pattern"}}`.
        -   `"record_separator"`: A regex pattern string that splits the text into individual records.
        -   `"sample_data"`: A list of dictionaries, where each dictionary is a row of extracted sample data.

        Example JSON:
        {{
            "columns": {{
                "Date": "(\\d{{2}}-\\d{{2}}-\\d{{4}})",
                "Description": "Description:\\s*(.*?)\\n",
                "Amount": "\\$([\\d,]+\\.\\d{{2}})"
            }},
            "record_separator": "\\n---Next Record---\\n",
            "sample_data": [
                {{"Date": "01-15-2024", "Description": "Office Supplies", "Amount": "150.00"}}
            ]
        }}
        """
        try:
            response = self.model.generate_content(analysis_prompt)
            await self._update_progress("Parsing AI response...", 40)
            return self._parse_analysis_response(response.text)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        logger.info("Parsing Gemini's analysis response.")
        try:
            json_match = re.search(r'```json\n(\{.*?\})\n```', response_text, re.DOTALL) or re.search(r'(\{.*?\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                if 'columns' not in result or 'record_separator' not in result:
                    raise ValueError("Parsed JSON is missing required keys.")
                return result
            raise ValueError("Invalid JSON response from AI.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing analysis JSON: {e}\nRaw response: {response_text}")
            raise

    async def extract_data_with_patterns(self, full_text: str, columns: Dict[str, str], record_separator: str) -> List[Dict[str, Any]]:
        await self._update_progress("Splitting document into records...", 70)
        try:
            # Split the entire text into chunks based on the record separator
            chunks = re.split(record_separator, full_text)
        except re.error as e:
            logger.error(f"Invalid record separator regex: {e}")
            chunks = full_text.split("\n\n") # Fallback

        all_records = []
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10: continue # Skip empty or tiny chunks
            
            progress_percent = 70 + int((i / total_chunks) * 25) # Progress from 70% to 95%
            await self._update_progress(f"Extracting from record {i+1}/{total_chunks}", progress_percent)

            record = {}
            for col_name, pattern in columns.items():
                try:
                    match = re.search(pattern, chunk, re.DOTALL | re.IGNORECASE)
                    record[col_name] = match.group(1).strip() if match and match.groups() else ""
                except re.error as e:
                    logger.warning(f"Regex error for column '{col_name}': {e}")
                    record[col_name] = "REGEX_ERROR"
            
            if any(record.values()): # Add if at least one value was found
                all_records.append(record)
        
        logger.info(f"Extraction finished. Found {len(all_records)} records.")
        await self._update_progress("Finalizing export...", 98)
        return all_records

    def export_to_excel(self, data: List[Dict[str, Any]], original_filename: str) -> str:
        if not data: raise ValueError("No data to export")
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(original_filename)[0]
        file_path = os.path.join(self.output_dir, f"{base_filename}_{timestamp}.xlsx")
        df.to_excel(file_path, index=False)
        logger.info(f"Exported data to '{file_path}'.")
        return file_path

    def save_uploaded_file(self, file: UploadFile) -> str:
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        file_path = os.path.join(self.upload_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to '{file_path}'.")
        return file_path

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-and-analyze", response_model=ColumnAnalysisResponse)
async def upload_and_analyze_pdf(request: Request, file: UploadFile = File(...)):
    task_id = request.headers.get("X-Task-ID")
    if not task_id or task_id not in progress_queues:
        raise HTTPException(status_code=400, detail="Missing or invalid Task ID")
    
    processor = PDFProcessor(task_id)
    await processor._update_progress("Saving uploaded file...", 5)
    file_path = processor.save_uploaded_file(file)

    try:
        await processor._update_progress("Extracting text from PDF...", 10)
        sample_text = processor._get_layout_aware_text(file_path, max_pages=3)
        if not sample_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text.")
        
        analysis_result = await processor.analyze_document_structure(sample_text)
        await processor._update_progress("Analysis complete.", 50)
        
        return ColumnAnalysisResponse(
            success=True,
            columns=analysis_result["columns"],
            record_separator=analysis_result.get("record_separator", "\\n\\s*\\n"),
            sample_data=analysis_result.get("sample_data", []),
            message="Analysis complete. Review patterns before extraction.",
            file_path=file_path
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-data", response_model=DataExtractionResponse)
async def extract_data_from_pdf(request: Request, payload: ExtractionPayload):
    task_id = request.headers.get("X-Task-ID")
    if not task_id or task_id not in progress_queues:
        raise HTTPException(status_code=400, detail="Missing or invalid Task ID")

    processor = PDFProcessor(task_id)
    file_path = payload.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        await processor._update_progress("Extracting full text...", 60)
        full_text = processor._get_layout_aware_text(file_path)
        
        extracted_data = await processor.extract_data_with_patterns(full_text, payload.columns, payload.record_separator)
        if not extracted_data:
            raise HTTPException(status_code=400, detail="No data extracted.")

        original_filename = os.path.basename(file_path)
        excel_file_path = processor.export_to_excel(extracted_data, original_filename)
        await processor._update_progress("Done!", 100)
        
        return DataExtractionResponse(
            success=True,
            file_path=os.path.basename(excel_file_path),
            total_records=len(extracted_data),
            message=f"Successfully extracted {len(extracted_data)} records."
        )
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    if task_id not in progress_queues:
        # This can happen if the client connects before the task is created.
        # We'll create a queue here to be safe.
        progress_queues[task_id] = asyncio.Queue()

    async def progress_generator():
        while True:
            try:
                progress = await asyncio.wait_for(progress_queues[task_id].get(), timeout=30)
                yield f"data: {json.dumps(progress)}\n\n"
                if progress["percentage"] == 100:
                    break
            except asyncio.TimeoutError:
                # If no update for 30s, assume connection is stale and close it.
                break
        del progress_queues[task_id]

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.post("/save-patterns")
async def save_patterns(payload: dict):
    # In a real app, you'd save this to a user account or a database.
    # For this demo, we save it to a local file.
    patterns_dir = "patterns"
    os.makedirs(patterns_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(patterns_dir, f"pattern_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
    return JSONResponse({"success": True, "message": f"Patterns saved to {file_path}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
