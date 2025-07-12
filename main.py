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
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import re
import logging
import asyncio
from dotenv import load_dotenv
from collections import Counter
import unicodedata
from dateutil.parser import parse as parse_date

# --- Initial Setup ---
load_dotenv()

# Logging Configuration
LOG_FILE = "app.log"
### FIX: Replaced os.remove() with the filemode='w' argument in basicConfig.
# This safely overwrites the log file on each application start, avoiding file lock errors during hot-reloading.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # mode='w' will overwrite the file
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI(title="Gemini PDF Extractor v4 (Professional)")
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
class ExtractionPayload(BaseModel):
    file_path: str
    columns: Dict[str, str]
    record_separator: str
    data_start_indicator: str
    data_types: Dict[str, str]

class ColumnAnalysisResponse(BaseModel):
    success: bool
    columns: Dict[str, str]
    record_separator: str
    data_start_indicator: str
    data_types: Dict[str, str]
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
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        self.upload_dir = "uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.task_id = task_id
        progress_queues[self.task_id] = asyncio.Queue()

    async def _update_progress(self, status: str, percentage: int):
        await progress_queues[self.task_id].put({"status": status, "percentage": percentage})

    def _normalize_text(self, text: str) -> str:
        """Cleans and standardizes text block content."""
        ligatures = {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl'}
        for L, l in ligatures.items():
            text = text.replace(L, l)
        text = unicodedata.normalize('NFC', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _detect_and_remove_headers_footers(self, pages_blocks: List[List[Tuple[float, float, float, float, str, int, int]]], page_heights: List[float]) -> List[List[Tuple]]:
        """Identifies and removes common headers/footers from page blocks."""
        header_margin, footer_margin = 0.15, 0.90
        potential_hf_texts = Counter()
        
        for i, page_blocks in enumerate(pages_blocks):
            h = page_heights[i]
            for b in page_blocks:
                if b[1] < h * header_margin or b[3] > h * footer_margin:
                    normalized_text = self._normalize_text(b[4])
                    if normalized_text and not normalized_text.isdigit() and len(normalized_text) > 3:
                        potential_hf_texts[normalized_text] += 1
        
        num_pages = len(pages_blocks)
        if num_pages < 2: return pages_blocks # Can't detect repetition on a single page
        
        common_texts = {text for text, count in potential_hf_texts.items() if count > num_pages / 2}
        
        if common_texts:
            logger.info(f"Identified common header/footer text to remove: {common_texts}")
        
        clean_pages_blocks = []
        for page_blocks in pages_blocks:
            clean_page = [b for b in page_blocks if self._normalize_text(b[4]) not in common_texts]
            clean_pages_blocks.append(clean_page)
            
        return clean_pages_blocks

    def _get_layout_aware_text(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        logger.info(f"Starting advanced text extraction from '{pdf_path}'.")
        full_text = ""
        try:
            with fitz.open(pdf_path) as doc:
                pages_to_process = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
                
                all_pages_blocks = []
                page_heights = []
                for page_num in range(pages_to_process):
                    page = doc.load_page(page_num)
                    all_pages_blocks.append(page.get_text("blocks"))
                    page_heights.append(page.rect.height)

                logger.info("Detecting and removing headers/footers...")
                clean_pages_blocks = self._detect_and_remove_headers_footers(all_pages_blocks, page_heights)

                for page_num, blocks in enumerate(clean_pages_blocks):
                    sorted_blocks = sorted(blocks, key=lambda b: b[1]) 
                    full_text += f"--- Page {page_num + 1} ---\n"
                    for b in sorted_blocks:
                        normalized_block_text = self._normalize_text(b[4])
                        if normalized_block_text:
                            full_text += normalized_block_text + "\n"
                    full_text += "\n"
            
            logger.info("Advanced text extraction and cleaning complete.")
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error in _get_layout_aware_text: {e}", exc_info=True)
            raise

    async def analyze_document_structure(self, sample_text: str) -> Dict[str, Any]:
        await self._update_progress("Analyzing document with Gemini AI...", 25)
        analysis_prompt = f"""
        You are an expert data extraction AI. Analyze the following document sample.

        Document Sample:
        {sample_text[:8000]}

        Your Task:
        1.  Identify the main data fields/columns in the repeating data table.
        2.  For each field, create a specific Python-compatible regex pattern to capture its value. Use a single capturing group `()`.
        3.  Determine the `data_types` for each column. Use one of: "string", "number", "date".
        4.  Determine the `data_start_indicator`: A string/regex that marks the beginning of the repeating data (often the table header row).
        5.  Determine the `record_separator`: A regex pattern that separates individual data records (e.g., `\\n`).
        6.  Provide a few examples of extracted data *before* any type conversion.

        Return your analysis in a clean JSON format.

        Example JSON output:
        {{
            "columns": {{
                "Transaction Date": "(\\d{{2}}-\\d{{2}}-\\d{{4}})",
                "Description": "Description:\\s*(.*?)(?=\\s*Amount:|$)",
                "Amount": "Amount:\\s*\\$([\\d,.-]+)"
            }},
            "data_types": {{
                "Transaction Date": "date",
                "Description": "string",
                "Amount": "number"
            }},
            "data_start_indicator": "Transaction Date\\s+Description\\s+Amount",
            "record_separator": "\\n",
            "sample_data": [
                {{"Transaction Date": "01-15-2024", "Description": "Office Supplies", "Amount": "150.00"}}
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
            if not json_match:
                raise ValueError("No JSON block found in the AI response.")
            
            json_str = json_match.group(1)
            result = json.loads(json_str)

            required_keys = ['columns', 'record_separator', 'data_start_indicator', 'data_types']
            if not all(key in result for key in required_keys):
                raise ValueError(f"Parsed JSON is missing one or more required keys: {required_keys}.")
            
            result.setdefault('sample_data', [])
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing analysis JSON: {e}\nRaw response: {response_text}")
            raise

    def _clean_record(self, record: Dict[str, str], data_types: Dict[str, str]) -> Dict[str, Any]:
        cleaned = {}
        for col_name, value in record.items():
            if not value:
                cleaned[col_name] = None # Use None for empty values for better pandas/DB handling
                continue

            dtype = data_types.get(col_name, "string")
            
            if dtype == "date":
                try:
                    cleaned[col_name] = parse_date(value).strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    cleaned[col_name] = value 
            elif dtype == "number":
                try:
                    num_str = value.strip()
                    is_negative = num_str.startswith('(') and num_str.endswith(')')
                    num_str = re.sub(r'[^\d.-]', '', num_str)
                    if not num_str:
                        cleaned[col_name] = None
                        continue
                    number = float(num_str)
                    if is_negative:
                        number *= -1
                    cleaned[col_name] = int(number) if number.is_integer() else number
                except (ValueError, TypeError):
                    cleaned[col_name] = value
            else:
                cleaned[col_name] = value.strip()
        return cleaned

    async def extract_data_with_patterns(self, full_text: str, columns: Dict[str, str], record_separator: str, data_start_indicator: str, data_types: Dict[str, str]) -> List[Dict[str, Any]]:
        data_text = full_text
        if data_start_indicator:
            try:
                match = re.search(data_start_indicator, full_text, re.IGNORECASE | re.DOTALL)
                if match:
                    data_text = full_text[match.end():]
                    logger.info(f"Data start indicator found. Processing text after it.")
                else:
                    logger.warning("Data start indicator not found. Processing from beginning.")
            except re.error as e:
                 logger.error(f"Invalid data_start_indicator regex: {e}. Processing from beginning.")

        await self._update_progress("Splitting document into records...", 70)
        chunks = re.split(record_separator, data_text)
        
        all_records = []
        total_chunks = len(chunks)
        column_names = list(columns.keys())

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 5: continue
            
            progress_percent = 70 + int((i / total_chunks) * 25) if total_chunks > 0 else 95
            await self._update_progress(f"Extracting record {i+1}/{total_chunks}", progress_percent)

            raw_record = {}
            for col_name, pattern in columns.items():
                try:
                    match = re.search(pattern, chunk, re.DOTALL | re.IGNORECASE)
                    raw_record[col_name] = match.group(1).strip() if match and match.groups() else ""
                except re.error as e:
                    raw_record[col_name] = "REGEX_ERROR"
            
            cleaned_record = self._clean_record(raw_record, data_types)
            
            if any(val is not None and val != "" for val in cleaned_record.values()):
                if list(cleaned_record.values()) != column_names:
                    all_records.append(cleaned_record)
        
        logger.info(f"Extraction & cleaning finished. Found {len(all_records)} records.")
        await self._update_progress("Finalizing export...", 98)
        return all_records

    def export_to_excel(self, data: List[Dict[str, Any]], original_filename: str) -> str:
        if not data: raise ValueError("No data to export")
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(original_filename)[0]
        file_path = os.path.join(self.output_dir, f"output_{base_filename}_{timestamp}.xlsx")
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
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>PDF Extractor API</h1><p>index.html not found.</p>")

@app.post("/upload-and-analyze", response_model=ColumnAnalysisResponse)
async def upload_and_analyze_pdf(request: Request, file: UploadFile = File(...)):
    task_id = request.headers.get("X-Task-ID")
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing X-Task-ID header")
    
    progress_queues[task_id] = asyncio.Queue()
    processor = PDFProcessor(task_id)
    file_path = ""
    try:
        await processor._update_progress("Saving uploaded file...", 5)
        file_path = processor.save_uploaded_file(file)
        
        await processor._update_progress("Pre-processing PDF text...", 10)
        sample_text = processor._get_layout_aware_text(file_path, max_pages=5)
        if not sample_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")
        
        analysis_result = await processor.analyze_document_structure(sample_text)
        await processor._update_progress("Analysis complete.", 50)
        
        return ColumnAnalysisResponse(
            success=True,
            columns=analysis_result["columns"],
            record_separator=analysis_result["record_separator"],
            data_start_indicator=analysis_result["data_start_indicator"],
            data_types=analysis_result["data_types"],
            sample_data=analysis_result.get("sample_data", []),
            message="Analysis complete. Review patterns before extraction.",
            file_path=file_path
        )
    except Exception as e:
        logger.error(f"Analysis error for task {task_id}: {e}", exc_info=True)
        if file_path and os.path.exists(file_path): 
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/extract-data", response_model=DataExtractionResponse)
async def extract_data_from_pdf(request: Request, payload: ExtractionPayload):
    task_id = request.headers.get("X-Task-ID")
    if not task_id or task_id not in progress_queues:
        raise HTTPException(status_code=400, detail="Missing or invalid Task ID")

    processor = PDFProcessor(task_id)
    file_path = payload.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found on server: {file_path}")

    try:
        await processor._update_progress("Extracting and cleaning full text...", 60)
        full_text = processor._get_layout_aware_text(file_path)
        
        extracted_data = await processor.extract_data_with_patterns(
            full_text, 
            payload.columns, 
            payload.record_separator, 
            payload.data_start_indicator,
            payload.data_types
        )

        if not extracted_data:
            await processor._update_progress("No records found.", 100)
            raise HTTPException(status_code=400, detail="No data could be extracted. Try refining the regex or separators.")

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
        logger.error(f"Extraction error for task {task_id}: {e}", exc_info=True)
        await processor._update_progress("Error!", 100)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during extraction: {str(e)}")
    finally:
        if os.path.exists(file_path): 
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {file_path}")

@app.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    if task_id not in progress_queues:
        progress_queues[task_id] = asyncio.Queue()

    async def progress_generator():
        cleanup = False
        try:
            while True:
                try:
                    progress = await asyncio.wait_for(progress_queues[task_id].get(), timeout=60)
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress["percentage"] >= 100:
                        cleanup = True
                        break
                except asyncio.TimeoutError:
                    cleanup = True
                    break
        finally:
            if cleanup and task_id in progress_queues:
                del progress_queues[task_id]
                logger.info(f"Progress stream for task {task_id} closed and queue deleted.")

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    file_path = os.path.join("outputs", safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=safe_filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.post("/save-patterns")
async def save_patterns(payload: dict):
    patterns_dir = "patterns"
    os.makedirs(patterns_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(patterns_dir, f"pattern_{timestamp}.json")
    
    required_keys = ["columns", "record_separator", "data_start_indicator", "data_types"]
    if not all(k in payload for k in required_keys):
         raise HTTPException(status_code=400, detail="Invalid patterns format.")

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
    return JSONResponse({"success": True, "message": f"Patterns saved to {file_path}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)