import os
import io
import json
import re
import logging
import time
import base64
from datetime import datetime
from typing import Dict, List, Set

import atexit
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import google.generativeai as genai
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
import dotenv

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    logger.info("Google API configured successfully")
else:
    logger.critical("FATAL: GOOGLE_API_KEY not found. The application will not work.")

model = genai.GenerativeModel('gemini-2.5-flash')

for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# --- Core PDF Processing Logic ---
class EnhancedPDFProcessor:
    def _parse_page_range(self, range_str: str, total_pages: int) -> Set[int]:
        if not range_str:
            return set(range(1, total_pages + 1))
        pages_to_process = set()
        for part in range_str.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    pages_to_process.update(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid range: '{part}'")
            elif part.isdigit():
                pages_to_process.add(int(part))
        return {p for p in pages_to_process if 0 < p <= total_pages}

    def extract_text_from_pdf(self, pdf_path: str, page_range_str: str) -> List[Dict]:
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            pages_to_process = self._parse_page_range(page_range_str, total_pages)
            logger.info(f"Processing pages: {sorted(list(pages_to_process))}")

            extracted_pages = []
            for page_num in pages_to_process:
                page = pdf_document[page_num - 1]
                text = page.get_text("text")
                if text.strip():
                    extracted_pages.append({'page_number': page_num, 'text': text})
            pdf_document.close()
            return extracted_pages
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return []

    def _parse_gemini_response(self, response_text: str) -> Dict:
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
            else:
                raise ValueError("No valid JSON object found in the response.")
        return json.loads(json_str)

    ### FIX 1: ADDED AUTOMATIC RETRY LOGIC ###
    def process_page_with_gemini(self, page_content: Dict, column_names: List[str]) -> Dict:
        page_number = page_content['page_number']
        prompt = f"""
        You are an expert data extraction AI. From the content of page {page_number}, extract the information for these columns: {', '.join(column_names)}.

        **CRITICAL INSTRUCTIONS:**
        1.  Your entire response MUST be ONLY the JSON object.
        2.  Do NOT include markdown fences like ```json or ```.
        3.  Do NOT add any introductory text, concluding text, or any explanations.
        4.  If no information is found for a column, you MUST return "N/A" for that column's value.
        5.  Ignore all headers, footers, and page numbers in the document content.

        **DOCUMENT CONTENT (Page {page_number}):**
        ---
        {page_content.get('text', '')}
        ---

        **REQUIRED JSON OBJECT:**
        {{
            {', '.join([f'"{col}": "..."' for col in column_names])}
        }}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                extracted_data = self._parse_gemini_response(response.text)
                
                validated_data = {col: str(extracted_data.get(col, "N/A")).strip() for col in column_names}
                validated_data['page_number'] = page_number # Keep for internal use
                return validated_data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_number}: {e}")
                if attempt == max_retries - 1: # If this was the last attempt
                    logger.error(f"All {max_retries} attempts failed for page {page_number}. Skipping.")
                    # Return a clear error dictionary for this page
                    error_data = {col: f"ERROR: Processing failed after {max_retries} attempts" for col in column_names}
                    error_data['page_number'] = page_number
                    return error_data
                time.sleep(1) # Wait a second before retrying

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not api_key:
        return jsonify({'error': 'Server is not configured with a GOOGLE_API_KEY'}), 500

    try:
        file = request.files.get('pdf_file')
        if not file or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'A valid PDF file is required.'}), 400

        column_names = [col.strip() for col in request.form.get('column_names', '').split(',') if col.strip()]
        if not column_names:
            return jsonify({'error': 'Column names are required.'}), 400

        page_range_str = request.form.get('page_range', '').strip()

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        processor = EnhancedPDFProcessor()
        text_pages = processor.extract_text_from_pdf(filepath, page_range_str)

        if not text_pages:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from the specified pages.'}), 400

        all_extracted_data = [processor.process_page_with_gemini(page, column_names) for page in text_pages]
        
        # Check if any pages completely failed after retries
        failed_pages = [row['page_number'] for row in all_extracted_data if "ERROR:" in str(row.values())]
        if len(failed_pages) == len(all_extracted_data):
             os.remove(filepath)
             return jsonify({'error': f'Processing failed for all pages. Please check the document or try again.'}), 400

        meaningful_data = [row for row in all_extracted_data if any(val not in ["N/A", ""] and "ERROR:" not in str(val) for key, val in row.items() if key != 'page_number')]

        if not meaningful_data:
            os.remove(filepath)
            return jsonify({'error': 'No relevant data found for the specified columns.'}), 400

        df = pd.DataFrame(meaningful_data)
        
        ### FIX 2: REMOVED UNWANTED COLUMNS FROM FINAL OUTPUT ###
        # Ensure only the requested columns are in the final output file
        final_columns = [col for col in column_names if col in df.columns]
        df_final = df[final_columns]

        output_base = f"extracted_{os.path.splitext(filename)[0]}_{timestamp}"
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_base}.csv")
        df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')

        data_coverage = {col: int(df_final[col].ne('N/A').sum()) for col in final_columns}
        summary = {
            "total_pages_processed": len(text_pages),
            "pages_with_data": len(df_final),
            "columns_extracted": len(final_columns),
            "data_coverage": data_coverage,
            "failed_pages": failed_pages # Also report which pages failed
        }

        os.remove(filepath)

        return jsonify({
            'success': True,
            'output_filename': output_base,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500

@app.route('/download/<format>/<filename>')
def download_file_route(format, filename):
    safe_filename = secure_filename(filename)
    if format == 'csv':
        path = os.path.join(app.config['OUTPUT_FOLDER'], f"{safe_filename}.csv")
        mimetype = 'text/csv'
    elif format == 'excel':
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{safe_filename}.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Original CSV file not found.'}), 404
        df = pd.read_csv(csv_path)
        path = os.path.join(app.config['OUTPUT_FOLDER'], f"{safe_filename}.xlsx")
        df.to_excel(path, index=False, sheet_name='Extracted Data')
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        return jsonify({'error': 'Invalid format'}), 400

    if not os.path.exists(path):
        return jsonify({'error': 'File not found.'}), 404
    return send_file(path, as_attachment=True, mimetype=mimetype)

@app.route('/preview/<filename>')
def preview_file(filename):
    try:
        safe_filename = secure_filename(filename)
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{safe_filename}.csv")
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': 'Preview file not found.'}), 404

        df = pd.read_csv(csv_path)
        html_table = df.to_html(classes='table table-striped table-hover', border=0, index=False, table_id='dataTable')

        return jsonify({
            'success': True,
            'html_table': html_table,
            'shape': df.shape,
            'columns': df.columns.tolist()
        })
    except Exception as e:
        logger.error(f"Preview error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

def cleanup_old_files():
    cutoff = time.time() - (24 * 3600)
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                logger.info(f"Cleaned up old file: {path}")

if __name__ == '__main__':
    if not api_key:
        print("="*60)
        print("CRITICAL ERROR: GOOGLE_API_KEY is not set.")
        print("Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        print("="*60)
    else:
        atexit.register(cleanup_old_files)
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)