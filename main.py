import os
import io
import json
import re
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import base64
from PIL import Image
import fitz  # PyMuPDF for better PDF handling
from werkzeug.utils import secure_filename
import dotenv

# Load environment variables from a .env file
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# Using a more capable model for better extraction
model = genai.GenerativeModel(
    'gemini-2.5-flash-lite-preview-06-17',
    generation_config=genai.types.GenerationConfig(
        temperature=0.1,
          # Increased for more complete extraction
    )
)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class PDFProcessor:
    def __init__(self):
        self.extracted_data = []
        self.common_headers_footers = set()
        
    def clean_text(self, text):
        """Clean and preprocess text content while preserving important information"""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve line breaks for structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # FIXED: Corrected character range - moved ! to proper position
        # Keep most characters, only remove truly problematic ones
        text = re.sub(r'[^\w\s@.,;:()[\]{}/"\'$%&*+=<>?!\-\n]', ' ', text)
        
        return text.strip()
    
    def detect_headers_footers(self, all_pages_text):
        """Detect common headers and footers across pages"""
        if len(all_pages_text) < 2:
            return
            
        first_lines = []
        last_lines = []
        
        for page_text in all_pages_text:
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            if len(lines) > 0:
                first_lines.append(lines[0])
            if len(lines) > 1:
                last_lines.append(lines[-1])
        
        # Only mark as header/footer if appears on most pages and is short
        for line in first_lines:
            if line and len(line) < 100 and first_lines.count(line) > len(all_pages_text) * 0.6:
                self.common_headers_footers.add(line)
        
        for line in last_lines:
            if line and len(line) < 100 and last_lines.count(line) > len(all_pages_text) * 0.6:
                self.common_headers_footers.add(line)
    
    def remove_headers_footers(self, text):
        """Remove detected headers and footers"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in self.common_headers_footers:
                # Only remove obvious page numbers, not all numbers
                if not (re.match(r'^\d{1,3}$', line) and len(line) <= 3):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_text_from_pdf(self, pdf_path, max_pages=None):
        """Extract and clean text from PDF with better page handling"""
        try:
            # Try PyMuPDF first for better text extraction
            pdf_document = fitz.open(pdf_path)
            pages_text = []
            raw_texts = []
            
            total_pages = len(pdf_document)
            
            # Handle page range properly
            if max_pages:
                pages_to_process = min(max_pages, total_pages)
            else:
                pages_to_process = total_pages
            
            print(f"Processing {pages_to_process} pages out of {total_pages} total pages")
            
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                raw_text = page.get_text()
                if raw_text:
                    raw_texts.append(raw_text)
            
            pdf_document.close()
            
            # Detect headers/footers across all pages
            self.detect_headers_footers(raw_texts)
            
            for page_num, raw_text in enumerate(raw_texts):
                try:
                    cleaned_text = self.clean_text(raw_text)
                    cleaned_text = self.remove_headers_footers(cleaned_text)
                    
                    # Keep all pages with any content
                    if cleaned_text.strip():
                        pages_text.append({
                            'page_number': page_num + 1,
                            'text': cleaned_text
                        })
                except Exception as clean_error:
                    print(f"Error cleaning text for page {page_num + 1}: {clean_error}")
                    # Add raw text if cleaning fails
                    if raw_text.strip():
                        pages_text.append({
                            'page_number': page_num + 1,
                            'text': raw_text.strip()
                        })
            
            return pages_text
            
        except Exception as e:
            print(f"Error with PyMuPDF, trying PyPDF2: {str(e)}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    pages_text = []
                    raw_texts = []
                    
                    total_pages = len(pdf_reader.pages)
                    
                    if max_pages:
                        pages_to_process = min(max_pages, total_pages)
                    else:
                        pages_to_process = total_pages
                    
                    print(f"PyPDF2 fallback: Processing {pages_to_process} pages out of {total_pages} total pages")
                    
                    for page_num in range(pages_to_process):
                        page = pdf_reader.pages[page_num]
                        raw_text = page.extract_text()
                        if raw_text:
                            raw_texts.append(raw_text)
                    
                    self.detect_headers_footers(raw_texts)
                    
                    for page_num, raw_text in enumerate(raw_texts):
                        try:
                            cleaned_text = self.clean_text(raw_text)
                            cleaned_text = self.remove_headers_footers(cleaned_text)
                            
                            if cleaned_text.strip():
                                pages_text.append({
                                    'page_number': page_num + 1,
                                    'text': cleaned_text
                                })
                        except Exception as clean_error:
                            print(f"Error cleaning text for page {page_num + 1}: {clean_error}")
                            if raw_text.strip():
                                pages_text.append({
                                    'page_number': page_num + 1,
                                    'text': raw_text.strip()
                                })
                    
                    return pages_text
            except Exception as fallback_error:
                print(f"Error with PyPDF2 fallback: {str(fallback_error)}")
                return []
    
    def extract_images_from_pdf(self, pdf_path, max_pages=None):
        """Extract images from PDF with better page handling"""
        try:
            pdf_document = fitz.open(pdf_path)
            pages_images = []
            
            total_pages = len(pdf_document)
            
            # Handle page range properly
            if max_pages:
                pages_to_process = min(max_pages, total_pages)
            else:
                pages_to_process = total_pages
            
            for page_num in range(pages_to_process):
                try:
                    page = pdf_document[page_num]
                    mat = fitz.Matrix(2, 2)  # Reduced matrix for better performance
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    pages_images.append({
                        'page_number': page_num + 1,
                        'image_data': base64.b64encode(img_data).decode('utf-8')
                    })
                except Exception as page_error:
                    print(f"Error extracting image for page {page_num + 1}: {page_error}")
                    continue
            
            pdf_document.close()
            return pages_images
        except Exception as e:
            print(f"Error extracting images from PDF: {str(e)}")
            return []
    
    def validate_and_clean_extracted_data(self, extracted_data, column_names):
        """Validate and clean extracted data with less aggressive filtering"""
        cleaned_data = {}
        
        # Only use specified columns, ignore any extra columns Gemini might add
        for col in column_names:
            value = extracted_data.get(col, "N/A")
            
            if isinstance(value, str):
                value = value.strip()
                # Less aggressive N/A detection
                if value.lower() in ['n/a', 'na', 'not available', 'none', 'null', '']:
                    value = "N/A"
                elif value == '-' and len(value) == 1:
                    value = "N/A"
                # Don't truncate long values - keep complete information
                value = re.sub(r'\s+', ' ', value)
            elif isinstance(value, (int, float)):
                value = str(value)
            
            cleaned_data[col] = value
        
        return cleaned_data
    
    def process_with_gemini(self, page_content, column_names, page_number):
        """Process page content with enhanced Gemini prompt for complete extraction"""
        try:
            # Enhanced prompt for complete information extraction
            prompt = f"""
            You are an expert data extraction specialist. Extract COMPLETE and COMPREHENSIVE information from page {page_number} of this PDF document.

            REQUIRED COLUMNS (extract for these ONLY): {', '.join(column_names)}

            CONTENT TO ANALYZE:
            {page_content.get('text', 'No text content available')}

            CRITICAL EXTRACTION RULES:
            1. Extract ALL relevant information for each specified column - DO NOT summarize or truncate
            2. Include complete sentences, full descriptions, and all details
            3. NEVER create new columns - only use the exact column names provided
            4. If no information exists for a column, return "N/A"
            5. Preserve all numbers, dates, names, and specific details
            6. Include multi-line content as single values with proper spacing
            7. Do not ignore any relevant information even if it seems lengthy

            RESPONSE FORMAT:
            Return ONLY a valid JSON object with the exact column names as keys:
            {{
                "{column_names[0] if column_names else 'example'}": "complete_extracted_information_here",
                "{column_names[1] if len(column_names) > 1 else 'example2'}": "complete_extracted_information_here"
            }}

            EXTRACT EVERYTHING RELEVANT - DO NOT LEAVE OUT DETAILS!
            """

            content_parts = [prompt]
            if 'image_data' in page_content:
                try:
                    img_data = base64.b64decode(page_content['image_data'])
                    image = Image.open(io.BytesIO(img_data))
                    content_parts.append(image)
                except Exception as img_error:
                    print(f"Error processing image for page {page_number}: {img_error}")

            response = model.generate_content(content_parts)
            response_text = response.text.strip()
            
            # Clean up response formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            response_text = response_text.replace('**', '').replace('*', '').strip()
            
            try:
                extracted_data = json.loads(response_text)
                
                # Handle list responses from Gemini
                if isinstance(extracted_data, list):
                    if extracted_data:
                        extracted_data = extracted_data[0]
                    else:
                        print(f"Warning: Gemini returned an empty list for page {page_number}.")
                        extracted_data = {}
                
                # Ensure we have a dictionary
                if not isinstance(extracted_data, dict):
                    print(f"Warning: Gemini response for page {page_number} was not a dictionary: {extracted_data}")
                    raise json.JSONDecodeError("Response was not a parsable JSON object.", response_text, 0)

                # Validate and clean data using only specified columns
                cleaned_data = self.validate_and_clean_extracted_data(extracted_data, column_names)
                cleaned_data['page_number'] = page_number
                
                return cleaned_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on page {page_number}: {e}")
                print(f"Response text: '{response_text}'")
                
                # Fallback regex extraction
                fallback_data = {}
                for col in column_names:
                    # More flexible regex patterns
                    patterns = [
                        rf'"{re.escape(col)}":\s*"([^"]*)"',
                        rf'"{re.escape(col)}":\s*([^,}}\n]*)',
                        rf'{re.escape(col)}:\s*"([^"]*)"',
                        rf'{re.escape(col)}:\s*([^,}}\n]*)'
                    ]
                    
                    match = None
                    for pattern in patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                        if match:
                            break
                    
                    if match:
                        fallback_data[col] = match.group(1).strip()
                    else:
                        fallback_data[col] = "N/A (parse failed)"
                
                fallback_data['page_number'] = page_number
                return fallback_data
                
        except Exception as e:
            print(f"Error processing with Gemini on page {page_number}: {str(e)}")
            error_data = {col: "Error" for col in column_names}
            error_data['page_number'] = page_number
            return error_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not api_key:
        return jsonify({'error': 'Server is not configured with a GOOGLE_API_KEY'}), 500
    
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        
        file = request.files['pdf_file']
        if file.filename == '' or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Please select a valid PDF file'}), 400
        
        column_names = [col.strip() for col in request.form.get('column_names', '').split(',') if col.strip()]
        if not column_names:
            return jsonify({'error': 'Please provide comma-separated column names'}), 400
        
        # Handle max_pages input properly
        max_pages = request.form.get('max_pages', '').strip()
        if max_pages and max_pages.isdigit():
            max_pages = int(max_pages)
            if max_pages <= 0:
                max_pages = None
        else:
            max_pages = None
        
        print(f"Processing with max_pages: {max_pages}")
        print(f"Extracting columns: {column_names}")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        processor = PDFProcessor()
        
        # Extract text and images with proper page limits
        text_pages = processor.extract_text_from_pdf(filepath, max_pages)
        if not text_pages:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract any text from the PDF. It might be an image-only PDF or corrupted.'}), 400
        
        image_pages = processor.extract_images_from_pdf(filepath, max_pages)
        
        # Combine text and image data
        combined_pages = []
        text_pages_dict = {p['page_number']: p for p in text_pages}
        image_pages_dict = {p['page_number']: p for p in image_pages}

        all_page_numbers = sorted(list(set(text_pages_dict.keys()) | set(image_pages_dict.keys())))

        for page_num in all_page_numbers:
            combined_content = {'page_number': page_num}
            if page_num in text_pages_dict:
                combined_content['text'] = text_pages_dict[page_num]['text']
            if page_num in image_pages_dict:
                combined_content['image_data'] = image_pages_dict[page_num]['image_data']
            combined_pages.append(combined_content)
        
        # Process each page with Gemini
        all_extracted_data = []
        for page_content in combined_pages:
            print(f"Processing page {page_content['page_number']}...")
            extracted_data = processor.process_with_gemini(
                page_content, column_names, page_content['page_number']
            )
            all_extracted_data.append(extracted_data)
        
        # Less aggressive filtering - keep more data
        meaningful_data = []
        for data in all_extracted_data:
            has_meaningful_data = False
            for key, value in data.items():
                if key != 'page_number':
                    if (value not in ["N/A", "Error", "N/A (parse failed)", ""] and 
                        str(value).strip() and 
                        len(str(value).strip()) > 0):
                        has_meaningful_data = True
                        break
            
            if has_meaningful_data:
                meaningful_data.append(data)
        
        if not meaningful_data:
            os.remove(filepath)
            return jsonify({'error': 'No relevant data found for the specified columns in the processed pages.'}), 400
        
        # Create DataFrame with proper column ordering
        df = pd.DataFrame(meaningful_data)
        column_order = [col for col in column_names if col in df.columns] + ['page_number']
        df = df[column_order]
        
        # Save outputs
        output_filename = f"extracted_data_{os.path.splitext(filename)[0]}"
        
        csv_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        excel_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extracted Data', index=False)
            worksheet = writer.sheets['Extracted Data']
            
            # Auto-adjust column widths
            for column_cells in worksheet.columns:
                max_length = 0
                column = column_cells[0].column_letter
                for cell in column_cells:
                    try:
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 100)  # Increased max width
                worksheet.column_dimensions[column].width = adjusted_width
        
        os.remove(filepath)
        
        # Generate summary statistics
        summary_stats = {
            'total_pages_processed': len(all_extracted_data),
            'pages_with_data': len(meaningful_data),
            'columns_extracted': len(column_names),
            'max_pages_requested': max_pages,
            'data_coverage': {
                col: f"{sum(1 for row in meaningful_data if row.get(col, 'N/A') not in ['N/A', 'Error', 'N/A (parse failed)', ''])} / {len(meaningful_data)}"
                for col in column_names
            }
        }
        
        return jsonify({
            'success': True,
            'message': f'Successfully extracted complete data from {len(meaningful_data)} pages.',
            'data': meaningful_data[:5],  # Show first 5 rows as preview
            'total_rows': len(meaningful_data),
            'output_filename': output_filename,
            'summary': summary_stats
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        print(f"An unexpected error occurred in /upload: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500

@app.route('/download/<format>/<filename>')
def download_file_route(format, filename):
    try:
        safe_filename = secure_filename(filename)
        if format == 'csv':
            file_path = os.path.join(OUTPUT_FOLDER, f"{safe_filename}.csv")
            return send_file(file_path, as_attachment=True)
        elif format == 'excel':
            file_path = os.path.join(OUTPUT_FOLDER, f"{safe_filename}.xlsx")
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Invalid format specified'}), 400
    except FileNotFoundError:
        return jsonify({'error': 'File not found. It may have been cleaned up.'}), 404
    except Exception as e:
        print(f"Error during download: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.getenv('GOOGLE_API_KEY'):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL WARNING: GOOGLE_API_KEY not set         !!!")
        print("!!! The application will not be able to process files. !!!")
        print("!!! Create a .env file with GOOGLE_API_KEY='your-key'  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)