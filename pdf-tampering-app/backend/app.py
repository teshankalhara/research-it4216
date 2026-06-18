"""
Flask API Server for PDF Tampering Detection
Compatible with Python 3.14+
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import analyze_pdf

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if file is PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'PDF Tampering Detector'})

@app.route('/detect', methods=['POST'])
def detect_tampering():
    """
    Detect tampering in uploaded PDF
    Expects: multipart/form-data with 'file' field
    Returns: JSON with detection results
    """
    try:
        # Check if file exists
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        temp_filename = f"temp_{unique_id}_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        file.save(temp_path)
        
        # Analyze PDF
        result = analyze_pdf(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Add filename to result
        result['filename'] = filename
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-detect', methods=['POST'])
def batch_detect():
    """
    Detect tampering in multiple PDFs
    Expects: multipart/form-data with multiple 'files' fields
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_id = str(uuid.uuid4())[:8]
                temp_filename = f"temp_{unique_id}_{filename}"
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                
                file.save(temp_path)
                result = analyze_pdf(temp_path)
                result['filename'] = filename
                results.append(result)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("PDF TAMPERING DETECTION API SERVER")
    print("="*60)
    print(f"Server running at: http://localhost:5000")
    print(f"Health check: http://localhost:5000/health")
    print(f"Detection endpoint: POST http://localhost:5000/detect")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)