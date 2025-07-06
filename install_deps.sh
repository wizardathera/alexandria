#!/bin/bash

# Alexandria App - Dependency Installation Script
# This script installs all required dependencies including system packages for PDF processing

set -e  # Exit on any error

echo "🚀 Installing Alexandria App Dependencies..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: No virtual environment detected!"
    echo "Please activate your virtual environment first:"
    echo "source .venv/bin/activate  # or your venv path"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Install system dependencies (Ubuntu/Debian)
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libmagic1 \
        libmagic-dev \
        python3-dev \
        build-essential
elif command -v yum &> /dev/null; then
    echo "Detected RHEL/CentOS system"
    sudo yum install -y \
        poppler-utils \
        tesseract \
        file-devel \
        python3-devel \
        gcc
elif command -v brew &> /dev/null; then
    echo "Detected macOS system"
    brew install \
        poppler \
        tesseract \
        libmagic
else
    echo "⚠️  Warning: Unknown system. Please install these packages manually:"
    echo "  - poppler-utils (for PDF processing)"
    echo "  - tesseract-ocr (for OCR)"
    echo "  - libmagic (for file type detection)"
fi

# Upgrade pip
echo "📈 Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional unstructured dependencies
echo "📄 Installing additional PDF processing dependencies..."
pip install "unstructured[pdf]"

# Install OCR dependencies
echo "🔍 Installing OCR dependencies for scanned PDFs..."
pip install pytesseract pdf2image pillow

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import langchain_community.document_loaders
import unstructured
import pypdf
import chromadb
import openai
print('✅ All core dependencies imported successfully!')
"

# Test PDF processing specifically
echo "📝 Testing PDF processing capability..."
python -c "
try:
    from unstructured.partition.pdf import partition_pdf
    print('✅ PDF processing (partition_pdf) available!')
except ImportError as e:
    print(f'❌ PDF processing not available: {e}')
    print('You may need to install additional dependencies manually.')
"

# Test OCR processing
echo "🔍 Testing OCR processing capability..."
python -c "
try:
    import pytesseract
    import pdf2image
    from PIL import Image
    print('✅ OCR Python packages available!')
    
    # Test tesseract binary
    try:
        version = pytesseract.get_tesseract_version()
        print(f'✅ Tesseract OCR system binary available: {version}')
    except Exception as e:
        print(f'⚠️ Tesseract OCR system binary not found: {e}')
        print('   Install with: sudo apt-get install tesseract-ocr (Ubuntu/Debian)')
        print('   or: brew install tesseract (macOS)')
        
except ImportError as e:
    print(f'⚠️ OCR packages not fully available: {e}')
    print('   OCR is optional but enables processing of scanned PDFs')
"

# Run comprehensive test
echo ""
echo "🧪 Running comprehensive pipeline test..."
python test_ingestion_pipeline.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Start the server: uvicorn src.main:app --reload"
echo "3. Visit http://127.0.0.1:8000/docs to test the API"
echo ""
echo "Troubleshooting:"
echo "- If PDF processing still fails, try: pip install 'unstructured[pdf]' --force-reinstall"
echo "- For Ubuntu/Debian, you may need: sudo apt-get install python3-distutils"
echo "- Check the README.md for additional setup instructions"