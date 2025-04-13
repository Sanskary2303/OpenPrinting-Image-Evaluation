# OpenPrinting Image Quality Evaluation

A comprehensive toolkit for evaluating print/scan job processing by comparing original and processed document images.

## Project Overview

This project implements automated testing for print quality assurance by evaluating images before and after processing through a print pipeline. It uses computer vision and image processing techniques to detect differences, verify text preservation, and ensure proper rendering of graphical elements.

The system supports extensive testing with various document types including text in different languages, multiple fonts, images, and complex layouts. Results are presented in detailed reports with visual comparison tools.

## Features

- **Comprehensive Image Comparison** 
    - SSIM (Structural Similarity Index) analysis
    - Feature detection and matching using ORB
    - Edge detection and comparison
    - Color fidelity analysis with histogram comparison and Delta-E metrics
    - Visual difference heatmaps
    - Edge quality analysis with multi-scale edge detection

- **Page Integrity Verification**
    - Multi-page document processing
    - Page completeness verification
    - Page order validation
    - Missing and duplicate page detection
    - Page-by-page quality comparison

- **Text Preservation Testing**
    - OCR-based text extraction
    - Text similarity analysis
    - Font rendering verification

- **Diverse Test Documents**
    - Text in multiple languages (English, Spanish, French, German, etc.)
    - Various fonts and sizes
    - Pure image documents
    - Mixed content (text and images)
    - Complex layouts with tables and columns
    - Multi-page documents with orientation variations

- **Detailed Reporting**
    - HTML reports with visual comparisons
    - CSV summary files
    - JSON result data for each test
    - Pass/fail determination based on customizable thresholds
    - Visual page integrity maps for multi-page documents

## Components

### Core Testing Scripts

- **`test_pipeline.py`**: Basic single-document test pipeline
- **`test_document_pipeline.py`**: Comprehensive testing with multiple document types
- **`multipage_test.py`**: Multi-page document testing with page integrity verification
- **`enhanced_comparison.py`**: Advanced image comparison techniques

### Document Generation

- **`generate_pdf.py`**: Simple PDF generation for basic tests
- **`generate_test_documents.py`**: Creates a variety of test documents including multi-page with page markers

### Utilities

- **`extract_text.py`**: Standalone tool for OCR text extraction
- **`compare_image.py`**: Simple image comparison utility
- **`clean_workspace.py`**: Python script for cleaning the workspace
- **`cleanup.sh`**: User-friendly shell wrapper for cleanup operations

## Installation Requirements

```bash
pip install reportlab pillow opencv-python scikit-image pytesseract pandas colormath matplotlib diffoscope PyMuPDF
```

Additionally, ensure you have the following system packages:

- Ghostscript (for PDF to image conversion)
- CUPS (for print processing)
- Tesseract OCR (for text extraction)

## Usage

### Basic Testing

To run a basic test with a simple document:

```bash
python test_pipeline.py
```

This will:

1. Generate a simple PDF
2. Process it through the print pipeline
3. Compare the original and processed versions
4. Display similarity metrics
5. Extract and compare text

### Comprehensive Testing

For testing with multiple document types:

```bash
python test_document_pipeline.py
```
This will:

1. Generate various test documents
2. Process each document through the print pipeline
3. Perform detailed comparison of each document
4. Generate an HTML report with results

### Multi-page Document Testing

For testing multi-page documents with page integrity verification:

```bash
python multipage_test.py
```

This will:

1. Generate multi-page test documents
2. Process documents through the print pipeline
3. Extract and compare individual pages
4. Check for missing or reordered pages
5. Generate visualizations of page mapping and quality metrics

### Workspace Cleanup

After testing, clean up your workspace:

```bash
# Using the shell script (interactive)
./cleanup.sh

# Common operations
./cleanup.sh quick    # Remove temporary files
./cleanup.sh full     # Full cleanup (keep reports)
./cleanup.sh backup   # Backup reports then clean

# Direct Python usage
python clean_workspace.py --temp-files --keep-reports
```

## Customization

- Adjust quality thresholds in the test scripts
- Modify document generation in [`generate_test_documents.py`](generate_test_documents.py)
- Add new comparison techniques in [`enhanced_comparison.py`](enhanced_comparison.py)
- Update page integrity detection parameters in the [`page_integrity_comparison`](enhanced_comparison.py) method

## Structure

```
├── test_pipeline.py            # Basic testing pipeline
├── test_document_pipeline.py   # Comprehensive testing
├── multipage_test.py           # Multi-page document testing
├── enhanced_comparison.py      # Advanced image comparison
├── generate_pdf.py             # Basic PDF generation
├── generate_test_documents.py  # Test document creation
├── extract_text.py             # OCR utility
├── compare_image.py            # Basic image comparison
├── clean_workspace.py          # Cleanup utility
├── cleanup.sh                  # Shell wrapper for cleanup
├── PDF/                        # Output directory for CUPS
└── test_documents/             # Generated during testing
    ├── *.pdf                   # Test documents
    ├── *.png                   # Rendered images
    ├── *_results.json          # Individual test results
    ├── test_summary.csv        # Results summary
    ├── test_report.html        # Visual HTML report
    └── multipage/              # Multi-page test results
        ├── complete_doc/       # Complete document results
        ├── reordered/          # Reordered page tests
        └── missing/            # Missing page tests
```

## Major Features

### Edge Quality Analysis

The edge quality analysis uses multi-scale edge detection to evaluate how well fine details and contours are preserved in the printing process. This helps identify issues with:

- Loss of fine details
- Edge blurring or sharpening
- Contour discontinuities
- Line width variations

### Page Integrity Analysis

The page integrity verification system analyzes multi-page documents to detect common errors:

- Missing pages
- Duplicated pages
- Reordered pages
- Page orientation errors

The system uses feature matching to identify corresponding pages regardless of minor distortions and generates visualizations showing page mapping between reference and processed documents.

## Credits

This project builds on work from the GNOME openQA project and uses free software components including OpenCV, diffoscope, PyMuPDF, and others for image comparison and analysis.