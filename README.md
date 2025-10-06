# OpenPrinting Image Quality Evaluation

A comprehensive toolkit for evaluating print/scan job processing by comparing original and processed document images.

## Prerequisites & Setup

### Virtual PDF Printer Setup

You need a virtual PDF printer that saves output to files instead of physical printing.

#### **For Fedora/RHEL:**

```bash
# Install cups-pdf
sudo dnf install cups-pdf

# This creates a printer queue named "Cups-PDF"
# Verify it exists:
lpstat -p Cups-PDF
# Output: printer Cups-PDF is idle

# Output files will be saved in: ~/PDF/
```

#### **For Ubuntu/Debian:**

```bash
# Install cups-pdf
sudo apt-get install cups-pdf

# This creates a printer queue named "PDF"
# Verify it exists:
lpstat -p PDF
# Output: printer PDF is idle

# Output files will be saved in: ~/PDF/
```

#### **Finding Your Printer Queue Name:**

```bash
# List all available printers
lpstat -p

# Common names:
# - "Cups-PDF" (Fedora/RHEL with cups-pdf)
# - "PDF" (Ubuntu/Debian with cups-pdf)
# - "Virtual_PDF_Printer" (custom manual setup)
```

**Important:** Use your actual printer queue name in the commands below. Replace `Cups-PDF` with your printer name (e.g., `PDF` on Ubuntu).

## Quick Start - See Results in 60 Seconds

**New to this project? Start here for instant impressive results:**

### HOWTO-1: Print Quality Analysis (2 minutes)

```bash
# Use your printer queue name
python print_quality_pipeline.py Cups-PDF    # Fedora/RHEL
# OR
python print_quality_pipeline.py PDF         # Ubuntu/Debian

# See quality scores, visual differences, problem detection
```

### HOWTO-2: CUPS Filter Detective (3 minutes)  

```bash
# Use your printer queue name
python filter_chain_test_pipeline.py Cups-PDF --fast-mode

# X-ray vision: see what each filter does to your document
```

### HOWTO-3: Document Testing Suite (5 minutes)

```bash
python test_document_pipeline.py  
# Test 15+ document types, get comprehensive quality report
```

### HOWTO-4: Image Analysis Demo (3 minutes)

```bash
python example_usage.py
# Explore 11 advanced image analysis algorithms
```

**📚 [Complete HOWTOs with examples →](quick_start_examples/)**

**✅ Verify everything works:** `python verify_setup.py`

---

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
  - Advanced algorithms: density detection, rotation analysis, noise analysis, texture analysis, and geometric analysis

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
- **`print_quality_pipeline.py`**: Complete end-to-end print quality testing pipeline
- **`filter_chain_test_pipeline.py`**: Isolated CUPS filter chain correctness testing
- **`enhanced_comparison.py`**: Advanced image comparison techniques
- **`printer_info.py`**: CUPS filter chain analysis and execution

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
- cups-filters (for accessing CUPS filter components)

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

### Comprehensive Print Quality Testing

For complete end-to-end print quality assessment:

```bash
python print_quality_pipeline.py
```

This comprehensive pipeline will:

1. **Automatic Printer Detection**: Discover available CUPS printers and their capabilities
2. **Test Document Generation**: Create diverse test images with various characteristics (text, images, patterns, colors)
3. **Filter Chain Analysis**: Analyze and display the complete CUPS filter chain for each printer
4. **Automated Processing**: Process all test documents through each detected printer
5. **Advanced Comparison**: Apply comprehensive image analysis algorithms including:
   - SSIM and feature matching
   - Edge quality analysis
   - Color fidelity assessment
   - Density and texture analysis
   - Rotation and geometric distortion detection
   - Noise analysis
6. **Detailed Reporting**: Generate HTML reports with visual comparisons and quantitative metrics
7. **Quality Scoring**: Provide overall quality scores and pass/fail determinations

**Use Cases**: 
- Production print quality validation
- Printer performance benchmarking
- Comprehensive quality assurance testing
- Print system validation across multiple printers

### Filter Chain Correctness Testing

For testing CUPS filter chain correctness without physical printing:

```bash
python filter_chain_test_pipeline.py
```

This specialized pipeline will:

1. **Filter Chain Isolation**: Test CUPS filter processing independent of physical hardware
2. **Systematic Test Image Generation**: Create targeted test images for specific filter scenarios:
   - Pure text documents
   - High-resolution images
   - Color and grayscale content
   - Various DPI settings
   - Complex layouts
3. **Direct Filter Processing**: Process images through CUPS filters without physical printing
4. **Digital Comparison**: Compare filter outputs using pixel-perfect analysis
5. **Filter-Specific Validation**: Identify issues specific to individual filters in the chain
6. **Regression Testing**: Detect filter processing changes between CUPS versions

**Use Cases**:
- CUPS filter development and debugging
- Filter chain regression testing
- Digital print processing validation
- Isolating software issues from hardware problems
- CI/CD integration for CUPS testing

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

### Image Processing Algorithms Testing

To test and demonstrate the image processing algorithms:

```bash
python test_algorithms.py
```

This will:

1. Generate test images with various characteristics
2. Run all implemented algorithms (density, rotation, noise, texture, geometric)
3. Display quantitative results for each algorithm
4. Generate visualization outputs for analysis

For a practical example with existing images:

```bash
python example_usage.py
```

### Filter Chain Analysis and Testing

To analyze and test CUPS filter chains:

```python3
>>> from printer_info import get_filter_chain, run_cups_filter_chain, get_comprehensive_printer_info, get_available_printers
>>> printers = get_available_printers()
>>> printer_name = printers[0]
>>> get_comprehensive_printer_info(printer_name)  # View detailed printer information
>>> filter_chain = get_filter_chain(printer_name, input_mime_type='application/pdf')  # Identify filter chain
>>> print(filter_chain)
>>> # Test a filter chain with direct execution:
>>> run_cups_filter_chain(
...     printer_name=printer_name, 
...     input_file_path="sample.pdf", 
...     output_file_path="output_raster.data",
...     custom_filters=['pdftopdf', 'pdftoraster'],
...     filter_options={'pdftoraster': 'ColorModel=RGB Resolution=300'}
... )
```

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

## Pipeline Comparison Guide

### When to Use `print_quality_pipeline.py`

**Best for:**
- Production print quality validation
- Testing complete printing systems (software + hardware)
- Benchmarking printer performance
- Real-world print quality assessment
- Testing with physical printers and scanners

**What it tests:**
- Complete print-to-paper-to-scan workflow
- Hardware-specific rendering quality
- Paper and ink interactions
- Scanner quality and calibration
- End-to-end system performance

**Output:** Physical print quality metrics with real-world variability

### When to Use `filter_chain_test_pipeline.py`

**Best for:**
- CUPS filter development and debugging
- Software-only testing without hardware dependencies
- Regression testing for CUPS updates
- CI/CD integration
- Isolating filter chain issues from hardware problems

**What it tests:**
- Pure digital filter processing accuracy
- CUPS software pipeline correctness
- Filter chain configuration validation
- Pixel-perfect digital processing
- Software regression detection

**Output:** Precise digital processing validation with minimal variability

## Customization

- Adjust quality thresholds in the test scripts
- Modify document generation in [`generate_test_documents.py`](generate_test_documents.py)
- Add new comparison techniques in [`enhanced_comparison.py`](enhanced_comparison.py)
- Update page integrity detection parameters in the [`page_integrity_comparison`](enhanced_comparison.py) method

## Structure

```text
├── test_pipeline.py            # Basic testing pipeline
├── test_document_pipeline.py   # Comprehensive testing
├── multipage_test.py           # Multi-page document testing
├── print_quality_pipeline.py   # Complete end-to-end print quality testing
├── filter_chain_test_pipeline.py # CUPS filter chain correctness testing
├── enhanced_comparison.py      # Advanced image comparison
├── test_algorithms.py          # Test script for image processing algorithms
├── test_color_detection.py     # Test script specifically for color/monochrome detection
├── example_usage.py            # Example usage of image processing algorithms
├── generate_pdf.py             # Basic PDF generation
├── generate_test_documents.py  # Test document creation
├── extract_text.py             # OCR utility
├── compare_image.py            # Basic image comparison
├── clean_workspace.py          # Cleanup utility
├── cleanup.sh                  # Shell wrapper for cleanup
├── printer_info.py             # CUPS filter chain analysis and testing
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

### Filter Chain Analysis

The filter chain analysis system provides a way to inspect, understand, and test the exact sequence of filters that CUPS would use for a specific printer and document type:

- **Filter Chain Detection**: Query the filters that would be used based on printer PPD configurations
- **Direct Filter Execution**: Manually control and test specific filter sequences for precise debugging
- **Troubleshooting Tools**: Compare output from different filters to isolate issues in complex print pipelines
- **Printer Information Collection**: Gather comprehensive details about available printers and their capabilities
- **PPD Analysis**: Extract filter information directly from printer PPD files when needed

### Filter Chain Correctness Testing

The `filter_chain_test_pipeline.py` provides specialized testing focused on CUPS filter processing accuracy:

**Key Features:**

- **Hardware-Independent Testing**: Tests filter chains without requiring physical printers or scanners
- **Systematic Test Generation**: Creates targeted test images designed to reveal specific filter processing issues
- **Digital Precision**: Performs pixel-perfect comparisons without hardware-induced variations
- **Filter Isolation**: Identifies which specific filters in a chain may be causing processing problems
- **Automated Test Suite**: Runs comprehensive tests across multiple document types and configurations

**Test Categories:**
- **Text Rendering**: Validates text processing and font rendering accuracy
- **Image Processing**: Tests color space conversion, resolution scaling, and image compression
- **Layout Preservation**: Ensures document structure and positioning remain intact
- **Color Fidelity**: Verifies color accuracy and grayscale conversion
- **DPI Handling**: Tests resolution scaling and DPI-specific processing

**Reporting:**
- Detailed filter-by-filter processing logs
- Visual diff outputs showing exact processing changes
- Quantitative metrics for each processing step
- Pass/fail determination for regression testing
- Export-friendly formats for CI/CD integration

## Image Processing Algorithms

The following basic image processing algorithms have been implemented in the `enhanced_comparison.py` module:

### 1. Image Density Detection

**Purpose**: Analyzes the distribution and density of content within an image to understand layout characteristics.

**Metrics Calculated**:
- **Overall Content Density**: Ratio of non-white pixels to total pixels
- **Text Density**: Estimated density of text-like structures using morphological operations
- **Image Content Density**: Density of larger image elements
- **White Space Ratio**: Proportion of white/empty space
- **Density Variance**: Measure of how uniformly content is distributed
- **Density Entropy**: Information-theoretic measure of density distribution
- **Dense Clusters**: Number of highly dense content regions

**Use Cases**: Document layout analysis, print quality assessment, content distribution evaluation, scan quality verification

**Output**: Density heatmap visualization (`density_heatmap.png`) and quantitative metrics

### 2. Rotation Detection

**Purpose**: Detects if an image is rotated and estimates the rotation angle using multiple methods.

**Methods Used**:
1. **Hough Line Transform**: Detects straight lines and calculates their angles
2. **Text Line Detection**: Specifically looks for horizontal text lines
3. **Principal Component Analysis (PCA)**: Finds the main direction of edge features

**Metrics Calculated**:
- **Rotation Angle**: Estimated rotation in degrees (-45° to +45°)
- **Rotation Confidence**: Reliability of the rotation detection
- **Is Rotated**: Boolean indicating if significant rotation is detected
- **Method-specific Angles**: Results from each detection method

**Use Cases**: Document orientation correction, scan quality assessment, automated document processing, quality control for printing/scanning

**Output**: Rotation detection visualization (`rotation_detection.png`) and angle measurements

### 3. Noise Analysis

**Purpose**: Analyzes various types of noise present in the image to assess quality.

**Noise Types Detected**:
1. **Gaussian Noise**: General background noise using Laplacian variance
2. **Salt and Pepper Noise**: Isolated bright/dark pixels
3. **General Noise Level**: Difference from median-filtered version

**Metrics Calculated**:
- **Noise Variance**: Laplacian-based noise estimation
- **Noise Level**: Average noise magnitude
- **Salt/Pepper Counts**: Number of isolated noise pixels
- **Signal-to-Noise Ratio (SNR)**: Quality metric in dB
- **Noise Entropy**: Distribution characteristics of noise
- **Noise Quality**: Categorical assessment (low/medium/high)

**Use Cases**: Image quality assessment, scanner performance evaluation, print quality analysis, preprocessing requirements determination

**Output**: Noise visualization (`noise_analysis.png`) and comprehensive noise metrics

### 4. Texture Analysis

**Purpose**: Analyzes texture characteristics to understand surface properties and detail preservation.

**Methods Used**:
1. **Local Binary Patterns (LBP)**: Texture descriptor for local patterns
2. **Contrast Measures**: Local standard deviation analysis
3. **Gradient Analysis**: Edge magnitude and direction
4. **Roughness Calculation**: Overall texture variation

**Metrics Calculated**:
- **LBP Uniformity**: Consistency of local binary patterns
- **Contrast Measure**: Average local contrast
- **Texture Energy**: Overall edge/gradient magnitude
- **Direction Uniformity**: Consistency of texture orientation
- **Roughness**: Surface variation measure
- **Texture Complexity**: Categorical assessment

**Use Cases**: Material surface analysis, print texture quality, detail preservation assessment, surface defect detection

**Output**: LBP texture visualization (`texture_lbp.png`) and contrast visualization (`texture_contrast.png`)

### 5. Geometric Analysis

**Purpose**: Analyzes geometric properties and distortions in the image.

**Analyses Performed**:
1. **Shape Detection**: Identifies rectangular and circular objects
2. **Symmetry Analysis**: Horizontal and vertical symmetry
3. **Line Straightness**: Quality of linear elements
4. **Aspect Ratio**: Overall image proportions

**Metrics Calculated**:
- **Aspect Ratio**: Width to height ratio
- **Average Rectangularity**: How well rectangular objects preserve their shape
- **Horizontal/Vertical Symmetry**: SSIM-based symmetry scores
- **Line Straightness**: Quality of linear elements
- **Geometric Quality**: Overall geometric preservation score

**Use Cases**: Geometric distortion detection, shape preservation assessment, perspective correction evaluation, printing alignment verification

### 6. Color/Monochrome Detection

**Purpose**: Detects whether images are color or monochrome (grayscale) and can distinguish RGB-encoded grayscale from RGB-encoded color images.

**Detection Methods**:
1. **Channel Variance Analysis**: Examines differences between RGB channels
2. **Saturation Analysis**: Analyzes HSV saturation values
3. **Correlation Analysis**: Measures correlation between RGB channels
4. **Lab Color Space Analysis**: Examines color variance in perceptually uniform space
5. **Identical Channel Counting**: Counts pixels where R=G=B

**Metrics Calculated**:
- **Image Type Classification**: Color vs Grayscale for each image
- **Detection Confidence**: Reliability of the classification
- **Channel Differences**: Maximum and mean differences between RGB channels
- **Saturation Statistics**: Mean and maximum saturation values
- **Color Pixel Ratio**: Proportion of pixels with significant color content
- **Channel Correlation**: Correlation coefficients between RGB channels
- **Color Preservation Score**: Whether color type is maintained between original and processed

**Use Cases**: Color fidelity assessment, grayscale conversion detection, color space analysis, print color mode verification

**Output**: Detailed classification and confidence metrics for both images

### Algorithm Integration

The new algorithms are integrated into the overall quality assessment with weighted contributions:

```python
quality_score = (
    ssim * 0.15 +                          # Basic similarity
    edge_quality * 0.12 +                  # Edge preservation
    histogram_similarity * 0.08 +          # Color preservation
    content_density * 0.05 +               # Layout preservation
    rotation_score * 0.05 +                # Orientation correctness
    noise_score * 0.05 +                   # Noise absence
    texture_score * 0.05 +                 # Texture preservation
    geometric_score * 0.05 +               # Shape preservation
    color_preservation_score * 0.04 +      # Color type preservation
    # ... other factors
)
```

### Algorithm Usage Example

```python
from enhanced_comparison import ImageComparator

# Initialize comparator
comparator = ImageComparator('original.png', 'processed.png')

# Run individual algorithms
density_results = comparator.image_density_detection()
rotation_results = comparator.rotation_detection()
noise_results = comparator.noise_analysis()
texture_results = comparator.texture_analysis()
geometric_results = comparator.geometric_analysis()
color_results = comparator.color_monochrome_detection()

# Or run all algorithms together
all_results = comparator.run_all_comparisons()
```

### Algorithm Visualization Outputs

Each algorithm generates visualization files to help understand the analysis:

- `density_heatmap.png`: Color-coded density distribution
- `rotation_detection.png`: Detected lines used for rotation analysis
- `noise_analysis.png`: Noise distribution visualization
- `texture_lbp.png`: Local Binary Pattern texture map
- `texture_contrast.png`: Contrast variation map

## Credits

This project builds on work from the GNOME openQA project and uses free software components including OpenCV, diffoscope, PyMuPDF, and others for image comparison and analysis.