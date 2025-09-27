# HOWTO-3: Document Testing Suite (5 minutes)

## What you'll achieve

Stress-test your print pipeline with 15+ different document types automatically! Get a comprehensive quality report showing which document types have problems and exactly where issues occur.

## Quick Start

### Step 1: Run Comprehensive Document Tests

```bash
python test_document_pipeline.py
```

### Step 2: View the Results

Check the generated `test_results/` directory:

- **`test_report.html`** - Visual report with document comparisons
- **`test_summary.csv`** - Spreadsheet with all results
- **`processed_documents/`** - All processed test documents
- **`individual_results/`** - Per-document analysis

## What you'll see

### Comprehensive Test Coverage

```text
📄 DOCUMENT TYPES TESTED:
✅ English Text (Helvetica, Times, Arial)
✅ Multi-language (Spanish, French, German, Chinese)
✅ Pure Images (Photos, Graphics, Charts)
✅ Mixed Content (Text + Images)
✅ Complex Layouts (Tables, Columns, Headers)
✅ Multi-page Documents (5+ pages)
✅ Different Paper Sizes (A4, Letter, Legal)
```

### Success Rate Summary

```text
OVERALL RESULTS:
Success Rate: 92% (14/15 document types passed)
Average Quality Score: 0.847
Processing Time: 4.2 minutes

🏆 BEST PERFORMERS:
• Simple Text Documents: 98% quality
• Pure Images: 95% quality  
• Mixed Content: 89% quality

⚠️ PROBLEMATIC DOCUMENTS:
• Complex Tables: 67% quality (BELOW THRESHOLD)
• Multi-column Layout: 72% quality (NEEDS ATTENTION)
```

### Per-Document Analysis

Each document gets detailed analysis:

- **Text Preservation**: OCR comparison of original vs processed
- **Image Quality**: SSIM, PSNR, edge quality metrics
- **Layout Integrity**: Structure and formatting preservation
- **Color Accuracy**: Color fidelity assessment
- **Page Completeness**: All pages present and correct

## Visual Results

### HTML Report Features

- **Side-by-side comparisons** for each document type
- **Quality heatmaps** showing problem areas
- **Text difference highlighting** with OCR analysis
- **Pass/fail indicators** with clear explanations
- **Recommendation sections** for failed documents

### Example Results Table

| Document Type | Quality Score | Text Accuracy | Image Quality | Status |
|---------------|---------------|---------------|---------------|---------|
| English Text | 0.956 | 99.2% | N/A | ✅ PASS |
| Spanish Text | 0.943 | 97.8% | N/A | ✅ PASS |
| Photo Document | 0.889 | N/A | 88.9% | ✅ PASS |
| Mixed Content | 0.834 | 94.1% | 82.3% | ✅ PASS |
| Complex Table | 0.672 | 78.4% | 71.2% | ❌ FAIL |

## 🔧 Customization Options

### Custom Test Document Directory

```bash
# Use your own test documents
mkdir my_test_docs
# Add your PDF/image files
python test_document_pipeline.py --input-dir my_test_docs
```

### Multi-page Document Focus

```bash
# Test multi-page documents specifically
python multipage_test.py
```

This creates multi-page documents and tests:

- Page order preservation
- Missing page detection
- Individual page quality
- Page boundary integrity

### Specific Document Types

```bash
# Generate and test only specific types
python generate_test_documents.py --text-only
python test_document_pipeline.py --filter-text
```

## Understanding Results

### Quality Thresholds

- **≥ 0.85**: Excellent - Production ready
- **0.70-0.84**: Good - Minor issues acceptable
- **0.60-0.69**: Poor - Needs investigation  
- **< 0.60**: Critical - Major problems

### Common Issues Detected

- **Text blur**: OCR accuracy drops, SSIM < 0.8
- **Color shifts**: Histogram correlation < 0.9
- **Layout corruption**: Geometric analysis fails
- **Missing content**: Feature matching drops significantly
- **Page reordering**: Multi-page integrity check fails

## Troubleshooting Guide

### If Many Documents Fail

```bash
# Check printer configuration
python printer_info.py

# Test with simpler documents first
python test_pipeline.py --simple-images-only
```

### If Specific Types Fail

```bash
# Debug specific document type
python debug_pipeline.py --document-type complex_layout

# Check filter chain for that type
python filter_chain_test_pipeline.py YOUR_PRINTER --verbose
```

## Performance

- **15+ document types** tested automatically
- **37 test images per document** for thorough analysis
- **Multi-language support** (8 languages tested)
- **Complete analysis in 5-8 minutes**
- **Detailed HTML report** with visual comparisons

## Real-World Applications

### Pre-Production Validation

```bash
# Before deploying new printer setup
python test_document_pipeline.py
# Verify all document types work correctly
```

### Print Driver Testing

```bash
# Test new driver version
python test_document_pipeline.py --output-dir driver_v2_test
# Compare results with previous version
```

### Quality Assurance

```bash
# Regular quality checks
python test_document_pipeline.py --output-dir qa_$(date +%Y%m%d)
# Track quality trends over time
```

## Next Steps

After running comprehensive document tests:

- **Fix identified issues** using filter chain analysis
- **Compare different printer configurations** 
- **Set up automated testing** for CI/CD pipelines
- Try **HOWTO-4** for advanced image algorithm demonstrations

---

*This HOWTO provides industrial-strength document testing - like having a complete QA lab for your print pipeline!*
