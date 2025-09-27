# Quick Start Guide Index

## Choose Your Adventure

### **HOWTO-1**: [Print Quality Analysis](HOWTO-1-print-quality-analysis.md) *(2 minutes)*

- **Perfect for**: First-time users wanting immediate results
- **What you get**: Quality scores, visual comparisons, problem detection
- **Command**: `python print_quality_pipeline.py Virtual_PDF_Printer`

### **HOWTO-2**: [CUPS Filter Chain Detective](HOWTO-2-filter-chain-detective.md) *(3 minutes)*

- **Perfect for**: Debugging print pipeline issues
- **What you get**: Step-by-step filter analysis, bottleneck detection
- **Command**: `python filter_chain_test_pipeline.py Virtual_PDF_Printer --fast-mode`

### **HOWTO-3**: [Document Testing Suite](HOWTO-3-document-testing-suite.md) *(5 minutes)*

- **Perfect for**: Comprehensive quality assurance
- **What you get**: 15+ document types tested, success rate analysis
- **Command**: `python test_document_pipeline.py`

### **HOWTO-4**: [Advanced Image Analysis Demo](HOWTO-4-image-analysis-demo.md) *(3 minutes)*

- **Perfect for**: Exploring cutting-edge algorithms
- **What you get**: 11 advanced algorithms, detailed analysis
- **Command**: `python example_usage.py`
<!-- 
## Quick Verification

Before diving in, verify everything works:

```bash
# Quick check (30 seconds)
python verify_examples.py --quick

# Full verification (15 minutes)
python verify_examples.py
``` -->

## Prerequisites

- **Python 3.8+** with OpenCV, NumPy, PIL
- **CUPS printing system** installed
- **Virtual printer** set up (or any CUPS printer)

## What Each HOWTO Demonstrates

| HOWTO | Focus | Time | Key Feature |
|-------|-------|------|-------------|
| **1** | Basic Quality | 2 min | Instant quality scores |
| **2** | Filter Analysis | 3 min | Pipeline debugging |
| **3** | Document Testing | 5 min | Comprehensive testing |
| **4** | Algorithm Demo | 3 min | Advanced analysis |

## Tips for Success

1. **Start with HOWTO-1** for immediate gratification
2. **Use fast mode** (`--fast-mode`) for quicker results  
3. **Check HTML reports** for visual results
4. **Run verification** if anything doesn't work

---

*Each HOWTO is designed to wow you within minutes - choose based on your interests and available time!*
