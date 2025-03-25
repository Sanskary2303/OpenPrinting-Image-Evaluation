import os
import subprocess
import cv2
import pytesseract
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import glob
import time
import json
from enhanced_comparison import ImageComparator

# Step 1: Generate PDF
subprocess.run(['python3', 'generate_pdf.py'])

# Step 2: Convert to image
subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                '-sOutputFile=sample.png', 'sample.pdf'])

# Step 3: Print file using CUPS
subprocess.run(['lp', 'sample.pdf'])

# Step 4: Wait for processing
time.sleep(5)

# Step 5: Find the generated output file
output_dir = os.path.expanduser('~/gsoc/openprinting/PDF/')
output_files = sorted(glob.glob(f"{output_dir}/*.pdf"), key=os.path.getmtime)

if output_files:
    latest_output = output_files[-1]
    print(f"Found output file: {latest_output}")
    subprocess.run(['cp', latest_output, './output.pdf'])
else:
    raise FileNotFoundError("No output PDF found in the expected location!")

# Step 6: Convert processed output to image
subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                '-sOutputFile=output.png', latest_output])

# Step 7: Compare files using diffoscope
subprocess.run(['diffoscope', 'sample.pdf', 'output.pdf'])

# Step 8: Enhanced image comparison
print("\n=== Running Enhanced Image Comparison ===")
comparator = ImageComparator('sample.png', 'output.png')
results = comparator.run_all_comparisons()

# Save results to JSON file
with open('comparison_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Print key metrics
print(f"Overall Quality Score: {results['overall_quality']:.4f}")
print(f"SSIM Score: {results['ssim']:.4f}")
print(f"Feature Match Confidence: {results['match_confidence']:.4f}")
print(f"Color Histogram Similarity: {results['histogram_similarity']:.4f}")

# Determine if test passes based on thresholds
quality_threshold = 0.8 
if results['overall_quality'] >= quality_threshold:
    print("\n✅ TEST PASSED: Output image quality is acceptable")
else:
    print("\n❌ TEST FAILED: Output image quality is below threshold")
    if results['ssim'] < 0.7:
        print("  - Structural similarity is too low")
    if results['match_confidence'] < 0.6:
        print("  - Feature matching confidence is poor")
    if results['histogram_similarity'] < 0.7:
        print("  - Color reproduction is inaccurate")
    if results['avg_delta_e'] > 10:
        print("  - Color differences are too large")

# Step 9: Extract and compare text using OCR
original_text = pytesseract.image_to_string(Image.open('sample.png'))
processed_text = pytesseract.image_to_string(Image.open('output.png'))

if original_text.strip() == processed_text.strip():
    print("✅ Text matches!")
else:
    print("❌ Text mismatch!")
    print(f"Original text: {original_text.strip()}")
    print(f"Processed text: {processed_text.strip()}")

