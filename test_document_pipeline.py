#!/usr/bin/env python3

import os
import subprocess
import cv2
import pytesseract
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import glob
import time
import json
import pandas as pd
from enhanced_comparison import ImageComparator
from generate_test_documents import generate_all_test_documents

def run_single_page_tests(test_dir="test_documents", results_dir="test_results"):
    # Step 1: Setup directories
    print("Setting up directories...")
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Step 2: Generate test documents
    print("Generating test documents...")
    test_files = generate_all_test_documents(test_dir)

    results_data = []

    for test_file in test_files:
        try:
            test_name = os.path.basename(test_file)
            print(f"\n\n=== Processing test document: {test_name} ===")
            
            base_name = os.path.splitext(test_name)[0]
            
            # Step 3: Convert to image
            sample_png = os.path.join(test_dir, f"{base_name}.png")
            try:
                result = subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                                f'-sOutputFile={sample_png}', test_file], 
                                check=True, capture_output=True, text=True)
                print(f"Successfully converted {test_file} to PNG")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {test_file} to PNG: {e}")
                print(f"Error output: {e.stderr}")
                continue
            
            # Step 4: Print file using CUPS
            subprocess.run(['lp', test_file])
            
            # Step 5: Wait for processing
            print(f"Waiting for print processing...")
            time.sleep(5)
            
            # Step 6: Find the generated output file
            output_dir = os.path.expanduser('~/gsoc/openprinting/PDF/')
            output_files = sorted(glob.glob(f"{output_dir}/*.pdf"), key=os.path.getmtime)
            
            if output_files:
                latest_output = output_files[-1]
                print(f"Found output file: {latest_output}")
                output_pdf = os.path.join(results_dir, f"{base_name}_output.pdf")
                subprocess.run(['cp', latest_output, output_pdf])
            else:
                print("No output PDF found in the expected location!")
                continue
            
            # Step 7: Convert processed output to image
            output_png = os.path.join(results_dir, f"{base_name}_output.png")
            subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                            f'-sOutputFile={output_png}', output_pdf])
            
            # Step 8: Run enhanced image comparison
            print(f"Running image comparison for {base_name}...")
            comparator = ImageComparator(sample_png, output_png, results_dir)
            results = comparator.run_all_comparisons()
            
            # Store the visual diff in the results directory
            visual_diff = 'visual_diff.png'
            if os.path.exists(visual_diff):
                visual_diff_dest = os.path.join(results_dir, f"{base_name}_visual_diff.png")
                subprocess.run(['mv', visual_diff, visual_diff_dest])
            
            for artifact in ['feature_matches.png', 'edges_original.png', 'edges_processed.png']:
                if os.path.exists(artifact):
                    artifact_dest = os.path.join(results_dir, f"{base_name}_{artifact}")
                    subprocess.run(['mv', artifact, artifact_dest])
            
            # Save individual results to JSON
            results_json = os.path.join(results_dir, f"{base_name}_results.json")
            with open(results_json, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Extract text if the test document contains text
            if "text" in base_name or "mixed" in base_name or "complex" in base_name:
                original_text = pytesseract.image_to_string(Image.open(sample_png))
                processed_text = pytesseract.image_to_string(Image.open(output_png))
                
                # Calculate text similarity
                if original_text.strip() and processed_text.strip():
                    from difflib import SequenceMatcher
                    text_similarity = SequenceMatcher(None, original_text, processed_text).ratio()
                else:
                    text_similarity = 0
                    
                results["text_similarity"] = text_similarity
                
                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=4)
            else:
                text_similarity = None
            
            # Store results for summary
            result_entry = {
                "test_name": test_name,
                "ssim": results["ssim"],
                "psnr": results["psnr"],
                "match_confidence": results["match_confidence"],
                "edge_similarity": results["edge_similarity"],
                "histogram_similarity": results["histogram_similarity"],
                "text_similarity": text_similarity,
                "overall_quality": results["overall_quality"],
                "pass": results["overall_quality"] >= 0.8
            }
            results_data.append(result_entry)
        
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
            continue

    # Create summary report
    print("\n\n=== Test Summary ===")
    df = pd.DataFrame(results_data)
    summary_file = os.path.join(results_dir, "test_summary.csv")
    df.to_csv(summary_file, index=False)

    return results_data

def generate_html_report(results_data, test_dir, results_dir):
    # Generate HTML report
    html_report = os.path.join(results_dir, "test_report.html")
    with open(html_report, 'w') as f:
        f.write("<html><head><title>Print Pipeline Test Results</title>")
        f.write("<style>")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
        f.write("table { border-collapse: collapse; width: 100%; }")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        f.write("th { background-color: #f2f2f2; }")
        f.write("tr:nth-child(even) { background-color: #f9f9f9; }")
        f.write(".pass { color: green; font-weight: bold; }")
        f.write(".fail { color: red; font-weight: bold; }")
        f.write("</style></head><body>")
        
        f.write("<h1>Print Pipeline Test Results</h1>")
        f.write(f"<p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        f.write("<h2>Summary</h2>")
        total_tests = len(results_data)
        passed_tests = sum(1 for r in results_data if r["pass"])
        f.write(f"<p>Total Tests: {total_tests}</p>")
        f.write(f"<p>Passed Tests: {passed_tests} ({passed_tests/total_tests*100:.1f}%)</p>")
        
        f.write("<h2>Detailed Results</h2>")
        f.write("<table>")
        f.write("<tr><th>Test</th><th>SSIM</th><th>PSNR</th><th>Feature Match</th>")
        f.write("<th>Edge Similarity</th><th>Color Similarity</th><th>Text Similarity</th>")
        f.write("<th>Overall Quality</th><th>Result</th></tr>")
        
        for result in results_data:
            f.write("<tr>")
            f.write(f"<td>{result['test_name']}</td>")
            f.write(f"<td>{result['ssim']:.4f}</td>")
            
            # Handle infinity value for PSNR
            if result['psnr'] == float('inf'):
                f.write("<td>∞</td>")
            else:
                f.write(f"<td>{result['psnr']:.2f}</td>")
                
            f.write(f"<td>{result['match_confidence']:.4f}</td>")
            f.write(f"<td>{result['edge_similarity']:.4f}</td>")
            f.write(f"<td>{result['histogram_similarity']:.4f}</td>")
            
            if result['text_similarity'] is not None:
                f.write(f"<td>{result['text_similarity']:.4f}</td>")
            else:
                f.write("<td>N/A</td>")
                
            f.write(f"<td>{result['overall_quality']:.4f}</td>")
            
            if result["pass"]:
                f.write("<td class='pass'>PASS</td>")
            else:
                f.write("<td class='fail'>FAIL</td>")
            f.write("</tr>")
        
        f.write("</table>")
        
        f.write("<h2>Test Images</h2>")
        for result in results_data:
            base_name = os.path.splitext(result['test_name'])[0]
            f.write(f"<h3>{result['test_name']}</h3>")
            
            f.write("<div style='display: flex; gap: 20px;'>")
            
            sample_png = os.path.join("../", test_dir, f"{base_name}.png")
            output_png = f"{base_name}_output.png"
            diff_png = f"{base_name}_visual_diff.png"
            
            f.write("<div>")
            f.write("<h4>Original</h4>")
            f.write(f"<img src='{sample_png}' width='400' />")
            f.write("</div>")
            
            f.write("<div>")
            f.write("<h4>Processed</h4>")
            f.write(f"<img src='{output_png}' width='400' />")
            f.write("</div>")
            
            if os.path.exists(os.path.join(results_dir, diff_png)):
                f.write("<div>")
                f.write("<h4>Difference</h4>")
                f.write(f"<img src='{diff_png}' width='400' />")
                f.write("</div>")
            
            f.write("</div>")
        
        f.write("</body></html>")

if __name__ == "__main__":
    test_dir = "test_documents"
    results_dir = "test_results"
    
    results_data = run_single_page_tests(test_dir, results_dir)
    generate_html_report(results_data, test_dir, results_dir)
    
    summary_file = os.path.join(results_dir, "test_summary.csv")
    html_report = os.path.join(results_dir, "test_report.html")
    
    print(f"\nTest summary saved to {summary_file}")
    print(f"HTML report saved to {html_report}")
    print(f"All result files saved to {results_dir}/")