#!/usr/bin/env python3

import os
import json
import numpy as np
import cv2
import fitz  # PyMuPDF
import subprocess
import time
from enhanced_comparison import ImageComparator
import generate_test_documents

def test_multipage_document(reference_pdf, processed_pdf, output_dir, config=None):
    """
    Test a multi-page document by comparing pages between reference and processed PDFs
    
    Parameters:
    - reference_pdf: Path to reference PDF file
    - processed_pdf: Path to processed PDF file 
    - output_dir: Directory to save results and visualization
    - config: Configuration dictionary for test parameters
    
    Returns:
    - Results dictionary with comparison metrics
    """
    
    if config is None:
        config = {
            "dpi": 300,
            "save_artifacts": True,
            "detailed_output": True
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Opening reference PDF: {reference_pdf}")
        ref_doc = fitz.open(reference_pdf)
        print(f"Opening processed PDF: {processed_pdf}")
        proc_doc = fitz.open(processed_pdf)
        
        ref_pages = []
        proc_pages = []
        
        print(f"Extracting {len(ref_doc)} pages from reference document...")
        for page_num in range(len(ref_doc)):
            try:
                page = ref_doc.load_page(page_num)
                print(f"  Processing reference page {page_num+1}/{len(ref_doc)}")
                pix = page.get_pixmap(matrix=fitz.Matrix(config["dpi"]/72, config["dpi"]/72))
                
                if pix.samples is None:
                    print(f"Warning: Could not extract pixel data from reference page {page_num+1}")
                    continue
                    
                h, w, n = pix.h, pix.w, pix.n
                if h <= 0 or w <= 0 or n <= 0:
                    print(f"Warning: Invalid dimensions for reference page {page_num+1}: {h}x{w}x{n}")
                    continue
                
                try:
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, n)
                    
                    if img.shape[0] <= 0 or img.shape[1] <= 0 or img.shape[2] <= 0:
                        print(f"Warning: Created image has invalid shape {img.shape}")
                        continue
                        
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
                    ref_pages.append(img)
                    
                    if config["save_artifacts"]:
                        output_path = os.path.join(output_dir, f"reference_page_{page_num+1}.png")
                        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                except (ValueError, cv2.error) as e:
                    print(f"Error processing reference page {page_num+1}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error extracting reference page {page_num+1}: {e}")
                continue
        
        print(f"Extracting {len(proc_doc)} pages from processed document...")
        for page_num in range(len(proc_doc)):
            try:
                page = proc_doc.load_page(page_num)
                print(f"  Processing processed page {page_num+1}/{len(proc_doc)}")
                pix = page.get_pixmap(matrix=fitz.Matrix(config["dpi"]/72, config["dpi"]/72))
                
                if pix.samples is None:
                    print(f"Warning: Could not extract pixel data from processed page {page_num+1}")
                    continue
                    
                h, w, n = pix.h, pix.w, pix.n
                if h <= 0 or w <= 0 or n <= 0:
                    print(f"Warning: Invalid dimensions for processed page {page_num+1}: {h}x{w}x{n}")
                    continue
                
                try:
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, n)
                    
                    if img.shape[0] <= 0 or img.shape[1] <= 0 or img.shape[2] <= 0:
                        print(f"Warning: Created image has invalid shape {img.shape}")
                        continue
                        
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    
                    proc_pages.append(img)
                    
                    if config["save_artifacts"]:
                        output_path = os.path.join(output_dir, f"processed_page_{page_num+1}.png")
                        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                except (ValueError, cv2.error) as e:
                    print(f"Error processing processed page {page_num+1}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error extracting processed page {page_num+1}: {e}")
                continue
        
        print(f"Extracted {len(ref_pages)} reference pages and {len(proc_pages)} processed pages")
        
        if not ref_pages:
            return {"error": "No valid reference pages could be extracted"}
        
        if not proc_pages:
            return {"error": "No valid processed pages could be extracted"}
        
        all_results = {}

        print("Running basic image comparisons on first page...")
        comparator = ImageComparator(ref_pages[0], proc_pages[0], output_dir)
        all_results = comparator.run_all_comparisons()

        print(f"Running comparisons on all {len(ref_pages)} pages...")
        page_quality_scores = []

        for i in range(len(ref_pages)):
            if i >= len(proc_pages):
                break 
                
            print(f"  Comparing page {i+1}...")
            page_comparator = ImageComparator(ref_pages[i], proc_pages[i], output_dir)
            page_results = page_comparator.run_all_comparisons()
            
            all_results[f"page_{i+1}_metrics"] = {
                "ssim": page_results["ssim"],
                "psnr": page_results["psnr"],
                "edge_quality": page_results["edge_quality_score"],
                "overall_quality": page_results["overall_quality"]
            }
            
            page_quality_scores.append(page_results["overall_quality"])

        if page_quality_scores:
            all_results["avg_page_quality"] = sum(page_quality_scores) / len(page_quality_scores)

        print("Running page integrity analysis...")
        page_integrity_results = comparator.page_integrity_comparison(True, ref_pages, proc_pages, output_dir)
        all_results.update(page_integrity_results)
            
        print("Saving visualization files...")
        for vis_file in ['page_integrity_map.png', 'visual_diff.png', 'edges_original.png', 
                        'edges_processed.png', 'feature_matches.png']:
            if os.path.exists(vis_file):
                dest_path = os.path.join(output_dir, vis_file)
                os.replace(vis_file, dest_path)
                print(f"  Saved {dest_path}")
            
        if config["detailed_output"]:
            results_file = os.path.join(output_dir, "multipage_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"Saved detailed results to {results_file}")
                
        return all_results

    except Exception as e:
        import traceback
        print(f"Error comparing documents: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def run_multipage_tests(test_dir="test_documents", results_dir="test_results/multipage"):
    """
    Run tests on multipage documents to verify page integrity and order
    
    Parameters:
    - test_dir: Directory to store test documents
    - results_dir: Directory to store test results
    """
    print("\n=== Running Multi-page Document Tests ===")
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Generating multi-page test documents...")
    multipage_pdf_path = os.path.join(test_dir, "multipage_test.pdf")
    generate_test_documents.create_multipage_document(multipage_pdf_path, pages=5)
    
    sample_png_path = os.path.join(test_dir, "multipage_test.png")
    try:
        result = subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                    f'-sOutputFile={sample_png_path}', multipage_pdf_path], 
                    check=True, capture_output=True, text=True)
        print("Successfully created PNG preview of test document")
    except subprocess.CalledProcessError as e:
        print(f"Error creating PNG preview: {e}")
        print(f"Error output: {e.stderr}")
    
    print("Processing through print pipeline...")
    try:
        result = subprocess.run(['lp', multipage_pdf_path], 
                    check=True, capture_output=True, text=True)
        print("Successfully sent document to printer")
    except subprocess.CalledProcessError as e:
        print(f"Error sending document to printer: {e}")
        print(f"Error output: {e.stderr}")
    
    print("Waiting for print processing to complete...")
    time.sleep(5)
    
    output_dir = "PDF"
    if not os.path.exists(output_dir):
        print("Error: Output directory not found! Test failed.")
        return None
    
    output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                   if f.lower().endswith('.pdf')]
    
    if not output_files:
        print("Error: No output PDF found! Test failed.")
        return None
    
    output_files.sort(key=os.path.getmtime)
    processed_pdf = output_files[-1]
    print(f"Found processed output: {processed_pdf}")
    
    print(f"Comparing reference ({multipage_pdf_path}) with processed ({processed_pdf})...")
    results = test_multipage_document(multipage_pdf_path, processed_pdf, 
                                    os.path.join(results_dir, "complete_doc"))
    
    print("\nTesting page reordering detection...")
    reordered_pdf_path = os.path.join(test_dir, "reordered_pages.pdf")
    generate_test_documents.create_multipage_document(reordered_pdf_path, pages=5, reorder=True)
    
    results_reordered = test_multipage_document(multipage_pdf_path, reordered_pdf_path,
                                            os.path.join(results_dir, "reordered"))
    
    print("\nTesting missing pages detection...")
    missing_pdf_path = os.path.join(test_dir, "missing_pages.pdf")
    generate_test_documents.create_multipage_document(missing_pdf_path, pages=3)
    
    results_missing = test_multipage_document(multipage_pdf_path, missing_pdf_path,
                                           os.path.join(results_dir, "missing"))
    
    print("\n=== Multi-page Document Test Results ===")
    print(f"Complete Document - Page Integrity Score: {results.get('page_completeness_score', 0):.4f}")
    print(f"Reordered Document - Page Integrity Score: {results_reordered.get('page_completeness_score', 0):.4f}")
    print(f"Missing Pages Document - Page Integrity Score: {results_missing.get('page_completeness_score', 0):.4f}")
    
    print(f"\nDetailed results saved to {results_dir}")
    
    return {
        "complete": results,
        "reordered": results_reordered,
        "missing": results_missing
    }

if __name__ == "__main__":
    run_multipage_tests()