#!/usr/bin/env python3
"""Debug script to test the pipeline components individually"""

import sys
import os
import time
from print_quality_pipeline import PrintQualityTestPipeline

def test_printer_analysis():
    """Test printer capability analysis"""
    print("=== Testing Printer Analysis ===")
    pipeline = PrintQualityTestPipeline("PDF", "debug_output")
    
    start_time = time.time()
    try:
        capabilities = pipeline.analyze_printer_capabilities()
        elapsed = time.time() - start_time
        print(f"✓ Printer analysis completed in {elapsed:.2f}s")
        print(f"  Supported modes: {len(pipeline.supported_modes)}")
        print(f"  Filter chains: {list(pipeline.filter_chains.keys())}")
        return True
    except Exception as e:
        print(f"✗ Printer analysis failed: {e}")
        return False

def test_image_generation():
    """Test image generation"""
    print("\n=== Testing Image Generation ===")
    pipeline = PrintQualityTestPipeline("PDF", "debug_output")
    
    start_time = time.time()
    try:
        images = pipeline.generate_test_images()
        elapsed = time.time() - start_time
        print(f"✓ Image generation completed in {elapsed:.2f}s")
        print(f"  Generated {len(images)} images")
        return True
    except Exception as e:
        print(f"✗ Image generation failed: {e}")
        return False

def test_single_image_processing():
    """Test processing a single image"""
    print("\n=== Testing Single Image Processing ===")
    pipeline = PrintQualityTestPipeline("PDF", "debug_output")
    
    # First analyze printer and generate one test image
    try:
        pipeline.analyze_printer_capabilities()
        
        # Create a simple test image
        import cv2
        import numpy as np
        test_image_path = os.path.join(pipeline.dirs['test_images'], 'simple_test.png')
        simple_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(simple_img, (50, 50), (250, 250), (0, 0, 0), 5)
        cv2.imwrite(test_image_path, simple_img)
        
        if pipeline.supported_modes:
            mode = pipeline.supported_modes[0]
            mode_name = mode['name']
            
            start_time = time.time()
            processed_image = pipeline._process_image_through_filters(
                test_image_path, mode, mode_name
            )
            elapsed = time.time() - start_time
            
            if processed_image:
                print(f"✓ Single image processing completed in {elapsed:.2f}s")
                print(f"  Output: {processed_image}")
                return True
            else:
                print(f"✗ Single image processing failed (no output)")
                return False
        else:
            print("✗ No supported modes found")
            return False
            
    except Exception as e:
        print(f"✗ Single image processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("Pipeline Debug Tool")
    print("===================")
    
    results = []
    
    # Test 1: Printer Analysis
    results.append(test_printer_analysis())
    
    # Test 2: Image Generation  
    results.append(test_image_generation())
    
    # Test 3: Single Image Processing
    results.append(test_single_image_processing())
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All components working!")
    else:
        print("✗ Some components failed - check output above")
        
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
