#!/usr/bin/env python3
"""
Test script for the new image processing algorithms
"""

import cv2
import numpy as np
import os
from enhanced_comparison import ImageComparator

def create_test_images():
    """Create test images to demonstrate the algorithms"""
    
    # Create a test image with text and geometric shapes
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "SAMPLE DOCUMENT", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "This is a test document for image analysis", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "It contains various text densities and geometric shapes", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Add geometric shapes
    cv2.rectangle(img, (100, 300), (300, 450), (0, 0, 0), 2)
    cv2.circle(img, (500, 375), 75, (0, 0, 0), 2)
    cv2.ellipse(img, (650, 375), (60, 40), 0, 0, 360, (0, 0, 0), 2)
    
    # Add some texture
    for i in range(10):
        for j in range(10):
            x = 50 + i * 20
            y = 500 + j * 8
            cv2.circle(img, (x, y), 2, (128, 128, 128), -1)
    
    cv2.imwrite("test_original.png", img)
    
    # Create a modified version with rotation and noise
    img_modified = img.copy()
    
    # Add some noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img_modified = np.clip(img_modified.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Rotate slightly
    center = (img.shape[1]//2, img.shape[0]//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 2.5, 1.0)
    img_modified = cv2.warpAffine(img_modified, rotation_matrix, (img.shape[1], img.shape[0]), 
                                  borderValue=(255, 255, 255))
    
    cv2.imwrite("test_modified.png", img_modified)
    
    return "test_original.png", "test_modified.png"

def test_individual_algorithms():
    """Test each algorithm individually"""
    print("Creating test images...")
    original_path, modified_path = create_test_images()
    
    print("Initializing ImageComparator...")
    comparator = ImageComparator(original_path, modified_path, output_dir="algorithm_test_output")
    
    os.makedirs("algorithm_test_output", exist_ok=True)
    
    print("\n=== Testing Density Detection ===")
    density_results = comparator.image_density_detection()
    print(f"Overall Content Density: {density_results['overall_content_density']:.4f}")
    print(f"Text Density: {density_results['text_density']:.4f}")
    print(f"Image Content Density: {density_results['image_content_density']:.4f}")
    print(f"White Space Ratio: {density_results['white_space_ratio']:.4f}")
    print(f"Number of Dense Clusters: {density_results['num_dense_clusters']}")
    
    print("\n=== Testing Rotation Detection ===")
    rotation_results = comparator.rotation_detection()
    print(f"Detected Rotation Angle: {rotation_results['rotation_angle']:.2f}°")
    print(f"Rotation Confidence: {rotation_results['rotation_confidence']:.4f}")
    print(f"Is Rotated: {rotation_results['is_rotated']}")
    if rotation_results['hough_angles']:
        print(f"Hough Line Angles: {[f'{angle:.2f}°' for angle in rotation_results['hough_angles'][:5]]}")
    
    print("\n=== Testing Noise Analysis ===")
    noise_results = comparator.noise_analysis()
    print(f"Noise Level: {noise_results['noise_level']:.2f}")
    print(f"Noise Variance: {noise_results['noise_variance']:.2f}")
    print(f"Signal-to-Noise Ratio: {noise_results['snr_db']:.2f} dB")
    print(f"Salt & Pepper Noise Ratio: {noise_results['salt_pepper_ratio']:.6f}")
    print(f"Noise Quality: {noise_results['noise_quality']}")
    
    print("\n=== Testing Texture Analysis ===")
    texture_results = comparator.texture_analysis()
    print(f"Texture Complexity: {texture_results['texture_complexity']}")
    print(f"Contrast Measure: {texture_results['contrast_measure']:.2f}")
    print(f"Texture Energy: {texture_results['texture_energy']:.2f}")
    print(f"Direction Uniformity: {texture_results['direction_uniformity']:.4f}")
    print(f"Roughness: {texture_results['roughness']:.2f}")
    
    print("\n=== Testing Geometric Analysis ===")
    geometric_results = comparator.geometric_analysis()
    print(f"Aspect Ratio: {geometric_results['aspect_ratio']:.3f}")
    print(f"Average Rectangularity: {geometric_results['avg_rectangularity']:.4f}")
    print(f"Horizontal Symmetry: {geometric_results['horizontal_symmetry']:.4f}")
    print(f"Vertical Symmetry: {geometric_results['vertical_symmetry']:.4f}")
    print(f"Number of Rectangular Objects: {geometric_results['num_rectangular_objects']}")
    print(f"Geometric Quality: {geometric_results['geometric_quality']}")
    
    print("\n=== Testing Color/Monochrome Detection ===")
    color_results = comparator.color_monochrome_detection()
    print(f"Original Image: {'Grayscale' if color_results['original_is_grayscale'] else 'Color'} (confidence: {color_results['original_confidence']:.3f})")
    print(f"Processed Image: {'Grayscale' if color_results['processed_is_grayscale'] else 'Color'} (confidence: {color_results['processed_confidence']:.3f})")
    print(f"Color Type Assessment: {color_results['color_type_assessment']}")
    print(f"Color Consistency: {'Yes' if color_results['color_consistency'] else 'No'}")
    print(f"Color Preservation Score: {color_results['color_preservation_score']:.4f}")
    print(f"Original Max Channel Diff: {color_results['original_max_channel_diff']:.2f}")
    print(f"Original Mean Saturation: {color_results['original_mean_saturation']:.2f}")
    print(f"Processed Max Channel Diff: {color_results['processed_max_channel_diff']:.2f}")
    print(f"Processed Mean Saturation: {color_results['processed_mean_saturation']:.2f}")
    
    print(f"\nVisualization files saved in 'algorithm_test_output' directory:")
    print("- density_heatmap.png: Content density distribution")
    print("- rotation_detection.png: Detected lines for rotation analysis")
    print("- noise_analysis.png: Noise visualization")
    print("- texture_lbp.png: Local Binary Pattern analysis")
    print("- texture_contrast.png: Texture contrast visualization")

def test_comprehensive_analysis():
    """Test the comprehensive analysis with all algorithms"""
    print("\n\n=== Comprehensive Analysis Test ===")
    original_path, modified_path = create_test_images()
    
    comparator = ImageComparator(original_path, modified_path, output_dir="comprehensive_test_output")
    os.makedirs("comprehensive_test_output", exist_ok=True)
    
    results = comparator.run_all_comparisons()
    
    print(f"\nOverall Quality Score: {results['overall_quality']:.4f}")
    print(f"Quality factors:")
    print(f"  - SSIM: {results['ssim']:.4f}")
    print(f"  - Edge Quality: {results['edge_quality_score']:.4f}")
    print(f"  - Content Density: {results['overall_content_density']:.4f}")
    print(f"  - Rotation Score: {1.0 - min(abs(results['rotation_angle']) / 45.0, 1.0):.4f}")
    print(f"  - Noise Score: {1.0 - min(results['noise_level'] / 50.0, 1.0):.4f}")
    print(f"  - Geometric Quality: {results['avg_rectangularity']:.4f}")
    print(f"  - Color Preservation: {results['color_preservation_score']:.4f}")

if __name__ == "__main__":
    print("Testing Image Processing Algorithms")
    print("==================================")
    
    test_individual_algorithms()
    test_comprehensive_analysis()
    
    print("\nTest completed! Check the generated visualization files.")
