#!/usr/bin/env python3
"""
Example usage of the image processing algorithms with real images
"""

import cv2
import numpy as np
import os
from enhanced_comparison import ImageComparator

def analyze_image_quality(image_path, output_dir="analysis_output"):
    """
    Analyze a single image using all available algorithms
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save analysis results
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    comparator = ImageComparator(image_path, image_path, output_dir=output_dir)
    
    print(f"\n=== Analyzing Image: {os.path.basename(image_path)} ===")
    
    # Run density analysis
    print("\n--- Density Analysis ---")
    density = comparator.image_density_detection()
    print(f"Content Density: {density['overall_content_density']:.4f}")
    print(f"Text Density: {density['text_density']:.4f}")
    print(f"White Space: {density['white_space_ratio']:.4f}")
    print(f"Dense Clusters: {density['num_dense_clusters']}")
    
    # Run rotation analysis
    print("\n--- Rotation Analysis ---")
    rotation = comparator.rotation_detection()
    print(f"Rotation Angle: {rotation['rotation_angle']:.2f}°")
    print(f"Confidence: {rotation['rotation_confidence']:.4f}")
    print(f"Needs Correction: {rotation['is_rotated']}")
    
    # Run noise analysis
    print("\n--- Noise Analysis ---")
    noise = comparator.noise_analysis()
    print(f"Noise Level: {noise['noise_level']:.2f}")
    print(f"Quality Rating: {noise['noise_quality']}")
    print(f"SNR: {noise['snr_db']:.2f} dB")
    
    # Run texture analysis
    print("\n--- Texture Analysis ---")
    texture = comparator.texture_analysis()
    print(f"Complexity: {texture['texture_complexity']}")
    print(f"Contrast: {texture['contrast_measure']:.2f}")
    print(f"Roughness: {texture['roughness']:.2f}")
    
    # Run geometric analysis
    print("\n--- Geometric Analysis ---")
    geometric = comparator.geometric_analysis()
    print(f"Aspect Ratio: {geometric['aspect_ratio']:.3f}")
    print(f"Quality: {geometric['geometric_quality']}")
    print(f"Rectangularity: {geometric['avg_rectangularity']:.4f}")
    print(f"Symmetry (H/V): {geometric['horizontal_symmetry']:.3f}/{geometric['vertical_symmetry']:.3f}")
    
    print(f"\nVisualization files saved in '{output_dir}/' directory")
    
    return {
        'density': density,
        'rotation': rotation,
        'noise': noise,
        'texture': texture,
        'geometric': geometric
    }

def compare_images_with_algorithms(image1_path, image2_path, output_dir="comparison_output"):
    """
    Compare two images using all algorithms
    
    Args:
        image1_path: Path to first image (reference)
        image2_path: Path to second image (processed)
        output_dir: Directory to save comparison results
    """
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("Error: One or both image files not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparator = ImageComparator(image1_path, image2_path, output_dir=output_dir)
    
    print(f"\n=== Comparing Images ===")
    print(f"Reference: {os.path.basename(image1_path)}")
    print(f"Processed: {os.path.basename(image2_path)}")
    
    # Run comprehensive analysis
    results = comparator.run_all_comparisons()
    
    print(f"\n--- Overall Quality Score: {results['overall_quality']:.4f} ---")
    
    print("\n--- Traditional Metrics ---")
    print(f"SSIM: {results['ssim']:.4f}")
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"MSE: {results['mse']:.2f}")
    
    print("\n--- Advanced Analysis ---")
    print(f"Edge Quality: {results['edge_quality_score']:.4f}")
    print(f"Color Similarity: {results['histogram_similarity']:.4f}")
    print(f"Feature Matches: {results['keypoint_matches']}")
    
    print("\n--- New Algorithm Results ---")
    print(f"Density Preservation: {results['overall_content_density']:.4f}")
    print(f"Rotation Detected: {results['rotation_angle']:.2f}° (conf: {results['rotation_confidence']:.3f})")
    print(f"Noise Level: {results['noise_level']:.2f} ({results['noise_quality']})")
    print(f"Texture Preservation: {results['texture_complexity']}")
    print(f"Geometric Quality: {results['geometric_quality']}")
    
    print(f"\nDetailed analysis files saved in '{output_dir}/' directory")
    
    return results

def process_directory(input_dir, output_base_dir="batch_analysis"):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing images to analyze
        output_base_dir: Base directory for output files
    """
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found!")
        return
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"No image files found in '{input_dir}'")
        return
    
    print(f"\nFound {len(image_files)} images to analyze")
    
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        output_dir = os.path.join(output_base_dir, f"image_{i+1:03d}_{os.path.splitext(os.path.basename(image_path))[0]}")
        
        try:
            results = analyze_image_quality(image_path, output_dir)
            results_summary.append({
                'filename': os.path.basename(image_path),
                'path': image_path,
                'results': results
            })
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    #summary report
    print(f"\n=== Batch Analysis Summary ===")
    print(f"Processed {len(results_summary)} images successfully")
    
    if results_summary:
        print("\nDensity Summary:")
        avg_content_density = np.mean([r['results']['density']['overall_content_density'] for r in results_summary])
        avg_text_density = np.mean([r['results']['density']['text_density'] for r in results_summary])
        print(f"Average Content Density: {avg_content_density:.4f}")
        print(f"Average Text Density: {avg_text_density:.4f}")
        
        print("\nRotation Summary:")
        rotated_images = [r for r in results_summary if r['results']['rotation']['is_rotated']]
        print(f"Images needing rotation correction: {len(rotated_images)}/{len(results_summary)}")
        
        print("\nNoise Summary:")
        noise_levels = [r['results']['noise']['noise_level'] for r in results_summary]
        print(f"Average Noise Level: {np.mean(noise_levels):.2f}")
        print(f"Noise Level Range: {np.min(noise_levels):.2f} - {np.max(noise_levels):.2f}")

# Example usage functions
def demo_with_generated_images():
    """Demonstrate with generated test images"""
    print("=== Demo with Generated Test Images ===")
    
    #test images created by test_algorithms.py
    if os.path.exists("test_original.png") and os.path.exists("test_modified.png"):
        compare_images_with_algorithms("test_original.png", "test_modified.png", "demo_comparison")
    else:
        print("Test images not found. Run test_algorithms.py first.")

def demo_with_existing_images():
    """Demonstrate with existing images in the workspace"""
    print("\n=== Demo with Existing Images ===")
    
    # Check for existing PNG files
    png_files = [f for f in os.listdir('.') if f.endswith('.png') and f not in ['test_original.png', 'test_modified.png']]
    
    if png_files:
        print(f"Found {len(png_files)} PNG files to analyze")
        for png_file in png_files[:2]:  # Analyze first 2 files
            analyze_image_quality(png_file, f"analysis_{os.path.splitext(png_file)[0]}")
    else:
        print("No existing PNG files found for analysis")

if __name__ == "__main__":
    print("Image Processing Algorithms - Example Usage")
    print("==========================================")
    
    demo_with_generated_images()
    
    demo_with_existing_images()
    
    print("\n=== Usage Examples ===")
    print("1. Analyze single image:")
    print("   python example_usage.py")
    print("   analyze_image_quality('my_image.png')")
    print()
    print("2. Compare two images:")
    print("   compare_images_with_algorithms('original.png', 'processed.png')")
    print()
    print("3. Batch process directory:")
    print("   process_directory('/path/to/images')")
