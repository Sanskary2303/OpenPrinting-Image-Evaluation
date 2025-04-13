import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import collections
import os

class ImageComparator:
    def __init__(self, original_path, processed_path, output_dir=None):
        """
        Initialize the comparator with original and processed images.
        
        Parameters:
        - original_path: Either a string path to an image file or a numpy array containing the image data
        - processed_path: Either a string path to an image file or a numpy array containing the image data
        - output_dir: Directory where output visualization images will be saved (default: current directory)
        """
        try:
            self.output_dir = output_dir if output_dir else "."
            
            if isinstance(original_path, str):
                self.original = cv2.imread(original_path)
                if self.original is None:
                    raise ValueError(f"Failed to load original image from {original_path}")
            else:
                self.original = original_path

            if isinstance(processed_path, str):
                self.processed = cv2.imread(processed_path)
                if self.processed is None:
                    raise ValueError(f"Failed to load processed image from {processed_path}")
            else:
                self.processed = processed_path

            if self.original is None or self.processed is None:
                raise ValueError("One or both images are None. Check input paths or arrays.")
            
            if self.original.shape != self.processed.shape:
                self.processed = cv2.resize(self.processed, 
                                            (self.original.shape[1], self.original.shape[0]))
            
            self.original_gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            self.processed_gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
            
            self.original_lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
            self.processed_lab = cv2.cvtColor(self.processed, cv2.COLOR_BGR2LAB)
        except Exception as e:
            print(f"Error initializing ImageComparator: {str(e)}")
            raise

    def basic_metrics(self):
        """Calculate basic image similarity metrics"""
        ssim_score, ssim_diff = ssim(self.original_gray, self.processed_gray, full=True)
        
        mse_score = mean_squared_error(self.original_gray, self.processed_gray)
        
        if mse_score < 1e-10: 
            psnr_score = float('inf') 
        else:
            psnr_score = psnr(self.original_gray, self.processed_gray)
        
        return {
            "ssim": ssim_score,
            "mse": mse_score,
            "psnr": psnr_score
        }
    
    def feature_detection(self):
        """Detect and compare structural features between images"""
        
        orb = cv2.ORB_create()
        
        kp1, des1 = orb.detectAndCompute(self.original_gray, None)
        kp2, des2 = orb.detectAndCompute(self.processed_gray, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 0:
                avg_distance = sum(m.distance for m in matches) / len(matches)
                match_confidence = 1.0 - min(avg_distance / 100.0, 1.0)  # Normalize
            else:
                match_confidence = 0.0
                
            matched_img = cv2.drawMatches(self.original, kp1, self.processed, kp2, 
                                        matches[:30], None, flags=2)
            output_path = os.path.join(self.output_dir, 'feature_matches.png')
            cv2.imwrite(output_path, matched_img)
            
            return {
                "keypoint_matches": len(matches),
                "keypoints_original": len(kp1),
                "keypoints_processed": len(kp2),
                "match_confidence": match_confidence
            }
        else:
            return {
                "keypoint_matches": 0,
                "keypoints_original": len(kp1) if des1 is not None else 0,
                "keypoints_processed": len(kp2) if des2 is not None else 0,
                "match_confidence": 0.0
            }
    
    def edge_comparison(self):
        """Compare edges in the images"""
        
        edges1 = cv2.Canny(self.original_gray, 100, 200)
        edges2 = cv2.Canny(self.processed_gray, 100, 200)
        
        edge_score, _ = ssim(edges1, edges2, full=True)
        
        cv2.imwrite(os.path.join(self.output_dir, 'edges_original.png'), edges1)
        cv2.imwrite(os.path.join(self.output_dir, 'edges_processed.png'), edges2)
        
        return {
            "edge_similarity": edge_score
        }
    
    def enhanced_edge_comparison(self):
        """Advanced edge detection and comparison between images"""
        
        original_canny = cv2.Canny(self.original_gray, 100, 200)
        processed_canny = cv2.Canny(self.processed_gray, 100, 200)
        
        original_canny_tight = cv2.Canny(self.original_gray, 150, 250)
        processed_canny_tight = cv2.Canny(self.processed_gray, 150, 250)
        
        original_canny_loose = cv2.Canny(self.original_gray, 50, 150)
        processed_canny_loose = cv2.Canny(self.processed_gray, 50, 150)
        
        edge_similarity = ssim(original_canny, processed_canny, full=True)[0]
        edge_similarity_tight = ssim(original_canny_tight, processed_canny_tight, full=True)[0]
        edge_similarity_loose = ssim(original_canny_loose, processed_canny_loose, full=True)[0]
        
        orig_edge_count = np.count_nonzero(original_canny)
        proc_edge_count = np.count_nonzero(processed_canny)
        
        if orig_edge_count > 0:
            edge_density_ratio = min(proc_edge_count / orig_edge_count, 2.0)  # Cap at 2.0 to prevent outliers
            edge_density_score = 1.0 - abs(1.0 - edge_density_ratio)
        else:
            edge_density_score = 0.0
        
        contours_orig, _ = cv2.findContours(original_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_proc, _ = cv2.findContours(processed_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        orig_contour_lengths = [cv2.arcLength(cnt, False) for cnt in contours_orig]
        proc_contour_lengths = [cv2.arcLength(cnt, False) for cnt in contours_proc]
        
        orig_median_length = np.median(orig_contour_lengths) if len(orig_contour_lengths) > 0 else 0
        proc_median_length = np.median(proc_contour_lengths) if len(proc_contour_lengths) > 0 else 0
        
        if orig_median_length > 0:
            continuity_ratio = min(proc_median_length / orig_median_length, 2.0)
            edge_continuity_score = 1.0 - abs(1.0 - continuity_ratio)
        else:
            edge_continuity_score = 0.0
        
        cv2.imwrite(os.path.join(self.output_dir, 'edges_original.png'), original_canny)
        cv2.imwrite(os.path.join(self.output_dir, 'edges_processed.png'), processed_canny)
        cv2.imwrite(os.path.join(self.output_dir, 'edges_original_tight.png'), original_canny_tight)
        cv2.imwrite(os.path.join(self.output_dir, 'edges_processed_tight.png'), processed_canny_tight)
        
        vis_orig = cv2.cvtColor(self.original_gray, cv2.COLOR_GRAY2BGR)
        vis_proc = cv2.cvtColor(self.processed_gray, cv2.COLOR_GRAY2BGR)
        
        cv2.drawContours(vis_orig, contours_orig, -1, (0, 255, 0), 1)
        cv2.drawContours(vis_proc, contours_proc, -1, (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join(self.output_dir, 'edges_original_contours.png'), vis_orig)
        cv2.imwrite(os.path.join(self.output_dir, 'edges_processed_contours.png'), vis_proc)
        
        edge_quality_score = (
            edge_similarity * 0.4 +
            (edge_similarity_tight + edge_similarity_loose) * 0.15 +
            edge_density_score * 0.15 +
            edge_continuity_score * 0.15
        )
        
        return {
            "edge_similarity": edge_similarity,
            "edge_similarity_tight": edge_similarity_tight,
            "edge_similarity_loose": edge_similarity_loose,
            "edge_density_score": edge_density_score,
            "edge_continuity_score": edge_continuity_score,
            "edge_quality_score": edge_quality_score
        }
    
    def color_analysis(self):
        """Analyze color differences between images"""
        hist_similarity = 0
        for i in range(3):
            hist1 = cv2.calcHist([self.original], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([self.processed], [i], None, [256], [0, 256])
            hist_similarity += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_similarity /= 3.0 
        
        # Calculate average Delta-E (color difference)
        height, width = self.original.shape[:2]
        sample_points = 1000
        delta_e_sum = 0
        count = 0
        
        for _ in range(sample_points):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            lab1 = self.original_lab[y, x]
            lab2 = self.processed_lab[y, x]
            
            color1 = LabColor(lab1[0], lab1[1], lab1[2])
            color2 = LabColor(lab2[0], lab2[1], lab2[2])
            
            try:
                delta_e = delta_e_cie2000(color1, color2)
                delta_e_sum += delta_e
                count += 1
            except Exception:
                pass
        
        avg_delta_e = delta_e_sum / count if count > 0 else float('inf')
        
        color_stats = {}
        for i, channel in enumerate(['B', 'G', 'R']):
            orig_mean = np.mean(self.original[:,:,i])
            proc_mean = np.mean(self.processed[:,:,i])
            orig_std = np.std(self.original[:,:,i])
            proc_std = np.std(self.processed[:,:,i])
            
            color_stats[f"{channel}_mean_diff"] = abs(orig_mean - proc_mean)
            color_stats[f"{channel}_std_diff"] = abs(orig_std - proc_std)
        
        return {
            "histogram_similarity": hist_similarity,
            "avg_delta_e": avg_delta_e,
            **color_stats
        }
    
    def generate_visual_diff(self):
        """Generate visual difference map between images"""
        
        diff = cv2.absdiff(self.original, self.processed)
        
        diff_enhanced = cv2.convertScaleAbs(diff, alpha=5.0)
        
        diff_heat = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
        
        cv2.imwrite(os.path.join(self.output_dir, 'visual_diff.png'), diff_heat)
        
        significant_diff = np.sum(diff > 20) / (diff.shape[0] * diff.shape[1] * diff.shape[2])
        return {
            "significant_diff_percent": significant_diff * 100
        }
    
    def page_integrity_comparison(self, full_document=False, pages_reference=None, pages_processed=None):
        """
        Verify page completeness and correct page ordering.
        
        Parameters:
        - full_document: If True, uses external page collections provided in pages_reference and pages_processed
        - pages_reference: List of reference page images (for multi-page documents)
        - pages_processed: List of processed page images to verify (for multi-page documents)
        
        Returns dictionary with:
        - page_count_match: Boolean indicating if page counts match
        - page_ordering_correct: Boolean indicating if pages are in correct order
        - correct_page_count: Number of expected pages
        - actual_page_count: Number of detected pages
        - page_completeness_score: Score between 0-1 indicating overall page integrity
        - missing_pages: List of missing page numbers (if any)
        - duplicated_pages: List of duplicated page numbers (if any)
        - reordered_pages: Dictionary mapping detected page to expected position
        """
        
        if full_document:
            if pages_reference is None or pages_processed is None:
                return {
                    "page_count_match": False,
                    "page_ordering_correct": False,
                    "page_completeness_score": 0,
                    "error": "Reference or processed page collections not provided"
                }
            
            ref_pages = pages_reference
            proc_pages = pages_processed
        else:
            return {
                "page_count_match": True, 
                "page_ordering_correct": True,
                "correct_page_count": 1,
                "actual_page_count": 1,
                "page_completeness_score": 1.0,
                "missing_pages": [],
                "duplicated_pages": []
            }
        
        correct_page_count = len(ref_pages)
        actual_page_count = len(proc_pages)
        page_count_match = correct_page_count == actual_page_count
        
        page_mapping = {}
        page_confidence = {}
        
        for i, proc_page in enumerate(proc_pages):
            best_match_idx = -1
            best_match_confidence = -1
            
            for j, ref_page in enumerate(ref_pages):
                if len(proc_page.shape) > 2:
                    proc_gray = cv2.cvtColor(proc_page, cv2.COLOR_BGR2GRAY)
                else:
                    proc_gray = proc_page
                    
                if len(ref_page.shape) > 2:
                    ref_gray = cv2.cvtColor(ref_page, cv2.COLOR_BGR2GRAY)
                else:
                    ref_gray = ref_page
                    
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(ref_gray, None)
                kp2, des2 = orb.detectAndCompute(proc_gray, None)
                
                if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    if len(matches) > 0:
                        avg_distance = sum(m.distance for m in matches) / len(matches)
                        num_matches = len(matches)
                        confidence = num_matches / (1 + avg_distance/100)
                        
                        if confidence > best_match_confidence:
                            best_match_confidence = confidence
                            best_match_idx = j
            
            if best_match_idx >= 0:
                page_mapping[i] = best_match_idx
                page_confidence[i] = best_match_confidence
        
        detected_pages = list(page_mapping.values())
        missing_pages = [i for i in range(correct_page_count) if i not in detected_pages]
        duplicated_pages = [page for page, count in collections.Counter(detected_pages).items() if count > 1]
        
        page_ordering_correct = True
        reordered_pages = {}
        
        for i in range(len(page_mapping) - 1):
            if i in page_mapping and i+1 in page_mapping:
                if page_mapping[i] >= page_mapping[i+1]:
                    page_ordering_correct = False
                    reordered_pages[i+1] = page_mapping[i+1]
        
        if correct_page_count == 0:
            page_completeness_score = 0
        else:
            missing_penalty = len(missing_pages) / correct_page_count if correct_page_count > 0 else 0
            duplicate_penalty = len(duplicated_pages) / correct_page_count if correct_page_count > 0 else 0
            reorder_penalty = len(reordered_pages) / correct_page_count if correct_page_count > 0 else 0
            
            page_completeness_score = 1.0 - (missing_penalty * 0.5 + duplicate_penalty * 0.3 + reorder_penalty * 0.2)
            page_completeness_score = max(0, min(1, page_completeness_score))  # Clip to [0,1]
        
        if len(proc_pages) > 0 and len(ref_pages) > 0:
            vis_height = 200
            vis_width = max(correct_page_count, actual_page_count) * 120
            visualization = np.ones((vis_height * 2 + 50, vis_width, 3), dtype=np.uint8) * 255
            
            for i in range(correct_page_count):
                left = i * 120 + 10
                cv2.rectangle(visualization, (left, 10), (left + 100, 160), (0, 0, 0), 2)
                cv2.putText(visualization, f"Ref #{i+1}", (left + 10, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                if i < len(ref_pages):
                    page_img = ref_pages[i]
                    if len(page_img.shape) > 2:
                        thumb = cv2.resize(page_img, (80, 80))
                        visualization[60:140, left+10:left+90] = thumb
            
            for i in range(actual_page_count):
                left = i * 120 + 10
                cv2.rectangle(visualization, (left, vis_height + 10), (left + 100, vis_height + 160), 
                           (0, 0, 0), 2)
                cv2.putText(visualization, f"Proc #{i+1}", (left + 10, vis_height + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                if i < len(proc_pages):
                    page_img = proc_pages[i]
                    if len(page_img.shape) > 2:
                        thumb = cv2.resize(page_img, (80, 80))
                        visualization[vis_height + 60:vis_height + 140, left+10:left+90] = thumb
                
                if i in page_mapping:
                    ref_idx = page_mapping[i]
                    ref_left = ref_idx * 120 + 60
                    proc_left = i * 120 + 60
                    color = (0, 255, 0) if ref_idx == i else (0, 0, 255)
                    cv2.line(visualization, (ref_left, 160), (proc_left, vis_height + 10), color, 2)
                    
                    if i in page_confidence:
                        conf_text = f"{page_confidence[i]:.2f}"
                        text_pos = ((ref_left + proc_left) // 2, (160 + vis_height + 10) // 2)
                        cv2.putText(visualization, conf_text, text_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.imwrite(os.path.join(self.output_dir, 'page_integrity_map.png'), visualization)
        
        return {
            "page_count_match": page_count_match,
            "page_ordering_correct": page_ordering_correct,
            "correct_page_count": correct_page_count,
            "actual_page_count": actual_page_count,
            "page_completeness_score": page_completeness_score,
            "missing_pages": missing_pages,
            "duplicated_pages": duplicated_pages,
            "reordered_pages": reordered_pages,
            "page_mapping": page_mapping
        }
    
    def run_all_comparisons(self, full_document=False, pages_reference=None, pages_processed=None):
        """Run all comparison methods and return consolidated results"""
        results = {}
        results.update(self.basic_metrics())
        results.update(self.feature_detection())
        results.update(self.edge_comparison())
        results.update(self.enhanced_edge_comparison())
        results.update(self.color_analysis())
        results.update(self.generate_visual_diff())
        
        if full_document:
            results.update(self.page_integrity_comparison(True, pages_reference, pages_processed))
        
        psnr_factor = 1.0 if results["psnr"] == float('inf') else min(results["psnr"] / 50.0, 1.0)
        
        quality_score = (
            results["ssim"] * 0.20 + 
            (1.0 - min(results["mse"] / 10000.0, 1.0)) * 0.1 + 
            psnr_factor * 0.1 + 
            results["match_confidence"] * 0.15 + 
            results["edge_similarity"] * 0.05 +
            results["edge_quality_score"] * 0.15 +
            results["histogram_similarity"] * 0.1
        )
        
        if "page_completeness_score" in results:
            quality_score = quality_score * 0.85 + results["page_completeness_score"] * 0.15
        
        results["overall_quality"] = quality_score
        
        return results

if __name__ == "__main__":
    comparator = ImageComparator('sample.png', 'output.png')
    results = comparator.run_all_comparisons()
    
    print("\n=== Image Comparison Results ===")
    print(f"SSIM Score: {results['ssim']:.4f} (higher is better, 1.0 is perfect)")
    print(f"MSE Score: {results['mse']:.2f} (lower is better, 0 is perfect)")
    print(f"PSNR Score: {results['psnr']:.2f} dB (higher is better)")
    print(f"Feature Match Confidence: {results['match_confidence']:.4f} (higher is better)")
    
    print("\n=== Edge Quality Analysis ===")
    print(f"Basic Edge Similarity: {results['edge_similarity']:.4f} (higher is better)")
    print(f"Fine Detail Edge Preservation: {results['edge_similarity_tight']:.4f} (higher is better)")
    print(f"Coarse Edge Preservation: {results['edge_similarity_loose']:.4f} (higher is better)")
    print(f"Edge Density Preservation: {results['edge_density_score']:.4f} (higher is better)")
    print(f"Edge Continuity Score: {results['edge_continuity_score']:.4f} (higher is better)")
    print(f"Overall Edge Quality: {results['edge_quality_score']:.4f} (higher is better)")
    
    print(f"Color Histogram Similarity: {results['histogram_similarity']:.4f} (higher is better)")
    print(f"Average Color Difference (Delta-E): {results['avg_delta_e']:.2f} (lower is better)")
    print(f"Significant Difference: {results['significant_diff_percent']:.2f}% of pixels")
    print(f"Overall Quality Score: {results['overall_quality']:.4f} (higher is better)")
    
    print("\nVisualization files generated:")
    print("- feature_matches.png: Matching features between images")
    print("- edges_original.png: Edge detection for original image")
    print("- edges_processed.png: Edge detection for processed image")
    print("- edges_original_tight.png: Fine detail edge detection for original image")
    print("- edges_processed_tight.png: Fine detail edge detection for processed image")
    print("- edges_original_contours.png: Contour visualization for original image")
    print("- edges_processed_contours.png: Contour visualization for processed image")
    print("- visual_diff.png: Heatmap showing differences between images")