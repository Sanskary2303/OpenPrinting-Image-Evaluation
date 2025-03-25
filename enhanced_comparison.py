import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

class ImageComparator:
    def __init__(self, original_path, processed_path):
        self.original = cv2.imread(original_path)
        self.processed = cv2.imread(processed_path)
        
        if self.original.shape != self.processed.shape:
            self.processed = cv2.resize(self.processed, 
                                        (self.original.shape[1], self.original.shape[0]))
        
        self.original_gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.processed_gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
        
        self.original_lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        self.processed_lab = cv2.cvtColor(self.processed, cv2.COLOR_BGR2LAB)

    def basic_metrics(self):
        """Calculate basic image similarity metrics"""
        ssim_score, ssim_diff = ssim(self.original_gray, self.processed_gray, full=True)
        
        mse_score = mean_squared_error(self.original_gray, self.processed_gray)
        
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
            cv2.imwrite('feature_matches.png', matched_img)
            
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
        
        cv2.imwrite('edges_original.png', edges1)
        cv2.imwrite('edges_processed.png', edges2)
        
        return {
            "edge_similarity": edge_score
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
        # Sample points to avoid excessive computation
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
        
        cv2.imwrite('visual_diff.png', diff_heat)
        
        significant_diff = np.sum(diff > 20) / (diff.shape[0] * diff.shape[1] * diff.shape[2])
        return {
            "significant_diff_percent": significant_diff * 100
        }
    
    def run_all_comparisons(self):
        """Run all comparison methods and return consolidated results"""
        results = {}
        results.update(self.basic_metrics())
        results.update(self.feature_detection())
        results.update(self.edge_comparison())
        results.update(self.color_analysis())
        results.update(self.generate_visual_diff())
        
        quality_score = (
            results["ssim"] * 0.3 + 
            (1.0 - min(results["mse"] / 10000.0, 1.0)) * 0.1 + 
            (min(results["psnr"] / 50.0, 1.0)) * 0.1 + 
            results["match_confidence"] * 0.2 + 
            results["edge_similarity"] * 0.1 + 
            results["histogram_similarity"] * 0.2
        )
        
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
    print(f"Edge Similarity: {results['edge_similarity']:.4f} (higher is better)")
    print(f"Color Histogram Similarity: {results['histogram_similarity']:.4f} (higher is better)")
    print(f"Average Color Difference (Delta-E): {results['avg_delta_e']:.2f} (lower is better)")
    print(f"Significant Difference: {results['significant_diff_percent']:.2f}% of pixels")
    print(f"Overall Quality Score: {results['overall_quality']:.4f} (higher is better)")
    
    print("\nVisualization files generated:")
    print("- feature_matches.png: Matching features between images")
    print("- edges_original.png: Edge detection for original image")
    print("- edges_processed.png: Edge detection for processed image")
    print("- visual_diff.png: Heatmap showing differences between images")