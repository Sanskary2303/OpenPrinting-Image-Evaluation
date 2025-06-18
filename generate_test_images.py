import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import math

class ComprehensiveTestImageGenerator:
    """Generate comprehensive test images for all image processing algorithms"""
    
    def __init__(self, output_dir="comprehensive_test_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard dimensions (A4 at 300 DPI equivalent)
        self.width = 2550
        self.height = 3508
    
    def create_grid_patterns(self):
        """Create various grid patterns for density estimation testing"""
        patterns = {}
        
        # 1. Vertical Grid Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for x in range(0, self.width, 20):  # Vertical lines every 20 pixels
            cv2.line(img, (x, 0), (x, self.height), 0, 2)
        patterns['vertical_grid'] = img
        
        # 2. Horizontal Grid Pattern  
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 20):  # Horizontal lines every 20 pixels
            cv2.line(img, (0, y), (self.width, y), 0, 2)
        patterns['horizontal_grid'] = img
        
        # 3. Square Grid Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for x in range(0, self.width, 20):
            cv2.line(img, (x, 0), (x, self.height), 0, 1)
        for y in range(0, self.height, 20):
            cv2.line(img, (0, y), (self.width, y), 0, 1)
        patterns['square_grid'] = img
        
        # 4. Diagonal Grid Pattern (45 degrees)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        spacing = 30
        # Draw diagonal lines from top-left to bottom-right
        for offset in range(-self.height, self.width, spacing):
            start_x = max(0, offset)
            start_y = max(0, -offset)
            end_x = min(self.width, offset + self.height)
            end_y = min(self.height, self.height - offset)
            cv2.line(img, (start_x, start_y), (end_x, end_y), 0, 2)
        patterns['diagonal_45_grid'] = img
        
        # 5. Diagonal Grid Pattern (-45 degrees)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Draw diagonal lines from top-right to bottom-left
        for offset in range(0, self.width + self.height, spacing):
            start_x = min(self.width, offset)
            start_y = max(0, offset - self.width)
            end_x = max(0, offset - self.height)
            end_y = min(self.height, offset)
            cv2.line(img, (start_x, start_y), (end_x, end_y), 0, 2)
        patterns['diagonal_minus45_grid'] = img
        
        # 6. Double Diagonal Grid (Cross-hatch)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        spacing = 40
        # 45-degree lines
        for offset in range(-self.height, self.width, spacing):
            start_x = max(0, offset)
            start_y = max(0, -offset)
            end_x = min(self.width, offset + self.height)
            end_y = min(self.height, self.height - offset)
            cv2.line(img, (start_x, start_y), (end_x, end_y), 0, 1)
        # -45-degree lines
        for offset in range(0, self.width + self.height, spacing):
            start_x = min(self.width, offset)
            start_y = max(0, offset - self.width)
            end_x = max(0, offset - self.height)
            end_y = min(self.height, offset)
            cv2.line(img, (start_x, start_y), (end_x, end_y), 0, 1)
        patterns['crosshatch_grid'] = img
        
        # 7. Variable Density Grid
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 20):
            # Density increases from left to right
            density = int(20 * (1 - y / self.height) + 5)
            for x in range(0, self.width, density):
                cv2.line(img, (x, y), (x, y + 10), 0, 1)
        patterns['variable_density_grid'] = img
        
        # 8. Radial Grid Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        center_x, center_y = self.width // 2, self.height // 2
        num_rays = 24
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays
            end_x = int(center_x + 1000 * math.cos(angle))
            end_y = int(center_y + 1000 * math.sin(angle))
            cv2.line(img, (center_x, center_y), (end_x, end_y), 0, 1)
        # Add concentric circles
        for radius in range(50, 800, 50):
            cv2.circle(img, (center_x, center_y), radius, 0, 1)
        patterns['radial_grid'] = img
        
        return patterns
    
    def create_density_patterns(self):
        """Create patterns with varying content density"""
        patterns = {}
        
        # 1. High Density Text Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(50, self.height - 50, 25):
            for x in range(50, self.width - 200, 180):
                # Simulate text blocks
                cv2.rectangle(img, (x, y), (x + 150, y + 20), 0, -1)
                # Add spacing between words
                for word_x in range(x + 10, x + 140, 30):
                    cv2.rectangle(img, (word_x, y + 3), (word_x + 20, y + 17), 255, -1)
        patterns['high_density_text'] = img
        
        # 2. Medium Density Mixed Content
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Add text blocks with more spacing
        for y in range(100, self.height - 100, 80):
            for x in range(100, self.width - 300, 300):
                cv2.rectangle(img, (x, y), (x + 200, y + 30), 0, -1)
                # Add breaks in text
                for break_x in range(x + 30, x + 170, 40):
                    cv2.rectangle(img, (break_x, y + 5), (break_x + 20, y + 25), 255, -1)
        # Add some geometric elements
        for i, y in enumerate(range(200, self.height - 200, 400)):
            x = 200 + (i % 3) * 600
            cv2.rectangle(img, (x, y), (x + 150, y + 150), 0, 2)
            cv2.circle(img, (x + 300, y + 75), 50, 0, 2)
        patterns['medium_density_mixed'] = img
        
        # 3. Low Density Sparse Content
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Sparse text
        for y in range(200, self.height - 200, 200):
            for x in range(200, self.width - 400, 600):
                cv2.rectangle(img, (x, y), (x + 150, y + 25), 0, -1)
        # Few geometric shapes
        cv2.rectangle(img, (self.width//2 - 100, self.height//2 - 100), 
                     (self.width//2 + 100, self.height//2 + 100), 0, 3)
        cv2.circle(img, (self.width//4, self.height//4), 80, 0, 3)
        patterns['low_density_sparse'] = img
        
        # 4. Gradient Density Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 20):
            density_factor = y / self.height  # 0 to 1 from top to bottom
            spacing = int(10 + 40 * (1 - density_factor))
            for x in range(0, self.width, spacing):
                size = int(5 + 10 * density_factor)
                cv2.rectangle(img, (x, y), (x + size, y + size), 0, -1)
        patterns['gradient_density'] = img
        
        # 5. Clustered Density Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Create dense clusters
        cluster_centers = [(400, 400), (1200, 800), (800, 1600), (1800, 2000), (600, 2800)]
        for cx, cy in cluster_centers:
            for i in range(50):  # 50 elements per cluster
                angle = 2 * math.pi * i / 50
                radius = np.random.normal(80, 30)
                x = int(cx + radius * math.cos(angle))
                y = int(cy + radius * math.sin(angle))
                if 0 <= x < self.width - 20 and 0 <= y < self.height - 20:
                    size = np.random.randint(5, 15)
                    cv2.rectangle(img, (x, y), (x + size, y + size), 0, -1)
        patterns['clustered_density'] = img
        
        return patterns
    
    def create_rotation_test_patterns(self):
        """Create patterns specifically for rotation detection testing"""
        patterns = {}
        
        # 1. Horizontal Lines Pattern (0 degrees)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 40):
            cv2.line(img, (100, y), (self.width - 100, y), 0, 3)
        patterns['horizontal_lines'] = img
        
        # 2. Vertical Lines Pattern (90 degrees)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for x in range(100, self.width - 100, 40):
            cv2.line(img, (x, 100), (x, self.height - 100), 0, 3)
        patterns['vertical_lines'] = img
        
        # 3. Slightly Rotated Pattern (5 degrees)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 40):
            cv2.line(img, (100, y), (self.width - 100, y), 0, 3)
        # Rotate the image
        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (self.width, self.height), 
                           borderValue=255)
        patterns['rotated_5_degrees'] = img
        
        # 4. Mixed Angle Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Horizontal lines
        for y in range(200, 800, 60):
            cv2.line(img, (100, y), (self.width - 100, y), 0, 2)
        # Vertical lines
        for x in range(200, 1000, 60):
            cv2.line(img, (x, 1000), (x, self.height - 100), 0, 2)
        # Diagonal lines
        for i in range(10):
            start_x = 1200 + i * 80
            cv2.line(img, (start_x, 200), (start_x + 300, 800), 0, 2)
        patterns['mixed_angles'] = img
        
        # 5. Text-like Horizontal Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(150, self.height - 150, 50):
            # Simulate text lines
            for x in range(100, self.width - 200, 200):
                cv2.rectangle(img, (x, y), (x + 150, y + 20), 0, -1)
                # Add word breaks
                for break_x in range(x + 30, x + 120, 30):
                    cv2.rectangle(img, (break_x, y + 3), (break_x + 15, y + 17), 255, -1)
        patterns['text_horizontal'] = img
        
        return patterns
    
    def create_noise_test_patterns(self):
        """Create patterns for noise analysis testing"""
        patterns = {}
        
        # 1. Clean Pattern (No Noise)
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 50):
            cv2.rectangle(img, (100, y), (self.width - 100, y + 30), 0, -1)
        patterns['clean_no_noise'] = img
        
        # 2. Gaussian Noise Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 50):
            cv2.rectangle(img, (100, y), (self.width - 100, y + 30), 0, -1)
        # Add Gaussian noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        patterns['gaussian_noise'] = img
        
        # 3. Salt and Pepper Noise Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 50):
            cv2.rectangle(img, (100, y), (self.width - 100, y + 30), 0, -1)
        # Add salt and pepper noise
        salt_pepper = np.random.random(img.shape)
        img[salt_pepper < 0.01] = 0  # Pepper
        img[salt_pepper > 0.99] = 255  # Salt
        patterns['salt_pepper_noise'] = img
        
        # 4. Uniform Noise Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 50):
            cv2.rectangle(img, (100, y), (self.width - 100, y + 30), 0, -1)
        # Add uniform noise
        noise = np.random.uniform(-20, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        patterns['uniform_noise'] = img
        
        # 5. Speckle Noise Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(100, self.height - 100, 50):
            cv2.rectangle(img, (100, y), (self.width - 100, y + 30), 0, -1)
        # Add speckle noise
        speckle = np.random.random(img.shape)
        img = img + img * speckle * 0.2
        img = np.clip(img, 0, 255).astype(np.uint8)
        patterns['speckle_noise'] = img
        
        return patterns
    
    def create_texture_test_patterns(self):
        """Create patterns for texture analysis testing"""
        patterns = {}
        
        # 1. Fine Texture Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 3):
            for x in range(0, self.width, 3):
                if (x + y) % 6 == 0:
                    img[y:y+2, x:x+2] = 0
        patterns['fine_texture'] = img
        
        # 2. Coarse Texture Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 20):
            for x in range(0, self.width, 20):
                if (x//20 + y//20) % 2 == 0:
                    cv2.rectangle(img, (x, y), (x+15, y+15), 0, -1)
        patterns['coarse_texture'] = img
        
        # 3. Directional Texture Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        for y in range(0, self.height, 4):
            for x in range(y % 8, self.width, 8):
                cv2.line(img, (x, y), (x+4, y), 0, 1)
        patterns['directional_texture'] = img
        
        # 4. Random Texture Pattern
        img = np.random.randint(200, 256, (self.height, self.width), dtype=np.uint8)
        # Add some structure
        for i in range(0, self.height, 50):
            for j in range(0, self.width, 50):
                cv2.rectangle(img, (j, i), (j+40, i+40), 0, 2)
        patterns['random_texture'] = img
        
        # 5. Smooth Gradient Pattern
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                # Create radial gradient
                center_x, center_y = self.width // 2, self.height // 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                intensity = int(255 * (1 - distance / max_distance))
                img[y, x] = max(0, min(255, intensity))
        patterns['smooth_gradient'] = img
        
        return patterns
    
    def create_color_test_patterns(self):
        """Create patterns for color/monochrome detection testing"""
        patterns = {}
        
        # 1. True RGB Color Pattern
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        # Add colored rectangles
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]
        for i, color in enumerate(colors):
            x = (i % 3) * (self.width // 3)
            y = (i // 3) * (self.height // 2)
            cv2.rectangle(img, (x + 50, y + 50), 
                         (x + self.width//3 - 50, y + self.height//2 - 50), 
                         color, -1)
        patterns['true_color_rgb'] = img
        
        # 2. RGB-Encoded Grayscale Pattern
        gray_img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Add grayscale content
        for y in range(100, self.height - 100, 80):
            for x in range(100, self.width - 100, 200):
                gray_val = np.random.randint(50, 200)
                cv2.rectangle(gray_img, (x, y), (x + 150, y + 50), gray_val, -1)
        # Convert to RGB (all channels identical)
        img = np.stack([gray_img, gray_img, gray_img], axis=2)
        patterns['rgb_encoded_grayscale'] = img
        
        # 3. Low Saturation Color Pattern
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        # Add very muted colors
        muted_colors = [(220, 210, 210), (210, 220, 210), (210, 210, 220), 
                       (225, 225, 210), (225, 210, 225), (210, 225, 225)]
        for i, color in enumerate(muted_colors):
            x = (i % 3) * (self.width // 3)
            y = (i // 3) * (self.height // 2)
            cv2.rectangle(img, (x + 50, y + 50), 
                         (x + self.width//3 - 50, y + self.height//2 - 50), 
                         color, -1)
        patterns['low_saturation_color'] = img
        
        # 4. Mixed Color and Grayscale Pattern
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        # Mostly grayscale with some color elements
        for y in range(100, self.height - 100, 60):
            for x in range(100, self.width - 200, 300):
                gray_val = np.random.randint(100, 200)
                cv2.rectangle(img, (x, y), (x + 200, y + 40), 
                             (gray_val, gray_val, gray_val), -1)
        # Add a few color elements
        cv2.rectangle(img, (200, 200), (400, 400), (255, 0, 0), -1)  # Red
        cv2.rectangle(img, (1000, 800), (1200, 1000), (0, 255, 0), -1)  # Green
        cv2.circle(img, (1500, 1500), 100, (0, 0, 255), -1)  # Blue
        patterns['mixed_color_grayscale'] = img
        
        return patterns
    
    def create_geometric_test_patterns(self):
        """Create patterns for geometric analysis testing"""
        patterns = {}
        
        # 1. Perfect Rectangles Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        rectangles = [(200, 200, 400, 300), (600, 500, 300, 200), 
                     (1000, 800, 500, 400), (300, 1200, 350, 250)]
        for x, y, w, h in rectangles:
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 3)
        patterns['perfect_rectangles'] = img
        
        # 2. Perfect Circles Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        circles = [(400, 400, 150), (1000, 600, 100), (700, 1200, 200), (1500, 1000, 120)]
        for x, y, r in circles:
            cv2.circle(img, (x, y), r, 0, 3)
        patterns['perfect_circles'] = img
        
        # 3. Distorted Shapes Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Create slightly distorted rectangles
        points1 = np.array([[200, 200], [600, 210], [590, 500], [190, 490]], np.int32)
        points2 = np.array([[800, 300], [1200, 290], [1210, 600], [810, 610]], np.int32)
        cv2.polylines(img, [points1, points2], True, 0, 3)
        patterns['distorted_shapes'] = img
        
        # 4. Mixed Geometric Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        # Rectangles, circles, and triangles
        cv2.rectangle(img, (100, 100), (400, 300), 0, 3)
        cv2.circle(img, (600, 200), 100, 0, 3)
        triangle = np.array([[800, 100], [900, 300], [700, 300]], np.int32)
        cv2.polylines(img, [triangle], True, 0, 3)
        # Add an ellipse
        cv2.ellipse(img, (1200, 200), (150, 100), 0, 0, 360, 0, 3)
        patterns['mixed_geometric'] = img
        
        # 5. Symmetry Test Pattern
        img = np.ones((self.height, self.width), dtype=np.uint8) * 255
        center_x, center_y = self.width // 2, self.height // 2
        # Create symmetric pattern
        for i in range(5):
            offset = 100 + i * 80
            cv2.rectangle(img, (center_x - offset, center_y - 50), 
                         (center_x - offset + 50, center_y + 50), 0, -1)
            cv2.rectangle(img, (center_x + offset - 50, center_y - 50), 
                         (center_x + offset, center_y + 50), 0, -1)
        patterns['symmetry_test'] = img
        
        return patterns
    
    def generate_comprehensive_test_suite(self):
        """Generate complete test suite for all algorithms"""
        print("Generating comprehensive test image suite...")
        
        all_patterns = {}
        
        # Generate all pattern types
        all_patterns.update(self.create_grid_patterns())
        all_patterns.update(self.create_density_patterns())
        all_patterns.update(self.create_rotation_test_patterns())
        all_patterns.update(self.create_noise_test_patterns())
        all_patterns.update(self.create_texture_test_patterns())
        all_patterns.update(self.create_color_test_patterns())
        all_patterns.update(self.create_geometric_test_patterns())
        
        # Save all patterns
        generated_files = []
        for pattern_name, pattern_image in all_patterns.items():
            filename = f"{pattern_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, pattern_image)
            generated_files.append(filepath)
            print(f"Created: {filename}")
        
        # Create a summary of what was generated
        self.create_test_suite_summary(all_patterns)
        
        return generated_files
    
    def create_test_suite_summary(self, patterns):
        """Create a summary document of all generated test patterns"""
        summary_path = os.path.join(self.output_dir, "test_suite_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE TEST IMAGE SUITE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("GRID PATTERNS (for density estimation):\n")
            f.write("- vertical_grid.png: Vertical lines for vertical density testing\n")
            f.write("- horizontal_grid.png: Horizontal lines for horizontal density testing\n")
            f.write("- square_grid.png: Regular square grid pattern\n")
            f.write("- diagonal_45_grid.png: 45-degree diagonal lines\n")
            f.write("- diagonal_minus45_grid.png: -45-degree diagonal lines\n")
            f.write("- crosshatch_grid.png: Cross-hatched diagonal pattern\n")
            f.write("- variable_density_grid.png: Grid with varying density\n")
            f.write("- radial_grid.png: Radial and circular grid pattern\n\n")
            
            f.write("DENSITY PATTERNS:\n")
            f.write("- high_density_text.png: Dense text-like content\n")
            f.write("- medium_density_mixed.png: Medium density mixed content\n")
            f.write("- low_density_sparse.png: Sparse, low-density content\n")
            f.write("- gradient_density.png: Gradually changing density\n")
            f.write("- clustered_density.png: Content in distinct clusters\n\n")
            
            f.write("ROTATION TEST PATTERNS:\n")
            f.write("- horizontal_lines.png: Perfect horizontal lines (0°)\n")
            f.write("- vertical_lines.png: Perfect vertical lines (90°)\n")
            f.write("- rotated_5_degrees.png: Slightly rotated content (5°)\n")
            f.write("- mixed_angles.png: Multiple angle orientations\n")
            f.write("- text_horizontal.png: Text-like horizontal patterns\n\n")
            
            f.write("NOISE TEST PATTERNS:\n")
            f.write("- clean_no_noise.png: Clean pattern without noise\n")
            f.write("- gaussian_noise.png: Pattern with Gaussian noise\n")
            f.write("- salt_pepper_noise.png: Pattern with salt & pepper noise\n")
            f.write("- uniform_noise.png: Pattern with uniform noise\n")
            f.write("- speckle_noise.png: Pattern with speckle noise\n\n")
            
            f.write("TEXTURE TEST PATTERNS:\n")
            f.write("- fine_texture.png: Fine, detailed texture\n")
            f.write("- coarse_texture.png: Coarse, blocky texture\n")
            f.write("- directional_texture.png: Directional texture pattern\n")
            f.write("- random_texture.png: Random texture with structure\n")
            f.write("- smooth_gradient.png: Smooth gradient (minimal texture)\n\n")
            
            f.write("COLOR TEST PATTERNS:\n")
            f.write("- true_color_rgb.png: True RGB color image\n")
            f.write("- rgb_encoded_grayscale.png: Grayscale encoded as RGB\n")
            f.write("- low_saturation_color.png: Very muted colors\n")
            f.write("- mixed_color_grayscale.png: Mixed color and grayscale\n\n")
            
            f.write("GEOMETRIC TEST PATTERNS:\n")
            f.write("- perfect_rectangles.png: Perfect rectangular shapes\n")
            f.write("- perfect_circles.png: Perfect circular shapes\n")
            f.write("- distorted_shapes.png: Slightly distorted shapes\n")
            f.write("- mixed_geometric.png: Mixed geometric shapes\n")
            f.write("- symmetry_test.png: Symmetric pattern for symmetry analysis\n\n")
            
            f.write(f"Total patterns generated: {len(patterns)}\n")
            f.write(f"Output directory: {self.output_dir}\n")
        
        print(f"Summary saved to: {summary_path}")


# Legacy compatibility function
def generate_from_existing_image(input_path, output_dir="derived_test_images"):
    """Generate DPI variants from an existing high-quality image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {input_path}")
        return []
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save original as 600 DPI reference
    original_path = os.path.join(output_dir, f"{base_name}_reference_600dpi.png")
    cv2.imwrite(original_path, img)
    
    # Create 300 DPI version
    h, w = img.shape
    img_300 = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    img_300 = cv2.resize(img_300, (w, h), interpolation=cv2.INTER_CUBIC)
    path_300 = os.path.join(output_dir, f"{base_name}_effective_300dpi.png")
    cv2.imwrite(path_300, img_300)
    
    # Create 150 DPI version
    img_150 = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_AREA)
    img_150 = cv2.resize(img_150, (w, h), interpolation=cv2.INTER_CUBIC)
    path_150 = os.path.join(output_dir, f"{base_name}_effective_150dpi.png")
    cv2.imwrite(path_150, img_150)
    
    print(f"Generated DPI variants in {output_dir}")
    return [original_path, path_300, path_150]


if __name__ == "__main__":
    # Generate comprehensive test suite
    generator = ComprehensiveTestImageGenerator()
    test_images = generator.generate_comprehensive_test_suite()
    
    print(f"\nGenerated {len(test_images)} test images in '{generator.output_dir}'")
    print("Check 'test_suite_summary.txt' for details on each pattern type.")
    
    # If you have existing images, generate variants from them
    existing_images = ["600dpi.png", "300dpi.png", "150dpi.png"]
    for img_path in existing_images:
        if os.path.exists(img_path):
            derived_images = generate_from_existing_image(img_path)
            print(f"\nGenerated variants from {img_path}:")
            for derived in derived_images:
                print(f"  - {derived}")
