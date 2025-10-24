#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import tempfile
import shutil
import re
import argparse
import cups
import logging
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Verify correct pycups module is imported
if not hasattr(cups, 'Connection'):
    raise ImportError(
        "Wrong 'cups' module detected. The 'cups' package from pip conflicts with the required 'pycups' library.\n"
        "Please uninstall the 'cups' package and ensure 'pycups' is installed:\n"
        "  pip uninstall cups\n"
        "  pip install pycups"
    )

from printer_info import (
    get_available_printers, 
    get_printer_model, 
    get_filter_chain, 
    get_printer_ppd_info,
    get_comprehensive_printer_info
)
from filter_chain import get_filter_chain as get_filter_chain_advanced
from generate_test_images import ComprehensiveTestImageGenerator
from enhanced_comparison import ImageComparator

class PrintQualityTestPipeline:
    """
    Complete testing pipeline that takes only a printer queue name
    and automatically tests all supported modes and filter chains.
    """
    
    def __init__(self, printer_queue: str, output_dir: Optional[str] = None):
        self.printer_queue = printer_queue
        self.output_dir = output_dir or f"pipeline_results_{printer_queue}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.setup_logging()
        
        self.image_generator = ComprehensiveTestImageGenerator(self.output_dir)
        
        self.printer_info = None
        self.supported_modes = []
        self.filter_chains = {}
        self.test_images = []
        self.results = {}
        
        self.create_output_structure()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_structure(self):
        """Create organized output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dirs = {
            'test_images': os.path.join(self.output_dir, 'test_images'),
            'printed_outputs': os.path.join(self.output_dir, 'printed_outputs'),
            'analysis_results': os.path.join(self.output_dir, 'analysis_results'),
            'comparisons': os.path.join(self.output_dir, 'comparisons'),
            'reports': os.path.join(self.output_dir, 'reports'),
            'logs': os.path.join(self.output_dir, 'logs')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(dir_path, exist_ok=True)
            
    def analyze_printer_capabilities(self) -> Dict[str, Any]:
        """Automatically detect printer capabilities and supported modes"""
        self.logger.info(f"Analyzing printer capabilities for: {self.printer_queue}")
        
        try:
            available_printers = get_available_printers()
            if self.printer_queue not in available_printers:
                raise ValueError(f"Printer queue '{self.printer_queue}' not found. Available: {available_printers}")
            
            self.printer_info = {
                'name': self.printer_queue,
                'model': get_printer_model(self.printer_queue),
                'ppd_info': get_printer_ppd_info(self.printer_queue),
                'comprehensive_info': get_comprehensive_printer_info(self.printer_queue)
            }
            
            self.supported_modes = self._extract_supported_modes()
            
            self.filter_chains = self._get_filter_chains_for_modes()
            
            self.logger.info(f"Found {len(self.supported_modes)} supported modes")
            self.logger.info(f"Supported modes: {[mode['name'] for mode in self.supported_modes]}")
            
            return {
                'printer_info': self.printer_info,
                'supported_modes': self.supported_modes,
                'filter_chains': self.filter_chains
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze printer capabilities: {e}")
            raise
            
    def _extract_supported_modes(self) -> List[Dict[str, Any]]:
        """Extract supported print modes from printer PPD"""
        modes = []
        
        ppd_info = self.printer_info.get('ppd_info', {}) if self.printer_info else {}
        if not ppd_info:
            self.logger.warning("No PPD info found, using default modes")
            return self._get_default_modes()
        
        try:
            conn = cups.Connection()
            ppd_path = conn.getPPD(self.printer_queue)
            
            if ppd_path and os.path.exists(ppd_path):
                with open(ppd_path, 'r', encoding='utf-8', errors='ignore') as f:
                    ppd_content = f.read()
                
                os.unlink(ppd_path)
                
                color_modes = self._parse_ppd_option(ppd_content, 'ColorModel')
                print_qualities = self._parse_ppd_option(ppd_content, 'Quality') or \
                                self._parse_ppd_option(ppd_content, 'PrintQuality')
                resolutions = self._parse_ppd_option(ppd_content, 'Resolution')
                media_types = self._parse_ppd_option(ppd_content, 'MediaType')
                
                modes = self._generate_mode_combinations(
                    color_modes, print_qualities, resolutions, media_types
                )
            else:
                self.logger.warning("Could not get PPD file, using default modes")
                modes = self._get_default_modes()
            
        except Exception as e:
            self.logger.warning(f"Failed to parse PPD file: {e}, using defaults")
            modes = self._get_default_modes()
            
        return modes
        
    def _parse_ppd_option(self, ppd_content: str, option_name: str) -> List[str]:
        """Parse PPD option values"""
        import re
        
        pattern = rf'\*{option_name}\s+([^:]+):'
        matches = re.findall(pattern, ppd_content, re.MULTILINE)
        
        if not matches:
            pattern = rf'\*{option_name}.*?:\s*"([^"]+)"'
            matches = re.findall(pattern, ppd_content, re.MULTILINE | re.DOTALL)
            
        return [match.strip() for match in matches if match.strip()]
        
    def _generate_mode_combinations(self, color_modes, qualities, resolutions, media_types) -> List[Dict[str, Any]]:
        """Generate all valid mode combinations"""
        modes = []
        
        color_modes = color_modes or ['RGB', 'Gray']
        qualities = qualities or ['Normal']
        resolutions = resolutions or ['600dpi']
        media_types = media_types or ['Plain']
        
        mode_id = 1
        for color in color_modes:
            for quality in qualities:
                for resolution in resolutions:
                    for media in media_types:
                        mode = {
                            'id': mode_id,
                            'name': f"{color}_{quality}_{resolution}_{media}",
                            'color_mode': color,
                            'quality': quality,
                            'resolution': resolution,
                            'media_type': media,
                            'options': {
                                'ColorModel': color,
                                'Quality': quality,
                                'Resolution': resolution,
                                'MediaType': media
                            }
                        }
                        modes.append(mode)
                        mode_id += 1
                        
        return modes
        
    def _get_default_modes(self) -> List[Dict[str, Any]]:
        """Fallback default modes when PPD parsing fails"""
        return [
            {
                'id': 1,
                'name': 'Color_Normal_600dpi',
                'color_mode': 'RGB',
                'quality': 'Normal',
                'resolution': '600dpi',
                'media_type': 'Plain',
                'options': {'ColorModel': 'RGB', 'Quality': 'Normal'}
            },
            {
                'id': 2,
                'name': 'Grayscale_Normal_600dpi',
                'color_mode': 'Gray',
                'quality': 'Normal',
                'resolution': '600dpi',
                'media_type': 'Plain',
                'options': {'ColorModel': 'Gray', 'Quality': 'Normal'}
            }
        ]
        
    def _get_filter_chains_for_modes(self) -> Dict[str, List[str]]:
        """Get filter chains for each supported mode"""
        filter_chains = {}
        
        for mode in self.supported_modes:
            try:
                filters_pdf = get_filter_chain(self.printer_queue, 'application/pdf')
                filters_image = get_filter_chain(self.printer_queue, 'image/png')
                
                if filters_image and len(filters_image) > 0:
                    filters = filters_image
                elif filters_pdf and len(filters_pdf) > 0:
                    filters = filters_pdf
                else:
                    if mode.get('color_mode', '').lower() in ['gray', 'grayscale']:
                        filters = ['imagetoraster', 'rastertopwg']
                    else:
                        filters = ['imagetoraster', 'rastertopwg']
                
                try:
                    advanced_filters = get_filter_chain_advanced(
                        self.printer_queue, 'image/png'
                    )
                    if advanced_filters and len(advanced_filters) > 1:
                        filters = advanced_filters
                except Exception:
                    pass  # Fall back to basic filter chain
                
                if not filters or filters == ['gziptoany']:
                    filters = ['imagetoraster']
                
                filter_chains[mode['name']] = filters
                self.logger.info(f"Mode {mode['name']}: {filters}")
                
            except Exception as e:
                self.logger.warning(f"Failed to get filter chain for mode {mode['name']}: {e}")
                filter_chains[mode['name']] = ['imagetoraster']
                
        return filter_chains
        
    def generate_test_images(self) -> List[str]:
        """Generate comprehensive test images"""
        self.logger.info("Generating comprehensive test images...")
        
        try:
            self.image_generator.generate_comprehensive_test_suite()
            
            generator_output_dir = self.image_generator.output_dir
            self.test_images = []
            
            if os.path.exists(generator_output_dir):
                for file in os.listdir(generator_output_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        src_path = os.path.join(generator_output_dir, file)
                        dst_path = os.path.join(self.dirs['test_images'], file)
                        shutil.copy2(src_path, dst_path)
                        self.test_images.append(dst_path)
            
            self.logger.info(f"Generated {len(self.test_images)} test images")
            return self.test_images
            
        except Exception as e:
            self.logger.error(f"Failed to generate test images: {e}")
            raise
            
    def run_print_tests(self) -> Dict[str, Any]:
        """Run print tests for all modes and images"""
        self.logger.info("Starting print tests for all modes...")
        
        test_results = {}
        
        for mode in self.supported_modes:
            mode_name = mode['name']
            self.logger.info(f"Testing mode: {mode_name}")
            
            mode_results = {
                'mode_info': mode,
                'filter_chain': self.filter_chains.get(mode_name, []),
                'image_results': {}
            }
            
            for test_image in self.test_images:
                image_name = os.path.basename(test_image)
                self.logger.info(f"  Processing image: {image_name}")
                
                try:
                    processed_image = self._process_image_through_filters(
                        test_image, mode, mode_name
                    )
                    
                    if processed_image:
                        # Compare original vs processed
                        comparison_results = self._compare_images(
                            test_image, processed_image, mode_name, image_name
                        )
                        
                        mode_results['image_results'][image_name] = {
                            'original_image': test_image,
                            'processed_image': processed_image,
                            'comparison_results': comparison_results,
                            'status': 'success'
                        }
                    else:
                        mode_results['image_results'][image_name] = {
                            'status': 'failed',
                            'error': 'Processing failed'
                        }
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {image_name} in mode {mode_name}: {e}")
                    mode_results['image_results'][image_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    
            test_results[mode_name] = mode_results
            
        self.results = test_results
        return test_results
        
    def _process_image_through_filters(self, image_path: str, mode: Dict[str, Any], mode_name: str) -> Optional[str]:
        """Process image through the filter chain for given mode"""
        try:
            mode_output_dir = os.path.join(self.dirs['printed_outputs'], mode_name)
            os.makedirs(mode_output_dir, exist_ok=True)
            
            image_name = os.path.basename(image_path)
            output_path = os.path.join(mode_output_dir, f"processed_{image_name}")
            
            # Method 1: Try using CUPS directly
            if self._process_with_cups(image_path, output_path, mode):
                return output_path
                
            # Method 2: Try manual filter chain execution
            if self._process_with_manual_filters(image_path, output_path, mode, mode_name):
                return output_path
                
            # Method 3: Fallback to basic conversion
            if self._process_with_fallback(image_path, output_path, mode):
                return output_path
                
            return None
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return None
            
    def _process_with_cups(self, input_path: str, output_path: str, mode: Dict[str, Any]) -> bool:
        """Process image using CUPS cupsfilter command"""
        try:
            # Method 1: Try cupsfilter
            cmd = ['cupsfilter', '-p', self.printer_queue]
            
            # Add mode-specific options
            for option, value in mode.get('options', {}).items():
                cmd.extend(['-o', f'{option}={value}'])
            
            # Add input file
            cmd.append(input_path)
            
            # Run cupsfilter and capture output
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                with open(output_path, 'wb') as f:
                    f.write(result.stdout)
                return True
            else:
                self.logger.debug(f"cupsfilter failed: {result.stderr.decode()}")
            
            # Method 2: Try lp with file output simulation
            temp_dir = tempfile.mkdtemp()
            try:
                temp_input = os.path.join(temp_dir, os.path.basename(input_path))
                shutil.copy2(input_path, temp_input)
                
                # Use lp to "print" to a file
                result = subprocess.run([
                    'lp', '-d', self.printer_queue, '-o', 'raw', temp_input
                ], capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    # output in common CUPS output directories
                    output_dirs = [
                        # '/var/spool/cups-pdf',
                        # f'/home/{os.getenv("USER")}/Desktop',
                        f'/home/{os.getenv("USER")}/gsoc/openprinting/PDF',
                        './PDF'
                    ]
                    
                    import time
                    time.sleep(2)
                    
                    for output_dir in output_dirs:
                        if os.path.exists(output_dir):
                            files = sorted(glob.glob(f"{output_dir}/*"), key=os.path.getmtime)
                            if files:
                                latest_file = files[-1]
                                shutil.copy2(latest_file, output_path)
                                return True
                                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                    
        except Exception as e:
            self.logger.debug(f"CUPS processing failed: {e}")
        
        return False
        
    def _process_with_manual_filters(self, input_path: str, output_path: str, mode: Dict[str, Any], mode_name: str) -> bool:
        """Process using manual filter chain execution"""
        try:
            filter_chain = self.filter_chains.get(mode_name, [])
            
            if not filter_chain or filter_chain == ['default']:
                return False
                
            current_input = input_path
            
            for i, filter_name in enumerate(filter_chain):
                temp_output = f"{output_path}.temp{i}"
                
                if self._execute_single_filter(current_input, temp_output, filter_name, mode):
                    current_input = temp_output
                else:
                    for j in range(i):
                        temp_file = f"{output_path}.temp{j}"
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    return False
                    
            if os.path.exists(current_input) and current_input != input_path:
                shutil.move(current_input, output_path)
                
                for i in range(len(filter_chain) - 1):
                    temp_file = f"{output_path}.temp{i}"
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        
                return True
                
        except Exception as e:
            self.logger.debug(f"Manual filter processing failed: {e}")
            
        return False
        
    def _execute_single_filter(self, input_path: str, output_path: str, filter_name: str, mode: Dict[str, Any]) -> bool:
        """Execute a single filter in the chain"""
        try:
            filter_commands = {
                'pstoraster': ['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=ppmraw', f'-sOutputFile={output_path}', input_path],
                'rastertopdf': ['rastertopdf', '1', 'user', 'title', '1', 'options', input_path],
                'gstoraster': ['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=cups', f'-sOutputFile={output_path}', input_path],
                'imagetoraster': ['imagetoraster', '1', 'user', 'title', '1', 'options'],
                'pdftoraster': ['pdftoraster', '1', 'user', 'title', '1', 'options']
            }
            
            if filter_name in filter_commands:
                cmd = filter_commands[filter_name].copy()
                
                if filter_name in ['imagetoraster', 'pdftoraster']:
                    with open(input_path, 'rb') as input_file:
                        with open(output_path, 'wb') as output_file:
                            result = subprocess.run(cmd, stdin=input_file, stdout=output_file, 
                                                  stderr=subprocess.PIPE, timeout=30)
                else:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    
                return result.returncode == 0 and os.path.exists(output_path)
                
        except Exception as e:
            self.logger.debug(f"Filter {filter_name} execution failed: {e}")
            
        return False
        
    def _process_with_fallback(self, input_path: str, output_path: str, mode: Dict[str, Any]) -> bool:
        """Fallback processing using ImageMagick/Pillow"""
        try:
            from PIL import Image, ImageOps
            
            img = Image.open(input_path)
            
            color_mode = mode.get('color_mode', 'RGB')
            
            if color_mode.lower() in ['gray', 'grayscale', 'mono']:
                img = ImageOps.grayscale(img)
            elif color_mode.lower() in ['rgb', 'color']:
                img = img.convert('RGB')
                
            if 'resolution' in mode and 'dpi' in mode['resolution'].lower():
                try:
                    dpi = int(mode['resolution'].lower().replace('dpi', ''))
                    # Simulate resolution change
                    width, height = img.size
                    new_width = int(width * dpi / 300)  # Assuming 300 DPI baseline
                    new_height = int(height * dpi / 300)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except:
                    pass
                    
            img.save(output_path)
            return os.path.exists(output_path)
            
        except Exception as e:
            self.logger.debug(f"Fallback processing failed: {e}")
            
        return False
        
    def _compare_images(self, original_path: str, processed_path: str, mode_name: str, image_name: str) -> Dict[str, Any]:
        """Compare original and processed images using enhanced comparison"""
        try:
            comparison_dir = os.path.join(self.dirs['comparisons'], mode_name, 
                                        os.path.splitext(image_name)[0])
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Always try to convert the processed file to PNG since filter outputs may be in various formats
            final_processed_path = processed_path + '.converted.png'
            
            if self._convert_to_png(processed_path, final_processed_path):
                import cv2
                test_img = cv2.imread(final_processed_path)
                if test_img is not None:
                    comparator = ImageComparator(original_path, final_processed_path, comparison_dir)
                    results = comparator.run_all_comparisons()
                    return results
                else:
                    return {'error': 'Converted file is not a valid image'}
            else:
                import cv2
                test_img = cv2.imread(processed_path)
                if test_img is not None:
                    comparator = ImageComparator(original_path, processed_path, comparison_dir)
                    results = comparator.run_all_comparisons()
                    return results
                else:
                    return {'error': 'Could not load or convert processed file to readable format'}
            
        except Exception as e:
            self.logger.error(f"Image comparison failed: {e}")
            return {'error': str(e)}
    
    def _convert_to_png(self, input_path: str, output_path: str) -> bool:
        """Convert various formats to PNG"""
        try:
            # Method 1: Try with Ghostscript (for PS/PDF formats)
            result = subprocess.run([
                'gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                f'-sOutputFile={output_path}', input_path
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            
            # Method 2: Try with ImageMagick
            result = subprocess.run([
                'convert', input_path, output_path
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            
            # Method 3: Try with pdftoppm for PDF files
            result = subprocess.run([
                'pdftoppm', '-png', '-r', '300', input_path, 
                output_path.replace('.png', '')
            ], capture_output=True, timeout=30)
            
            # pdftoppm creates files with suffix, check for them
            possible_files = [
                output_path.replace('.png', '-1.png'),
                output_path.replace('.png', '-001.png')
            ]
            
            for pfile in possible_files:
                if os.path.exists(pfile):
                    os.rename(pfile, output_path)
                    return True
            
            # Method 4: Try with PIL (for basic formats)
            from PIL import Image
            img = Image.open(input_path)
            img.save(output_path, 'PNG')
            return True
            
        except Exception as e:
            self.logger.debug(f"Conversion to PNG failed: {e}")
            return False
            
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        self.logger.info("Generating comprehensive report...")
        
        try:
            report_path = os.path.join(self.dirs['reports'], 'comprehensive_report.html')
            
            # Generate HTML report
            html_content = self._generate_html_report()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Generate JSON summary
            json_path = os.path.join(self.dirs['reports'], 'results_summary.json')
            summary = self._generate_json_summary()
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
                
            self.logger.info(f"Reports generated: {report_path}, {json_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
            
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Print Quality Test Report - {self.printer_queue}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .mode-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                .image-result {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
                .score {{ font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Print Quality Test Report</h1>
                <p><strong>Printer Queue:</strong> {self.printer_queue}</p>
                <p><strong>Test Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Modes Tested:</strong> {len(self.supported_modes)}</p>
                <p><strong>Total Images Tested:</strong> {len(self.test_images)}</p>
            </div>
        """
        
        for mode_name, mode_results in self.results.items():
            html += self._generate_mode_section_html(mode_name, mode_results)
            
        html += """
        </body>
        </html>
        """
        
        return html
        
    def _generate_mode_section_html(self, mode_name: str, mode_results: Dict[str, Any]) -> str:
        """Generate HTML section for a specific mode"""
        html = f"""
        <div class="mode-section">
            <h2>Mode: {mode_name}</h2>
            <p><strong>Filter Chain:</strong> {' → '.join(mode_results['filter_chain'])}</p>
            
            <h3>Image Test Results</h3>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Status</th>
                    <th>Overall Quality</th>
                    <th>SSIM</th>
                    <th>Edge Preservation</th>
                    <th>Color Preservation</th>
                </tr>
        """
        
        for image_name, image_result in mode_results['image_results'].items():
            if image_result['status'] == 'success':
                comparison = image_result['comparison_results']
                overall_quality = comparison.get('overall_quality', 0)
                ssim_score = comparison.get('ssim', 0)
                edge_score = comparison.get('edge_similarity', 0)
                color_score = comparison.get('color_preservation_score', 0)
                
                quality_class = self._get_score_class(overall_quality)
                
                html += f"""
                <tr>
                    <td>{image_name}</td>
                    <td class="good">Success</td>
                    <td class="{quality_class}">{overall_quality:.3f}</td>
                    <td>{ssim_score:.3f}</td>
                    <td>{edge_score:.3f}</td>
                    <td>{color_score:.3f}</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{image_name}</td>
                    <td class="error">{image_result['status']}</td>
                    <td colspan="4">N/A</td>
                </tr>
                """
                
        html += """
            </table>
        </div>
        """
        
        return html
        
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score"""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "warning"
        else:
            return "error"
            
    def _generate_json_summary(self) -> Dict[str, Any]:
        """Generate JSON summary of results"""
        summary = {
            'test_info': {
                'printer_queue': self.printer_queue,
                'test_date': datetime.now().isoformat(),
                'total_modes': len(self.supported_modes),
                'total_images': len(self.test_images)
            },
            'printer_capabilities': {
                'supported_modes': self.supported_modes,
                'filter_chains': self.filter_chains
            },
            'overall_statistics': self._calculate_overall_statistics(),
            'mode_results': self.results
        }
        
        return summary
        
    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall test statistics"""
        total_tests = 0
        successful_tests = 0
        total_quality = 0
        quality_scores = []
        
        for mode_results in self.results.values():
            for image_result in mode_results['image_results'].values():
                total_tests += 1
                if image_result['status'] == 'success':
                    successful_tests += 1
                    if 'comparison_results' in image_result:
                        quality = image_result['comparison_results'].get('overall_quality', 0)
                        total_quality += quality
                        quality_scores.append(quality)
                        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        avg_quality = total_quality / successful_tests if successful_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'average_quality_score': avg_quality,
            'min_quality_score': min(quality_scores) if quality_scores else 0,
            'max_quality_score': max(quality_scores) if quality_scores else 0
        }
        
    def run_complete_pipeline(self) -> str:
        """Run the complete testing pipeline"""
        self.logger.info("Starting complete print quality testing pipeline...")
        
        try:
            # Step 1: Analyze printer capabilities
            self.logger.info("Step 1: Analyzing printer capabilities...")
            self.analyze_printer_capabilities()
            
            # Step 2: Generate test images
            self.logger.info("Step 2: Generating test images...")
            self.generate_test_images()
            
            # Step 3: Run print tests for all modes
            self.logger.info("Step 3: Running print tests...")
            self.run_print_tests()
            
            # Step 4: Generate comprehensive report
            self.logger.info("Step 4: Generating reports...")
            report_path = self.generate_comprehensive_report()
            
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results available in: {self.output_dir}")
            self.logger.info(f"Main report: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Print Quality Testing Pipeline')
    parser.add_argument('printer_queue', help='Printer queue name')
    parser.add_argument('--output-dir', help='Output directory (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        pipeline = PrintQualityTestPipeline(args.printer_queue, args.output_dir)
        report_path = pipeline.run_complete_pipeline()
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results directory: {pipeline.output_dir}")
        print(f"Main report: {report_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()