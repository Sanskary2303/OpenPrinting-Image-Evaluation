#!/usr/bin/env python3
"""
Filter Chain Testing Pipeline

This pipeline tests the correctness of CUPS filter chains by:
1. Processing documents through the complete filter chain
2. Capturing the final output (what would be sent to printer)
3. Comparing filter chain output with original expectations
4. Avoiding physical printing to isolate filter chain issues

Compare document passed all filters (in the format that will be sent to the printer) with our expectations
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import re
import argparse
import logging
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from printer_info import (
    get_available_printers, 
    get_printer_model, 
    get_filter_chain, 
    get_printer_ppd_info,
)
from generate_test_images import ComprehensiveTestImageGenerator
from enhanced_comparison import ImageComparator

class FilterChainTestPipeline:
    """
    Pipeline to test CUPS filter chain correctness without physical printing
    
    Tests: Original Document → CUPS Filters → Final Output → Analysis
    (Stops before actual printer to isolate filter chain testing)
    """
    
    def __init__(self, printer_queue: str, output_dir: Optional[str] = None):
        self.printer_queue = printer_queue
        self.output_dir = output_dir or f"filter_test_results_{printer_queue}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.setup_logging()
        
        self.image_generator = ComprehensiveTestImageGenerator(self.output_dir)
        
        # Pipeline state
        self.printer_info = None
        self.supported_modes = []
        self.filter_chains = {}
        self.test_images = []
        self.results = {}
        
        self.create_output_structure()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.output_dir, 'filter_chain_test.log')
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
            'filter_outputs': os.path.join(self.output_dir, 'filter_outputs'),
            'intermediate_steps': os.path.join(self.output_dir, 'intermediate_steps'),
            'analysis_results': os.path.join(self.output_dir, 'analysis_results'),
            'comparisons': os.path.join(self.output_dir, 'comparisons'),
            'reports': os.path.join(self.output_dir, 'reports'),
            'logs': os.path.join(self.output_dir, 'logs')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
    def analyze_printer_capabilities(self) -> Dict[str, Any]:
        """Analyze printer capabilities and filter chains"""
        self.logger.info(f"Analyzing filter chains for printer: {self.printer_queue}")
        
        try:
            # Check if printer exist
            available_printers = get_available_printers()
            if self.printer_queue not in available_printers:
                raise ValueError(f"Printer queue '{self.printer_queue}' not found. Available: {available_printers}")
            
            #Get printer information
            self.printer_info = {
                'name': self.printer_queue,
                'model': get_printer_model(self.printer_queue),
                'ppd_info': get_printer_ppd_info(self.printer_queue),
            }
            
            #Extract supported modes and their filter chains
            self.supported_modes = self._extract_supported_modes()
            self.filter_chains = self._get_detailed_filter_chains()
            
            self.logger.info(f"Found {len(self.supported_modes)} supported modes")
            self.logger.info(f"Filter chains: {list(self.filter_chains.keys())}")
            
            return {
                'printer_info': self.printer_info,
                'supported_modes': self.supported_modes,
                'filter_chains': self.filter_chains
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze printer capabilities: {e}")
            raise
            
    def _extract_supported_modes(self) -> List[Dict[str, Any]]:
        """Extract supported modes from printer configuration"""
        # mode extraction for filter chain testing
        modes = [
            {
                'id': 1,
                'name': 'Default_Color',
                'color_mode': 'color',
                'options': {}
            },
            {
                'id': 2,
                'name': 'Default_Grayscale',
                'color_mode': 'grayscale',
                'options': {'ColorModel': 'Gray'}
            }
        ]
        
        #Add DPI variations
        try:
            ppd_info = self.printer_info.get('ppd_info', {})
            if 'resolutions' in ppd_info:
                additional_modes = []
                for resolution in ['300dpi', '600dpi']:
                    additional_modes.extend([
                        {
                            'id': len(modes) + len(additional_modes) + 1,
                            'name': f'Color_{resolution}',
                            'color_mode': 'color',
                            'options': {'Resolution': resolution}
                        },
                        {
                            'id': len(modes) + len(additional_modes) + 2,
                            'name': f'Grayscale_{resolution}',
                            'color_mode': 'grayscale', 
                            'options': {'ColorModel': 'Gray', 'Resolution': resolution}
                        }
                    ])
                modes.extend(additional_modes)
        except:
            pass  #default modes if PPD parsing fails
            
        return modes
        
    def _get_detailed_filter_chains(self) -> Dict[str, List[str]]:
        """Get detailed filter chains for each mode"""
        filter_chains = {}
        
        for mode in self.supported_modes:
            mode_name = mode['name']
            
            try:
                # Get filter chain for different input types
                chains = {
                    'image/png': self._discover_filter_chain('image/png', mode),
                    'application/pdf': self._discover_filter_chain('application/pdf', mode),
                    'text/plain': self._discover_filter_chain('text/plain', mode)
                }
                
                filter_chains[mode_name] = chains
                self.logger.info(f"Mode {mode_name} filter chains: {chains}")
                
            except Exception as e:
                self.logger.warning(f"Failed to get filter chains for mode {mode_name}: {e}")
                filter_chains[mode_name] = {'default': ['direct']}
                
        return filter_chains
        
    def _discover_filter_chain(self, input_mime: str, mode: Dict[str, Any]) -> List[str]:
        """Discover the filter chain CUPS would use for specific input type"""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                # Use cupsfilter with dry-run to discover filter chain
                cmd = ['cupsfilter', '-p', self.printer_queue, '-m', input_mime]
                
                # Add mode-specific options
                for option, value in mode.get('options', {}).items():
                    cmd.extend(['-o', f'{option}={value}'])
                    
                # Add debug flag to get filter chain info
                cmd.extend(['-v', temp_path])
                
                env = os.environ.copy()
                env['CUPS_DEBUG_LEVEL'] = '2'
                
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      env=env, timeout=30)
                
                filters = self._parse_filter_chain_from_output(result.stderr)
                
                return filters if filters else ['direct']
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.debug(f"Filter chain discovery failed for {input_mime}: {e}")
            try:
                return get_filter_chain(self.printer_queue, input_mime)
            except:
                return ['direct']
                
    def _parse_filter_chain_from_output(self, debug_output: str) -> List[str]:
        """Parse CUPS debug output to extract filter chain"""
        filters = []
        
        try:
            # filter execution patterns in CUPS debug output
            filter_patterns = [
                r'exec.*?/([^/\s]+filter[^/\s]*)\s',  # Standard filters
                r'Starting "([^"]+)"',  # Filter start messages
                r'Filter "([^"]+)"',    # Filter execution messages
            ]
            
            for pattern in filter_patterns:
                matches = re.findall(pattern, debug_output, re.IGNORECASE)
                filters.extend(matches)
            
            seen = set()
            unique_filters = []
            for f in filters:
                if f not in seen and 'filter' in f.lower():
                    seen.add(f)
                    unique_filters.append(f)
                    
            return unique_filters
            
        except Exception as e:
            self.logger.debug(f"Failed to parse filter chain: {e}")
            return []
            
    def generate_test_images(self) -> List[str]:
        """Generate test images for filter chain testing"""
        self.logger.info("Generating test images for filter chain testing...")
        
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
            
    def run_filter_chain_tests(self) -> Dict[str, Any]:
        """Run filter chain tests for all modes and images"""
        self.logger.info("Starting filter chain correctness tests...")
        
        test_results = {}
        
        for mode in self.supported_modes:
            mode_name = mode['name']
            self.logger.info(f"Testing filter chains for mode: {mode_name}")
            
            mode_results = {
                'mode_info': mode,
                'filter_chains': self.filter_chains.get(mode_name, {}),
                'image_results': {}
            }
            
            for test_image in self.test_images[:5]:  # Limit to first 5 for testing
                image_name = os.path.basename(test_image)
                self.logger.info(f"  Processing image: {image_name}")
                
                try:
                    # Process through filter chains
                    filter_outputs = self._process_through_filter_chains(
                        test_image, mode, mode_name
                    )
                    
                    if filter_outputs:
                        # Compare each filter chain output with original
                        comparison_results = {}
                        
                        for input_type, output_path in filter_outputs.items():
                            if output_path:
                                comparison_results[input_type] = self._compare_filter_output(
                                    test_image, output_path, mode_name, image_name, input_type
                                )
                        
                        mode_results['image_results'][image_name] = {
                            'filter_outputs': filter_outputs,
                            'comparison_results': comparison_results,
                            'status': 'success'
                        }
                    else:
                        mode_results['image_results'][image_name] = {
                            'status': 'failed',
                            'error': 'Filter chain processing failed'
                        }
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {image_name}: {e}")
                    mode_results['image_results'][image_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            test_results[mode_name] = mode_results
            
        self.results = test_results
        return test_results
        
    def _process_through_filter_chains(self, image_path: str, mode: Dict[str, Any], mode_name: str) -> Dict[str, Optional[str]]:
        """Process image through all filter chains for this mode"""
        
        filter_outputs = {}
        mode_filter_chains = self.filter_chains.get(mode_name, {})
        
        for input_type, filter_chain in mode_filter_chains.items():
            try:
                self.logger.debug(f"Processing {os.path.basename(image_path)} through {input_type} filter chain: {filter_chain}")
                
                output_dir = os.path.join(self.dirs['filter_outputs'], mode_name)
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_{input_type.replace('/', '_')}_output")
                
                final_output = self._execute_filter_chain(image_path, output_path, filter_chain, mode, input_type)
                
                filter_outputs[input_type] = final_output
                
            except Exception as e:
                self.logger.warning(f"Filter chain processing failed for {input_type}: {e}")
                filter_outputs[input_type] = None
                
        return filter_outputs
        
    def _execute_filter_chain(self, input_path: str, output_base: str, filter_chain: List[str], 
                            mode: Dict[str, Any], input_type: str) -> Optional[str]:
        """Execute the complete filter chain and return final output path"""
        
        try:
            if not filter_chain or filter_chain == ['direct']:
                # No filtering needed, just copy
                final_output = f"{output_base}_direct.png"
                shutil.copy2(input_path, final_output)
                return final_output
                
            # Execute filter chain step by step
            current_input = input_path
            intermediate_dir = os.path.join(self.dirs['intermediate_steps'], 
                                          os.path.splitext(os.path.basename(input_path))[0])
            os.makedirs(intermediate_dir, exist_ok=True)
            
            for i, filter_name in enumerate(filter_chain):
                step_output = os.path.join(intermediate_dir, f"step_{i}_{filter_name}")
                
                if self._execute_single_filter(current_input, step_output, filter_name, mode):
                    current_input = step_output
                    self.logger.debug(f"Filter step {i} ({filter_name}) completed")
                else:
                    self.logger.warning(f"Filter step {i} ({filter_name}) failed")
                    return None
                    
            # Convert final output to analyzable format
            final_output = f"{output_base}_final.png"
            if self._convert_for_analysis(current_input, final_output):
                return final_output
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Filter chain execution failed: {e}")
            return None
            
    def _execute_single_filter(self, input_path: str, output_path: str, filter_name: str, mode: Dict[str, Any]) -> bool:
        """Execute a single filter step"""
        
        try:
            if filter_name in ['imagetoraster', 'imagetopdf']:
                # Image conversion filters
                cmd = [filter_name, '1', 'user', 'title', '1']

                options = []
                for option, value in mode.get('options', {}).items():
                    options.append(f'{option}={value}')
                cmd.append(' '.join(options) if options else '')
                
                # Execute filter (reads stdin, writes stdout)
                with open(input_path, 'rb') as input_file:
                    with open(output_path, 'wb') as output_file:
                        result = subprocess.run(cmd, stdin=input_file, stdout=output_file, 
                                              stderr=subprocess.PIPE, timeout=60)
                        return result.returncode == 0
                        
            elif filter_name in ['pdftoraster', 'pstoraster']:
                # PDF/PS to raster filters  
                cmd = [filter_name, '1', 'user', 'title', '1']
                
                options = []
                for option, value in mode.get('options', {}).items():
                    options.append(f'{option}={value}')
                cmd.append(' '.join(options) if options else '')
                
                with open(input_path, 'rb') as input_file:
                    with open(output_path, 'wb') as output_file:
                        result = subprocess.run(cmd, stdin=input_file, stdout=output_file, 
                                              stderr=subprocess.PIPE, timeout=60)
                        return result.returncode == 0
                        
            elif filter_name == 'direct':
                shutil.copy2(input_path, output_path)
                return True
                
            else:
                self.logger.warning(f"Unknown filter: {filter_name}, using direct copy")
                shutil.copy2(input_path, output_path)
                return True
                
        except Exception as e:
            self.logger.error(f"Filter {filter_name} execution failed: {e}")
            return False
            
    def _convert_for_analysis(self, filter_output: str, analysis_output: str) -> bool:
        """Convert filter output to PNG for analysis"""
        
        try:
            file_type = self._detect_file_type(filter_output)
            
            if file_type == 'png':
                shutil.copy2(filter_output, analysis_output)
                return True
                
            elif file_type == 'pdf':
                # Convert PDF to PNG
                cmd = ['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                       f'-sOutputFile={analysis_output}', filter_output]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                return result.returncode == 0 and os.path.exists(analysis_output)
                
            elif file_type == 'postscript':
                # Convert PS to PNG
                cmd = ['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r300',
                       f'-sOutputFile={analysis_output}', filter_output]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                return result.returncode == 0 and os.path.exists(analysis_output)
                
            else:
                # Try ImageMagick as fallback
                cmd = ['convert', filter_output, analysis_output]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                return result.returncode == 0 and os.path.exists(analysis_output)
                
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return False
            
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from header"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            if header.startswith(b'%PDF'):
                return 'pdf'
            elif header.startswith(b'%!PS'):
                return 'postscript'
            elif header.startswith(b'\x89PNG'):
                return 'png'
            elif header.startswith(b'\xff\xd8\xff'):
                return 'jpeg'
            else:
                return 'unknown'
                
        except:
            return 'unknown'
            
    def _compare_filter_output(self, original_path: str, filtered_path: str, mode_name: str, 
                             image_name: str, input_type: str) -> Dict[str, Any]:
        """Compare original image with filter chain output"""
        
        try:
            # comparison directory
            comparison_dir = os.path.join(self.dirs['comparisons'], mode_name, 
                                        os.path.splitext(image_name)[0], input_type.replace('/', '_'))
            os.makedirs(comparison_dir, exist_ok=True)
            
            #enhanced comparison
            comparator = ImageComparator(original_path, filtered_path, comparison_dir)
            results = comparator.run_all_comparisons()
            
            # Add filter-specific analysis
            results['filter_chain_effects'] = self._analyze_filter_effects(
                original_path, filtered_path, mode_name
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Filter output comparison failed: {e}")
            return {'error': str(e)}
            
    def _analyze_filter_effects(self, original_path: str, filtered_path: str, mode_name: str) -> Dict[str, Any]:
        """Analyze effects introduced by filter chain"""
        
        effects = {}
        
        try:
            import cv2
            import numpy as np
            
            original = cv2.imread(original_path)
            filtered = cv2.imread(filtered_path)
            
            if original is None or filtered is None:
                return {'error': 'Could not load images for filter effect analysis'}
            
            # Analyze specific filter effects
            effects['size_change'] = {
                'original_size': original.shape[:2],
                'filtered_size': filtered.shape[:2],
                'size_preserved': original.shape[:2] == filtered.shape[:2]
            }
            
            # Color analysis
            if len(original.shape) == 3 and len(filtered.shape) == 3:
                orig_mean_color = np.mean(original, axis=(0, 1))
                filt_mean_color = np.mean(filtered, axis=(0, 1))
                effects['color_shift'] = {
                    'original_mean': orig_mean_color.tolist(),
                    'filtered_mean': filt_mean_color.tolist(),
                    'color_preserved': np.allclose(orig_mean_color, filt_mean_color, rtol=0.1)
                }
            
            # Sharpness analysis
            def calculate_sharpness(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                return cv2.Laplacian(gray, cv2.CV_64F).var()
            
            orig_sharpness = calculate_sharpness(original)
            filt_sharpness = calculate_sharpness(filtered)
            
            effects['sharpness_change'] = {
                'original_sharpness': float(orig_sharpness),
                'filtered_sharpness': float(filt_sharpness),
                'sharpness_ratio': float(filt_sharpness / orig_sharpness) if orig_sharpness > 0 else 0,
                'sharpness_preserved': abs(1.0 - (filt_sharpness / orig_sharpness)) < 0.2 if orig_sharpness > 0 else True
            }
            
            return effects
            
        except Exception as e:
            return {'error': f'Filter effect analysis failed: {e}'}
            
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive filter chain test report"""
        self.logger.info("Generating filter chain test report...")
        
        try:
            report_path = os.path.join(self.dirs['reports'], 'filter_chain_test_report.html')
            
            # HTML report
            html_content = self._generate_filter_chain_html_report()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # JSON summary
            json_path = os.path.join(self.dirs['reports'], 'filter_chain_results.json')
            summary = self._generate_json_summary()
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
                
            self.logger.info(f"Reports generated: {report_path}, {json_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
            
    def _generate_filter_chain_html_report(self) -> str:
        """Generate HTML report for filter chain testing"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Filter Chain Test Report - {printer_queue}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .mode-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                .filter-chain {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CUPS Filter Chain Correctness Test Report</h1>
                <p><strong>Printer:</strong> {printer_queue}</p>
                <p><strong>Test Date:</strong> {test_date}</p>
                <p><strong>Purpose:</strong> Testing filter chain correctness without physical printing</p>
            </div>
            
            <h2>Test Overview</h2>
            <ul>
                <li>Total Modes Tested: {total_modes}</li>
                <li>Total Images Processed: {total_images}</li>
                <li>Filter Chains Tested: {total_filter_chains}</li>
            </ul>
            
            {mode_sections}
            
            <h2>Summary</h2>
            <p>This report shows the correctness of CUPS filter chains by comparing:</p>
            <ul>
                <li><strong>Original Test Images</strong> with</li>
                <li><strong>Filter Chain Processed Output</strong> (what would be sent to printer)</li>
            </ul>
            <p>Issues found here indicate filter chain problems, not printer hardware issues.</p>
        </body>
        </html>
        """
        
        # mode sections
        mode_sections = ""
        for mode_name, mode_results in self.results.items():
            mode_sections += self._generate_mode_section_html(mode_name, mode_results)
            
        return html_template.format(
            printer_queue=self.printer_queue,
            test_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_modes=len(self.results),
            total_images=sum(len(mode_results['image_results']) for mode_results in self.results.values()),
            total_filter_chains=sum(len(mode_results['filter_chains']) for mode_results in self.results.values()),
            mode_sections=mode_sections
        )
        
    def _generate_mode_section_html(self, mode_name: str, mode_results: Dict[str, Any]) -> str:
        """Generate HTML section for a specific mode"""
        
        filter_chains_html = ""
        for input_type, chain in mode_results['filter_chains'].items():
            filter_chains_html += f"<div class='filter-chain'><strong>{input_type}:</strong> {' → '.join(chain)}</div>"
            
        image_results_html = "<table><tr><th>Image</th><th>Status</th><th>Quality Scores</th></tr>"
        
        for image_name, result in mode_results['image_results'].items():
            status_class = result['status']
            if result['status'] == 'success':
                # quality scores
                quality_info = ""
                comparison_results = result.get('comparison_results', {})
                for input_type, comp_result in comparison_results.items():
                    if isinstance(comp_result, dict) and 'overall_quality' in comp_result:
                        quality_info += f"{input_type}: {comp_result['overall_quality']:.3f}<br>"
            else:
                quality_info = result.get('error', 'Unknown error')
                
            image_results_html += f"""
            <tr>
                <td>{image_name}</td>
                <td class="{status_class}">{result['status']}</td>
                <td>{quality_info}</td>
            </tr>
            """
            
        image_results_html += "</table>"
        
        return f"""
        <div class="mode-section">
            <h3>Mode: {mode_name}</h3>
            <h4>Filter Chains:</h4>
            {filter_chains_html}
            <h4>Test Results:</h4>
            {image_results_html}
        </div>
        """
        
    def _generate_json_summary(self) -> Dict[str, Any]:
        """Generate JSON summary of filter chain test results"""
        
        return {
            'test_info': {
                'printer_queue': self.printer_queue,
                'test_type': 'filter_chain_correctness',
                'test_date': datetime.now().isoformat(),
                'purpose': 'Testing CUPS filter chain correctness without physical printing'
            },
            'results': self.results,
            'statistics': self._calculate_statistics()
        }
        
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics"""
        
        total_tests = 0
        successful_tests = 0
        filter_chain_stats = {}
        
        for mode_results in self.results.values():
            for image_result in mode_results['image_results'].values():
                total_tests += 1
                if image_result['status'] == 'success':
                    successful_tests += 1
                    
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_modes_tested': len(self.results),
            'total_filter_chains': sum(len(mode_results['filter_chains']) 
                                     for mode_results in self.results.values())
        }
        
    def run_complete_pipeline(self) -> str:
        """Run the complete filter chain testing pipeline"""
        self.logger.info("Starting complete filter chain correctness testing pipeline...")
        
        try:
            # Step 1: Analyze printer and filter chains
            self.logger.info("Step 1: Analyzing printer capabilities and filter chains...")
            self.analyze_printer_capabilities()
            
            # Step 2: Generate test images
            self.logger.info("Step 2: Generating test images...")
            self.generate_test_images()
            
            # Step 3: Run filter chain tests
            self.logger.info("Step 3: Running filter chain correctness tests...")
            self.run_filter_chain_tests()
            
            # Step 4: Generate comprehensive report
            self.logger.info("Step 4: Generating reports...")
            report_path = self.generate_comprehensive_report()
            
            self.logger.info("Filter chain testing pipeline completed successfully!")
            self.logger.info(f"Results available in: {self.output_dir}")
            self.logger.info(f"Main report: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Filter chain testing pipeline failed: {e}")
            raise

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='CUPS Filter Chain Correctness Testing Pipeline')
    parser.add_argument('printer_queue', help='Printer queue name')
    parser.add_argument('--output-dir', help='Output directory (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        print("="*80)
        print("CUPS FILTER CHAIN CORRECTNESS TESTING")
        print("="*80)
        print("Purpose: Test filter chain correctness by comparing:")
        print("  • Original test images")
        print("  • Filter chain processed output (what would be sent to printer)")
        print("Benefits: Isolates filter chain issues from printer hardware problems")
        print("="*80)
        
        # Create and run pipeline
        pipeline = FilterChainTestPipeline(args.printer_queue, args.output_dir)
        report_path = pipeline.run_complete_pipeline()
        
        print(f"\n{'='*60}")
        print("FILTER CHAIN TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results directory: {pipeline.output_dir}")
        print(f"Main report: {report_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Filter chain testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
