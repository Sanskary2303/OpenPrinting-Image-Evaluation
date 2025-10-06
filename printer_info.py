#!/usr/bin/env python3

import os
import subprocess
import re
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple, Union

def get_cups_filter_dir() -> str:
    """Attempts to find the CUPS filter directory using cups-config."""
    try:
        # cups-config --serverbin usually points to the directory containing filters, backend, etc.
        serverbin = subprocess.check_output(['cups-config', '--serverbin'], text=True).strip()
        filter_dir = os.path.join(serverbin, 'filter')
        if os.path.isdir(filter_dir):
            print(f"Detected CUPS filter directory: {filter_dir}")
            return filter_dir
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: 'cups-config' command failed or not found ({e}). Falling back to default.")
        
    default_dir = "/usr/lib/cups/filter/"
    print(f"Using default CUPS filter directory: {default_dir}")
    return default_dir

CUPS_FILTER_DIR: str = get_cups_filter_dir()

def get_printer_info() -> Dict[str, Dict[str, str]]:
    """Get information about available printers including names and models."""
    try:
        printer_names_output = subprocess.check_output(['lpstat', '-p'], text=True)
        
        printer_details_output = subprocess.check_output(['lpstat', '-v'], text=True)
        
        printers = {}
        
        for line in printer_names_output.splitlines():
            if line.startswith("printer "):
                parts = line.split()
                if len(parts) >= 2:
                    printer_name = parts[1]
                    printers[printer_name] = {"status": "enabled" if "enabled" in line else "disabled"}
        
        for line in printer_details_output.splitlines():
            if line.startswith("device for "):
                parts = line.split(":", 1)
                if len(parts) >= 2:
                    printer_name = parts[0].replace("device for ", "").strip()
                    connection = parts[1].strip()
                    if printer_name in printers:
                        printers[printer_name]["connection"] = connection
        
        return printers
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving printer information: {e}")
        return {}
    
def get_printer_details_with_pycups() -> Dict[str, Dict[str, str]]:
    """Get detailed printer information using pycups library."""
    try:
        import cups
        conn = cups.Connection()
        printers = conn.getPrinters()
        
        printer_info = {}
        for printer_name, attributes in printers.items():
            model = attributes.get('printer-make-and-model', 'Unknown')
            printer_info[printer_name] = {
                "model": model,
                "state": attributes.get('printer-state', 'Unknown'),
                "state_message": attributes.get('printer-state-message', ''),
                "location": attributes.get('printer-location', ''),
                "info": attributes.get('printer-info', '')
            }
        return printer_info
    except ImportError:
        print("pycups library not installed. Try: pip install pycups")
        return {}
    except Exception as e:
        print(f"Error retrieving printer information: {e}")
        return {}
    
def get_printer_ppd_info(printer_name: str) -> Dict[str, Union[str, List[str]]]:
    """Get detailed information from a printer's PPD using pycups."""
    try:
        import cups
        conn = cups.Connection()
        
        ppd_filename = conn.getPPD(printer_name)
        if not ppd_filename:
            print(f"Could not retrieve PPD for printer {printer_name}")
            return {}
            
        filter_info = {}
        
        with open(ppd_filename, 'r') as f:
            content = f.read()
            
        os.unlink(ppd_filename)
        
        filter_matches = re.findall(r'\*cupsFilter2?:\s*"([^"]+)"', content)
        if filter_matches:
            filter_info["filters"] = filter_matches
        
        model_match = re.search(r'\*ModelName:\s*"([^"]+)"', content)
        if model_match:
            filter_info["model"] = model_match.group(1)
            
        return filter_info
        
    except ImportError:
        print("pycups library not installed. Cannot retrieve PPD information.")
        return {}
    except Exception as e:
        print(f"Error retrieving PPD information with pycups: {e}")
        return {}

def get_available_printers() -> List[str]:
    """Get a list of available printer names."""
    try:
        output = subprocess.check_output(['lpstat', '-p'], text=True)
        printers = []
        for line in output.splitlines():
            if line.startswith("printer "):
                printers.append(line.split()[1])
        return printers
    except subprocess.CalledProcessError:
        return []

def get_printer_model(printer_name: str) -> str:
    """Get the model information for a specific printer using pycups."""
    try:
        import cups
        conn = cups.Connection()
        
        printers = conn.getPrinters()
        if printer_name in printers:
            model = printers[printer_name].get('printer-make-and-model', 'Unknown')
            return model
        
        try:
            attributes = conn.getPrinterAttributes(printer_name)
            model = attributes.get('printer-make-and-model', 'Unknown')
            return model
        except cups.IPPError:
            return "Printer not found"
            
    except ImportError:
        print("Warning: pycups library not installed. Falling back to lpoptions...")
        try:
            output = subprocess.check_output(['lpoptions', '-p', printer_name, '-l'], text=True)
            for line in output.splitlines():
                if "make-and-model" in line.lower():
                    return line.split(':')[1].strip()
        except Exception as e:
            print(f"Error with lpoptions fallback: {e}")
            
    except Exception as e:
        print(f"Error getting printer model with pycups: {e}")
        
    return "Unknown model"

def get_filter_chain(printer_name: str, input_mime_type: str = 'application/pdf') -> List[str]:
    """Get the filter chain CUPS would likely use for a printer and input type."""
    try:
        # Use cupsfilter --list-filters to show the filter chain
        # provide a dummy file path, as it's needed syntactically but not for listing filters.
        # Create a dummy empty file instead.
        with tempfile.NamedTemporaryFile() as dummy_file:
            command = [
                'cupsfilter',
                '--list-filters',
                '-p', printer_name,
                '-i', input_mime_type,  # -i for INPUT MIME type
                dummy_file.name         # Path to a dummy input file
            ]
            print(f"Running command: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout if result.stdout else result.stderr

            filters = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith("DEBUG:")]
            if not filters and result.returncode == 0:
                 return ["(No specific filters needed or listed)"]
            return filters

    except subprocess.CalledProcessError as e:
        print(f"Error getting filter chain for {printer_name} with type {input_mime_type}: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        ppd_info = get_printer_ppd_info(printer_name)
        if "filters" in ppd_info:
             return ["Filters from PPD (cupsfilter failed):"] + ppd_info["filters"]
        return ["Error: Could not determine filter chain."]
    except FileNotFoundError:
        print("Error: 'cupsfilter' command not found. Is CUPS installed correctly?")
        return ["Error: 'cupsfilter' not found."]
    except Exception as e:
        print(f"Unexpected error in get_filter_chain: {e}")
        return ["Error: Unexpected error."]

def run_cups_filter_chain(printer_name: str, input_file_path: str, output_file_path: str,
                          target_mime_type: str = 'application/vnd.cups-raster',
                          input_mime_type: Optional[str] = None, custom_filters: Optional[List[str]] = None,
                          filter_options: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Runs the CUPS filter chain for a specific printer and input file,
    or a custom filter chain using direct execution, saving the output.

    Parameters:
    - printer_name: Target printer (used if custom_filters is None).
    - input_file_path: Path to the input file.
    - output_file_path: Path to save the filtered output.
    - target_mime_type: Target output MIME type (used if custom_filters is None).
    - input_mime_type: Input MIME type (ignored if custom_filters is None).
    - custom_filters: List of filter executables to run in sequence (e.g., ['pdftopdf', 'pdftoraster']).
    - filter_options: Dict mapping filter_name (str) to options string (str)
                      for direct execution mode. Example: {'pdftoraster': 'ColorModel=RGB Resolution=300dpi'}
    """
    if custom_filters:
        # --- Direct Filter Execution Logic ---
        print(f"Running custom filter chain via direct execution: {' -> '.join(custom_filters)}")
        current_input_path = input_file_path
        temp_files = []
        filter_options = filter_options or {}

        try:
            for i, filter_name in enumerate(custom_filters):
                filter_executable = os.path.join(CUPS_FILTER_DIR, filter_name)
                if not os.path.exists(filter_executable) or not os.access(filter_executable, os.X_OK):
                    raise FileNotFoundError(f"Filter executable not found or not executable: {filter_executable}")

                if i == len(custom_filters) - 1:
                    step_output_path = output_file_path
                else:
                    temp_fd, step_output_path = tempfile.mkstemp(prefix=f"filter_{i}_", suffix=".out")
                    os.close(temp_fd) # Close the file descriptor, we just need the name
                    temp_files.append(step_output_path)

                current_filter_opts = filter_options.get(filter_name, "")

                # Standard CUPS filter arguments: job-id user title copies options [filename]
                # We read from stdin, so filename is omitted.
                filter_args = [
                    filter_executable,
                    "1",                 # job-id
                    "testuser",          # user
                    "Custom Chain Test", # title
                    "1",                 # copies
                    current_filter_opts  # Use specific options for this filter
                ]

                print(f"  Step {i+1}/{len(custom_filters)}: Running {filter_name}...")
                print(f"    Command: {' '.join(filter_args)}")
                print(f"    Input: {current_input_path}")
                print(f"    Output: {step_output_path}")

                with open(current_input_path, 'rb') as infile, open(step_output_path, 'wb') as outfile:
                    process = subprocess.run(filter_args, stdin=infile, stdout=outfile, stderr=subprocess.PIPE, check=True)

                if current_input_path != input_file_path:
                     if current_input_path in temp_files:
                          try:
                              # os.remove(current_input_path) # Keep intermediate files for debugging for now
                              pass
                          except OSError as e:
                              print(f"Warning: Could not remove intermediate file {current_input_path}: {e}")
                              
                current_input_path = step_output_path

            print(f"Successfully ran custom filter chain. Final output: {output_file_path}")
            return output_file_path

        except subprocess.CalledProcessError as e:
            error_message = f"Error running filter '{filter_name}' in custom chain: {e}\nStderr: {e.stderr.decode()}"
            print(error_message)
            raise RuntimeError(error_message) from e
        except FileNotFoundError as e:
             error_message = f"Filter executable error: {e}"
             print(error_message)
             raise FileNotFoundError(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error running custom filter chain: {e}"
            print(error_message)
            raise RuntimeError(error_message) from e
        finally:
            # print(f"Intermediate files kept for debugging: {temp_files}")
            for tmp_file in temp_files:
                 if tmp_file != output_file_path and os.path.exists(tmp_file): # Don't delete final output if it was a temp name
                      try:
                           os.remove(tmp_file)
                           # print(f"Removed intermediate file: {tmp_file}")
                      except OSError as e:
                           print(f"Warning: Could not remove intermediate file {tmp_file}: {e}")

    else:
        # --- Original PPD-based cupsfilter Logic ---
        try:
            command = [
                'cupsfilter',
                '-p', printer_name,        # Target printer
                '-m', target_mime_type,    # Target output MIME type
                '-o', 'document-format=' + target_mime_type, # Explicitly request output format
                input_file_path            # Input file
            ]
            print(f"Running PPD-based filter chain command: {' '.join(command)}")

            with open(output_file_path, 'wb') as outfile:
                result = subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, check=True)

            print(f"Successfully generated output via PPD chain: {output_file_path}")
            return output_file_path

        except subprocess.CalledProcessError as e:
            error_message = f"Error running PPD-based cupsfilter on {input_file_path}: {e}\nStderr: {e.stderr.decode()}"
            print(error_message)
            raise RuntimeError(error_message) from e
        except FileNotFoundError:
            error_message = "Error: 'cupsfilter' command not found. Is CUPS installed correctly?"
            print(error_message)
            raise FileNotFoundError(error_message)
        except Exception as e:
            error_message = f"Unexpected error in PPD-based run_cups_filter_chain: {e}"
            print(error_message)
            raise RuntimeError(error_message) from e

def get_comprehensive_printer_info(printer_name: str) -> str:
    """Get comprehensive information about printer filters and processing."""
    info = []
    info.append(f"=== Detailed Information for Printer: {printer_name} ===\n")
    
    # 1. Get printer model
    model = get_printer_model(printer_name)
    info.append(f"Model: {model}\n")
    
    # 2. Check if it's a CUPS-PDF printer (virtual) or a real printer
    if "PDF" in model or "pdf" in model:
        info.append("Type: Virtual PDF Printer\n")
    else:
        info.append("Type: Physical Printer\n")
    
    # 3. Get filter information from PPD
    ppd_info = get_printer_ppd_info(printer_name)
    if ppd_info:
        if "model" in ppd_info:
            info.append(f"PPD Model Name: {ppd_info['model']}")
        
        if "filters" in ppd_info:
            info.append("\nFilter Chain from PPD:")
            for f in ppd_info["filters"]:
                info.append(f"  {f}")
        else:
            info.append("\nNo cupsFilter directives found in PPD")
    else:
        info.append("No PPD file available or accessible. This might be a driverless printer.")
        
    
    # 4. Try to get CUPS filter chain information for PDF input
    info.append("\nAttempting to determine CUPS processing path for PDF input:")
    filters = get_filter_chain(printer_name, 'application/pdf')
    if filters:
        info.append("Filter Chain (PDF -> Printer):")
        for f in filters:
             info.append(f"  -> {f}")
    else:
        info.append("Could not determine filter chain using cupsfilter.")
        
    # 5. Check for cups-browsed involvement
    try:
        browsed_output = subprocess.check_output(['systemctl', 'status', 'cups-browsed'], 
                                                text=True, stderr=subprocess.STDOUT)
        info.append("\nCups-browsed service is running (may create implicit queues)")
    except:
        info.append("\nCups-browsed service not running or not installed")
    
    # 6. Check if it's an IPP Everywhere or driverless printer
    try:
        lpinfo_output = subprocess.check_output(['lpinfo', '-m'], text=True)
        if "driverless" in lpinfo_output and printer_name.lower() in lpinfo_output.lower():
            info.append("\nThis appears to be a driverless/IPP Everywhere printer")
    except:
        pass
        
    return "\n".join(info)

def print_printer_info() -> None:
    """Print information about all available printers."""
    printers = get_available_printers()
    
    if not printers:
        print("No printers found.")
        return
        
    print("\n=== Available Printers ===")
    for printer in printers:
        detailed_info = get_comprehensive_printer_info(printer)
        print(detailed_info)
        print("-" * 75)

def parse_cups_debug_log_for_job(job_id: int, log_file: str = "/var/log/cups/error_log") -> Optional[List[Tuple[str, str]]]:
    """
    Parses CUPS debug log (LogLevel debug2) to find the filter chain and options
    used for a specific job ID.

    NOTE: This requires CUPS LogLevel to be set to debug2 and appropriate
          permissions to read the log file. Implementation is complex.

    Returns:
        A list of tuples: [(filter_name, options_string), ...] or None if not found.
    """
    print(f"--- Attempting to parse CUPS debug log ({log_file}) for Job ID: {job_id} ---")
    print("--- NOTE: This requires LogLevel debug2 and read permissions. ---")
    
    filter_chain_with_options = []
    try:
        with open(log_file, 'r') as f:
            current_filter = None
            job_str = f"[Job {job_id}]"
            for line in f:
                if job_str in line:
                    if "Started filter" in line:
                        match = re.search(r'Started filter.*?/([^/\s]+)\s*\(PID', line)
                        if match:
                            current_filter = match.group(1)
                    elif current_filter and "argv[5]" in line: # argv[5] is typically the options string
                         options_match = re.search(r'argv\[5\]="(.*)"', line)
                         if options_match:
                              options_str = options_match.group(1)
                              filter_chain_with_options.append((current_filter, options_str))
                              current_filter = None
    except FileNotFoundError:
        print(f"Error: CUPS log file not found at {log_file}")
        return None
    except PermissionError:
         print(f"Error: Permission denied reading CUPS log file {log_file}")
         return None
    except Exception as e:
        print(f"Error parsing CUPS log: {e}")
        return None

    if not filter_chain_with_options:
        print(f"Could not find filter details for Job ID {job_id} in log.")
        return None
        
    print(f"Extracted filter chain and options for Job {job_id}: {filter_chain_with_options}")
    return filter_chain_with_options

if __name__ == "__main__":
    print_printer_info()
    
    # Example usage with custom filter chain:
    available_printers = get_available_printers()
    if available_printers:
        test_printer = available_printers[0]
        input_pdf = "sample.pdf" # Assuming generate_pdf.py has run
        output_raster_custom = "output_custom.raster"
        
        # Define your complex filter chain
        # Example: PDF -> PostScript -> Raster
        # Note: Ensure these filters exist in your CUPS filter directory (e.g., /usr/lib/cups/filter/)
        my_complex_chain = ['pdftops', 'gstoraster'] 
        
        # Define options for specific filters
        my_filter_options = {
            'pdftopdf': 'Collate=True number-up=1', # Example options for pdftopdf
            'pdftoraster': 'Resolution=300 ColorModel=RGB' # Example options for pdftoraster
        }

        print(f"\nTesting custom filter chain with options for file: {input_pdf}")
        if not os.path.exists(input_pdf):
             print(f"Input file {input_pdf} not found. Run generate_pdf.py first.")
        else:
            try:
                run_cups_filter_chain(
                    printer_name=test_printer,
                    input_file_path=input_pdf, 
                    output_file_path=output_raster_custom,
                    target_mime_type='image/pwg-raster', # Example target
                    input_mime_type='application/pdf',   # MUST specify input type
                    custom_filters=my_complex_chain,
                    filter_options=my_filter_options # Pass the options dict
                )
                print(f"Custom raster output generated successfully: {output_raster_custom}")
                # Add raster to PNG conversion here if needed for comparison
            except Exception as e:
                print(f"Failed to run custom filter chain: {e}")

    # Example usage with PPD-based chain (original behavior):
    # ... (similar structure as above, but without custom_filters and input_mime_type)