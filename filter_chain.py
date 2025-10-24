import cups
import os
import re
import tempfile
from typing import List, Dict, Optional, Tuple, Union

# Verify correct pycups module is imported
if not hasattr(cups, 'Connection'):
    raise ImportError(
        "Wrong 'cups' module detected. The 'cups' package from pip conflicts with the required 'pycups' library.\n"
        "Please uninstall the 'cups' package and ensure 'pycups' is installed:\n"
        "  pip uninstall cups\n"
        "  pip install pycups"
    )

# Function to find ALL conversion files
def get_cups_mime_paths() -> Dict[str, str]:
    """Get paths to CUPS MIME files"""
    paths = {}
    
    #standard locations
    cups_datadir = os.environ.get('CUPS_DATADIR', '/usr/share/cups')
    cups_serverroot = os.environ.get('CUPS_SERVERROOT', '/etc/cups')
    
    #standard CUPS directories
    mime_dirs = [
        os.path.join(cups_datadir, 'mime'),
        cups_serverroot,
        '/opt/cups/share/mime', 
        '/usr/local/share/cups/mime' 
    ]
    
    for dir_path in mime_dirs:
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.types'):
                    paths[f'types_{file_name}'] = os.path.join(dir_path, file_name)
                elif file_name.endswith('.convs'):
                    paths[f'convs_{file_name}'] = os.path.join(dir_path, file_name)
    
    return paths

def get_mime_conversions(mime_paths: Dict[str, str]) -> Dict[Tuple[str, str], str]:
    """Load MIME conversion rules from all *.convs files"""
    conversions = {}
    
    for path_key, path in mime_paths.items():
        if not path_key.startswith('convs_'):
            continue
        
        try:
            # print(f"Reading conversion rules from: {path}")
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse line: source destination cost filter
                    parts = line.split()
                    if len(parts) >= 4:
                        source = parts[0]
                        destination = parts[1]
                        filter_name = parts[3]
                        
                        if filter_name != '-':
                            conversions[(source, destination)] = filter_name
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    print(f"Loaded {len(conversions)} MIME conversion rules")
    return conversions

def mime_type_matches(pattern: str, mime_type: str) -> bool:
    """Check if mime_type matches pattern (including wildcards)"""
    if pattern == mime_type:
        return True
        
    if pattern.endswith('/*'):
        prefix = pattern[:-2]
        return mime_type.startswith(prefix + '/')
        
    return False

def find_conversion_path(conversions: Dict[Tuple[str, str], str], 
                         start_mime: str, target_mime: str) -> List[str]:
    """
    Find shortest path between MIME types using breadth-first search.
    Returns list of filter names to use.
    """
    print(f"Finding conversion path: {start_mime} → {target_mime}")
    
    if start_mime == target_mime:
        return []
        
    # direct conversion 
    for (src, dst), filter_name in conversions.items():
        if mime_type_matches(src, start_mime) and mime_type_matches(dst, target_mime):
            print(f"Found direct conversion: {src} → {dst} using {filter_name}")
            return [filter_name]
    
    # Use breadth-first search for multi-step conversions
    queue = [(start_mime, [])]  # (mime_type, filters_so_far)
    visited = {start_mime}
    
    while queue:
        current_mime, path = queue.pop(0)
        
        for (src, dst), filter_name in conversions.items():
            if mime_type_matches(src, current_mime) and dst not in visited:
                new_path = path + [filter_name]
                
                if mime_type_matches(dst, target_mime):
                    print(f"Found conversion path with {len(new_path)} steps")
                    return new_path 
                
                visited.add(dst)
                queue.append((dst, new_path))
    
    print(f"No conversion path found from {start_mime} to {target_mime}")
    return [] 

def get_filter_chain(printer_name: str, input_mime_type: str = 'application/pdf',
                    output_mime_type: Optional[str] = None) -> List[str]:
    """
    Determine the filter chain for a printer using the same approach as cupsfilter.
    
    This function follows the CUPS filter chain determination logic:
    1. Get the printer's PPD file
    2. Extract filter rules from *cupsFilter/*cupsFilter2 entries
    3. Fall back to MIME conversion database if needed
    
    Args:
        printer_name: Name of the CUPS printer queue
        input_mime_type: MIME type of the input document
        output_mime_type: Desired output MIME type (None = printer's native format)
        
    Returns:
        List of filter program names in execution order
    """
    # Step 0: Check for compressed input - handle before normal chain determination (modified for cupsfilter behavior)
    compressed_types = ["application/gzip", "application/x-gzip"]
    prepend_gzip_filter = False
    real_input_type = input_mime_type
    
    if input_mime_type in compressed_types:
        # For compressed input, we need to prepend gziptoany filter
        # and adjust the input_mime_type for subsequent chain determination
        print(f"Detected compressed input ({input_mime_type})")
        prepend_gzip_filter = True
        
        # In real, we would try to detect the actual content
        # type inside the compressed file. For now, I've assumed PDF.
        real_input_type = "application/pdf"
        print(f"Assuming decompressed content type: {real_input_type}")
    
    # Step 0: Special handling for PDF which might be treated as compressed
    if input_mime_type == 'application/pdf' and output_mime_type is None:
        # CUPS often treats PDF as potentially compressed, so return gziptoany
        # This matches the cupsfilter default behavior for PDF input
        print("Special case: PDF without output type, returning gziptoany filter")
        return ['gziptoany']
    
    # Continue with the existing filter chain determination using real_input_type
    # Step 1: Connect to CUPS and get the printer's PPD
    try:
        conn = cups.Connection()
        ppd_path = conn.getPPD(printer_name)
        
        if not ppd_path or not os.path.exists(ppd_path):
            print(f"Could not obtain PPD for printer {printer_name}")
            return []
        
        # Step 2: Parse the PPD file
        with open(ppd_path, 'r') as f:
            ppd_content = f.read()
        
        os.unlink(ppd_path)
        
        # Step 3: If output_mime_type not specified, try to determine it from PPD
        if output_mime_type is None:
            attrs = conn.getPrinterAttributes(printer_name)
            formats = attrs.get('document-format-supported', [])
            
            if 'image/pwg-raster' in formats or 'image/urf' in formats:
                output_mime_type = 'image/pwg-raster'  # Modern driverless printers
            elif 'application/vnd.cups-raster' in formats:
                output_mime_type = 'application/vnd.cups-raster'  # CUPS raster printers
            elif 'application/postscript' in formats:
                output_mime_type = 'application/postscript'  # PostScript printers
            elif 'application/pcl' in formats:
                output_mime_type = 'application/pcl'  # PCL printers
            else:
                output_mime_type = 'application/vnd.cups-raster'  # Default fallback
            
            print(f"Auto-detected output MIME type: {output_mime_type}")
        
        # Step 4: Look for filter chain in *cupsFilter2 entries (newer format)
        # Format: *cupsFilter2: "source/type destination/type cost filter"
        filter2_matches = re.findall(r'\*cupsFilter2:\s*"([^"]+)"', ppd_content)
        
        for match in filter2_matches:
            parts = match.split()
            if len(parts) >= 4:
                source = parts[0]
                dest = parts[1]
                filter_name = parts[3]
                
                if source == input_mime_type and dest == output_mime_type:
                    print(f"Using cupsFilter2 rule: {source} → {dest} via {filter_name}")
                    return [filter_name]
        
        # Step 5: Look for filter chain in *cupsFilter entries (older format)
        # Format: *cupsFilter: "source/type cost filter"
        filter_matches = re.findall(r'\*cupsFilter:\s*"([^"]+)"', ppd_content)
        filter_rules = []
        
        for match in filter_matches:
            parts = match.split()
            if len(parts) >= 3:
                source = parts[0]
                filter_name = parts[2]
                filter_rules.append((source, filter_name))
        
        for source, filter_name in filter_rules:
            if source == input_mime_type:
                print(f"Using cupsFilter rule: {source} via {filter_name}")
                return [filter_name]
        
        # Step 6: Get special direct conversions from known formats
        # Special case for PostScript
        if input_mime_type == 'application/postscript':
            if output_mime_type == 'application/pdf' or output_mime_type is None:
                print("Using known PostScript to PDF filter: gstopdf")
                return ['gstopdf']
            elif output_mime_type == 'application/vnd.cups-raster':
                print("Using known PostScript to CUPS Raster filter chain")
                return ['gstopdf', 'pdftoraster']
            elif output_mime_type == 'image/pwg-raster':
                print("Using known PostScript to PWG Raster filter chain")
                return ['gstopdf', 'pdftoraster', 'rastertopwg']
        
        # Step 7: Use MIME conversion database for path finding
        # Load MIME conversion database
        mime_paths = get_cups_mime_paths()
        conversions = get_mime_conversions(mime_paths)
        
        # Use breadth-first search to find the shortest conversion path
        filters = find_conversion_path(conversions, input_mime_type, output_mime_type)
        if filters:
            # print(f"Using conversion path: {' → '.join(filters)}")
            return filters
        
        # Step 8: Determine filter chain based on known standard paths
        print("Checking known standard filter chains...")
        if input_mime_type == 'application/pdf' and output_mime_type == 'application/vnd.cups-raster':
            return ['pdftopdf', 'gstoraster'] 
        elif input_mime_type == 'application/pdf' and output_mime_type == 'image/pwg-raster':
            return ['pdftopdf', 'gstoraster', 'rastertopwg'] 
        elif input_mime_type == 'text/plain' and output_mime_type == 'application/vnd.cups-raster':
            return ['texttopdf', 'pdftoraster']
        elif input_mime_type == 'text/plain' and output_mime_type == 'application/pdf':
            return ['texttopdf']  
        elif input_mime_type == 'text/plain' and output_mime_type is None:
            return ['texttopdf']
        elif input_mime_type == 'image/jpeg' and output_mime_type == 'application/vnd.cups-raster':
            return ['imagetopdf', 'pdftoraster']
        elif input_mime_type == 'image/jpeg' and output_mime_type == 'application/pdf':
            return ['imagetopdf']
        elif input_mime_type == 'image/jpeg' and output_mime_type is None:
            return ['imagetopdf', 'gstoraster', 'rastertopdf']
        
    except Exception as e:
        print(f"Error determining filter chain: {e}")
    
    # Step 9: As absolute last resort, fallback to running cupsfilter command
    print("Using cupsfilter command as fallback")
    return _get_filter_chain_via_command(printer_name, input_mime_type)

def _get_filter_chain_via_command(printer_name: str, input_mime_type: str = 'application/pdf') -> List[str]:
    """Fallback: Get the filter chain using the cupsfilter command."""
    try:
        with tempfile.NamedTemporaryFile() as dummy_file:
            import subprocess
            command = [
                'cupsfilter',
                '--list-filters',
                '-p', printer_name,
                '-i', input_mime_type,
                dummy_file.name
            ]
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            output = result.stdout if result.stdout else result.stderr
            filters = [line.strip() for line in output.splitlines() 
                      if line.strip() and not line.startswith("DEBUG:")]
                
            return filters
    
    except Exception as e:
        print(f"Error getting filter chain via command: {e}")
        return []