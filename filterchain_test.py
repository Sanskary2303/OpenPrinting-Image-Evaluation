from filter_chain import get_filter_chain
from printer_info import get_available_printers
import subprocess
import tempfile
import os
import sys
from typing import List, Dict, Tuple, Optional

# Known common filter chains for validation
COMMON_FILTER_CHAINS = {
    ('application/pdf', 'application/vnd.cups-raster'): ['pdftopdf', 'pdftoraster'],
    ('application/pdf', 'image/pwg-raster'): ['pdftopdf', 'pdftoraster', 'rastertopwg'],
    ('text/plain', 'application/pdf'): ['texttopdf'],
    ('text/plain', 'application/vnd.cups-raster'): ['texttopdf', 'pdftoraster'],
    ('image/jpeg', 'application/pdf'): ['imagetopdf'],
    ('application/postscript', 'application/pdf'): ['gstopdf'],
}

def get_reference_filter_chain(printer_name: str, input_mime_type: str, 
                              output_mime_type: Optional[str] = None) -> List[str]:
    """
    Get the reference filter chain using cupsfilter command.
    This is our "ground truth" for validation.
    """
    try:
        with tempfile.NamedTemporaryFile() as dummy_file:
            command = [
                'cupsfilter',
                '--list-filters',
                '-p', printer_name,
                '-i', input_mime_type,
            ]
            
            if output_mime_type:
                command.extend(['-m', output_mime_type])
                
            # Add dummy file path (required by cupsfilter)
            command.append(dummy_file.name)
            
            print(f"Running reference command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                print(f"Warning: cupsfilter command failed with code {result.returncode}")
                print(f"Error: {result.stderr}")
                return []
                
            output = result.stdout if result.stdout else result.stderr
            filters = [line.strip() for line in output.splitlines() 
                      if line.strip() and not line.startswith("DEBUG:")]
                
            return filters
    except Exception as e:
        print(f"Error getting reference filter chain: {e}")
        return []

def chains_match(chain1: List[str], chain2: List[str]) -> bool:
    """
    Compare two filter chains, considering various match patterns:
    1. Direct equality
    2. One chain is a prefix of the other
    3. Equivalent filters (pdftoraster vs gstoraster)
    """
    if chain1 == chain2:
        return True
        
    # Empty chains
    if not chain1 and not chain2:
        return True
    
    # Check if one chain is a prefix of the other
    if len(chain1) > len(chain2) and chain1[:len(chain2)] == chain2:
        return True
    if len(chain2) > len(chain1) and chain2[:len(chain1)] == chain1:
        return True
        
    # Known equivalences
    equivalences = {
        'gstoraster': 'pdftoraster',     # often interchangeable
        'pdftops': 'gstops',             # PostScript converters
        'rastertopdf': 'rastertopwg',    # Raster output converters
        'pstotiff': 'pdftopdf', 
        'imagetoraster': 'gstoraster',  
    }
    
    # Check if chains have same length and equivalent filters
    if len(chain1) == len(chain2):
        match = True
        for i in range(len(chain1)):
            if chain1[i] != chain2[i]:
                if chain1[i] in equivalences and equivalences[chain1[i]] == chain2[i]:
                    continue
                if chain2[i] in equivalences and equivalences[chain2[i]] == chain1[i]:
                    continue
                match = False
                break
        if match:
            return True
    
    # Check prefix with equivalences
    if len(chain1) > len(chain2):
        match = True
        for i in range(len(chain2)):
            if chain1[i] != chain2[i]:
                if chain1[i] in equivalences and equivalences[chain1[i]] == chain2[i]:
                    continue
                if chain2[i] in equivalences and equivalences[chain2[i]] == chain1[i]:
                    continue
                match = False
                break
        if match:
            return True
    
    if len(chain2) > len(chain1):
        match = True
        for i in range(len(chain1)):
            if chain1[i] != chain2[i]:
                if chain1[i] in equivalences and equivalences[chain1[i]] == chain2[i]:
                    continue
                if chain2[i] in equivalences and equivalences[chain2[i]] == chain1[i]:
                    continue
                match = False
                break
        if match:
            return True
        
    return False

def validate_filter_chain(printer_name: str, input_mime: str, 
                          output_mime: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate the filter chain detection by comparing programmatic result with reference.
    
    Returns:
        Tuple of (success, message)
    """
    print(f"\n--- Testing {input_mime} → {output_mime or 'auto'} ---")
    
    # Get filter chain using our programmatic method
    prog_filters = get_filter_chain(printer_name, input_mime, output_mime)
    
    # Get reference filter chain using cupsfilter command
    ref_filters = get_reference_filter_chain(printer_name, input_mime, output_mime)
    
    # Check if our result matches the reference
    matches = chains_match(prog_filters, ref_filters)
    
    # Format the chains for display
    prog_str = ' → '.join(prog_filters) if prog_filters else "No filters"
    ref_str = ' → '.join(ref_filters) if ref_filters else "No filters"
    
    if matches:
        result = "✓ PASS"
        msg = f"{result}: Filter chains match for {input_mime} → {output_mime or 'auto'}\n"
        msg += f"  Chain: {prog_str}"
    else:
        result = "✗ FAIL"
        msg = (f"{result}: Filter chains don't match for {input_mime} → {output_mime or 'auto'}\n"
               f"  Programmatic: {prog_str}\n"
               f"  Reference:    {ref_str}")
        
    return matches, msg

def generate_test_files():
    """Generate test files for different MIME types"""
    from generate_test_documents import create_test_documents
    
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate test files for different MIME types
    create_test_documents(test_dir)
    print(f"Test documents generated in {test_dir}/")

def test_filter_chains():
    """Test filter chain detection for available printers"""
    printers = get_available_printers()
    if not printers:
        print("No printers available")
        return False
    
    # printer to test with
    test_printer = printers[0]
    print(f"Testing filter chains for printer: {test_printer}")
    
    # Define test cases: (input_mime, output_mime)
    # Basic tests
    basic_tests = [
        ('application/pdf', None),
        ('application/pdf', 'application/vnd.cups-raster'),
        ('application/pdf', 'image/pwg-raster'),
        ('text/plain', None),
        ('text/plain', 'application/pdf'),
        ('image/jpeg', None),
        ('image/jpeg', 'application/pdf'),
        ('application/postscript', None),
    ]
    
    # common input formats
    image_tests = [
        ('image/png', None),
        ('image/png', 'application/pdf'),
        ('image/png', 'application/vnd.cups-raster'),
        ('image/tiff', None),
        ('image/tiff', 'application/pdf'),
    ]
    
    # Office document formats
    office_tests = [
        ('application/msword', None),
        ('application/vnd.ms-excel', None),
        ('application/vnd.oasis.opendocument.text', None),
        ('application/rtf', None),
    ]
    
    # Code and text formats
    code_tests = [
        ('application/x-cshell', None),
        ('application/x-csource', None),
        ('application/x-perl', None),
        ('application/x-shell', None),
    ]
    
    # Additional output formats
    output_tests = [
        ('application/pdf', 'application/postscript'),
        ('application/pdf', 'application/pcl'),
        ('text/plain', 'application/postscript'),
        ('image/jpeg', 'application/postscript'),
    ]
    
    all_tests = basic_tests
    
    run_extensive_tests = True  # Set to True for more thorough testing
    if run_extensive_tests:
        all_tests.extend(image_tests)
        all_tests.extend(office_tests)
        all_tests.extend(code_tests)
        all_tests.extend(output_tests)
    
    all_passed = True
    results = []
    
    for input_mime, output_mime in all_tests:
        try:
            passed, message = validate_filter_chain(test_printer, input_mime, output_mime)
            results.append((passed, message))
            all_passed = all_passed and passed
        except Exception as e:
            print(f"Error testing {input_mime} → {output_mime}: {e}")
            results.append((False, f"✗ ERROR: Test failed with exception for {input_mime} → {output_mime}: {e}"))
            all_passed = False
    
    print("\n=== TEST RESULTS ===")
    passed_count = sum(1 for passed, _ in results if passed)
    print(f"Passed: {passed_count}/{len(results)} tests\n")
    
    passed_tests = [msg for passed, msg in results if passed]
    failed_tests = [msg for passed, msg in results if not passed]
    
    if failed_tests:
        print("FAILED TESTS:")
        for msg in failed_tests:
            print(msg)
        print()
    
    if passed_tests:
        print("PASSED TESTS:")
        for msg in passed_tests:
            print(msg)
    
    print(f"\nOverall result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-files":
        generate_test_files()
    else:
        success = test_filter_chains()
        sys.exit(0 if success else 1)