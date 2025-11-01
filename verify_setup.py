#!/usr/bin/env python3
"""
Verify that the system is properly configured for running the test pipelines
"""

import subprocess
import os
import sys

def check_cups_pdf():
    """Check if cups-pdf is installed and find the printer queue"""
    print("🔍 Checking for virtual PDF printer...")
    
    try:
        result = subprocess.run(['lpstat', '-p'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ CUPS not running or lpstat command failed")
            print("   Make sure CUPS is installed and running")
            return False, None
            
        printers = result.stdout.strip().split('\n')
        
        pdf_printers = []
        for line in printers:
            if line and ('PDF' in line or 'pdf' in line):
                # Extract printer name (format: "printer NAME is idle/disabled")
                parts = line.split()
                if len(parts) >= 2 and parts[0] == 'printer':
                    pdf_printers.append(parts[1])
        
        if pdf_printers:
            print(f"✅ Found virtual PDF printer(s): {', '.join(pdf_printers)}")
            print(f"\n💡 Use this command:")
            print(f"   ./print_quality_pipeline.py {pdf_printers[0]}")
            return True, pdf_printers[0]
        else:
            print("❌ No virtual PDF printer found")
            print("\n📋 To install cups-pdf:")
            print("   Fedora/RHEL:   sudo dnf install cups-pdf")
            print("   Ubuntu/Debian: sudo apt install printer-driver-cups-pdf")
            print("                  sudo systemctl daemon-reload")
            print("\n   After installation, the printer will be named:")
            print("   • 'Cups-PDF' on Fedora/RHEL")
            print("   • 'PDF' on Ubuntu/Debian")
            return False, None
            
    except FileNotFoundError:
        print("❌ lpstat command not found - CUPS may not be installed")
        print("\n📋 To install CUPS:")
        print("   Fedora/RHEL:   sudo dnf install cups")
        print("   Ubuntu/Debian: sudo apt-get install cups")
        return False, None
    except Exception as e:
        print(f"❌ Error checking printers: {e}")
        return False, None

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n🔍 Checking Python dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'numpy': 'numpy',
        'skimage': 'scikit-image',
        'cups': 'pycups',
        'reportlab': 'reportlab'
    }
    
    missing = []
    wrong_cups = False
    
    for package, pip_name in required.items():
        try:
            module = __import__(package)
            # Special check for cups module to ensure it's pycups
            if package == 'cups' and not hasattr(module, 'Connection'):
                print(f"⚠️  {package:12} ({pip_name}) - WRONG MODULE DETECTED")
                print(f"    You have the 'cups' pip package, but need 'pycups'")
                wrong_cups = True
                missing.append('pycups')
            else:
                print(f"✅ {package:12} ({pip_name})")
        except ImportError:
            print(f"❌ {package:12} ({pip_name}) - MISSING")
            missing.append(pip_name)
    
    if wrong_cups:
        print(f"\n⚠️  IMPORTANT: Wrong 'cups' module detected!")
        print(f"   Uninstall the wrong package and install pycups:")
        print(f"   pip uninstall cups")
        print(f"   pip install pycups")
        return False
    
    if missing:
        print(f"\n📋 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_scripts_executable():
    """Check if main scripts are executable"""
    print("\n🔍 Checking if scripts are executable...")
    
    scripts = [
        'print_quality_pipeline.py',
        'filter_chain_test_pipeline.py',
        'test_document_pipeline.py',
        'example_usage.py',
        'test_pipeline.py',
        'multipage_test.py'
    ]
    
    all_executable = True
    for script in scripts:
        if os.path.exists(script):
            is_exec = os.access(script, os.X_OK)
            if is_exec:
                print(f"✅ {script}")
            else:
                print(f"⚠️  {script} (not executable - will still work with 'python')")
                all_executable = False
        else:
            print(f"❌ {script} (not found)")
    
    if not all_executable:
        print(f"\n📋 To make scripts executable (optional but convenient):")
        print(f"   chmod 755 *.py")
    
    return True  # Not critical, scripts work with 'python' command

def check_shebang():
    """Check if main scripts have shebang line"""
    print("\n🔍 Checking shebang lines...")
    
    scripts = [
        'print_quality_pipeline.py',
        'filter_chain_test_pipeline.py',
        'test_document_pipeline.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            with open(script, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!') and 'python' in first_line:
                    print(f"✅ {script}")
                else:
                    print(f"⚠️  {script} (missing shebang)")
    
    return True

def main():
    print("=" * 60)
    print("OpenPrinting Image Quality Evaluation - Setup Verification")
    print("=" * 60)
    
    # Check all requirements
    has_printer, printer_name = check_cups_pdf()
    has_deps = check_dependencies()
    has_exec = check_scripts_executable()
    check_shebang()
    
    print("\n" + "=" * 60)
    if has_printer and has_deps:
        print("🎉 SETUP COMPLETE - Ready to run!")
        print("=" * 60)
        print("\n🚀 Quick Start Commands:")
        if printer_name:
            print(f"\n   # Option 1: Direct execution (if chmod 755 was run)")
            print(f"   ./print_quality_pipeline.py {printer_name}")
            print(f"   ./filter_chain_test_pipeline.py {printer_name}")
            print(f"   ./test_document_pipeline.py")
            print(f"   ./example_usage.py")
            print(f"\n   # Option 2: With python command (always works)")
            print(f"   python print_quality_pipeline.py {printer_name}")
            print(f"   python filter_chain_test_pipeline.py {printer_name}")
            print(f"   python test_document_pipeline.py")
            print(f"   python example_usage.py")
        
        print("\n📚 See quick_start_examples/ for detailed HOWTOs")
        return 0
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
        print("=" * 60)
        print("\n📋 What you need:")
        if not has_printer:
            print("   1. Install cups-pdf package")
        if not has_deps:
            print("   2. Install missing Python packages")
        return 1

if __name__ == "__main__":
    sys.exit(main())
