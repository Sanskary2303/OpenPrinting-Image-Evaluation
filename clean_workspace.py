#!/usr/bin/env python3

import os
import shutil
import argparse
import glob
import datetime
import zipfile

def create_backup(backup_dir, test_documents_dir, test_results_dir):
    """Create a backup of important test results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"openprinting_results_{timestamp}"
    backup_path = os.path.join(backup_dir, backup_name)
    os.makedirs(backup_path, exist_ok=True)
    
    summary_files = [
        os.path.join(test_documents_dir, "test_summary.csv"),
        os.path.join(test_documents_dir, "test_report.html")
    ]
    for file in summary_files:
        if os.path.exists(file):
            shutil.copy2(file, backup_path)
    
    for json_file in glob.glob(os.path.join(test_documents_dir, "*_results.json")):
        shutil.copy2(json_file, backup_path)
    
    if os.path.exists(test_results_dir):
        summary_files = [
            os.path.join(test_results_dir, "test_summary.csv"),
            os.path.join(test_results_dir, "test_report.html")
        ]
        for file in summary_files:
            if os.path.exists(file):
                shutil.copy2(file, backup_path)
        
        for json_file in glob.glob(os.path.join(test_results_dir, "*_results.json")):
            shutil.copy2(json_file, backup_path)
    
    zip_path = f"{backup_path}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(backup_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, backup_path)
                zipf.write(file_path, arcname)
    
    print(f"Created backup: {zip_path}")
    
    shutil.rmtree(backup_path)
    
    return zip_path

def cleanup_workspace(args):
    """Clean up the workspace based on provided arguments"""
    
    workspace_dir = os.getcwd()
    test_documents_dir = os.path.join(workspace_dir, "test_documents")
    test_results_dir = os.path.join(workspace_dir, "test_results")
    pdf_dir = os.path.join(workspace_dir, "PDF")
    
    if args.backup:
        backup_dir = args.backup_dir if args.backup_dir else workspace_dir
        create_backup(backup_dir, test_documents_dir, test_results_dir)
    
    total_files_before = sum(len(files) for _, _, files in os.walk(workspace_dir))
    
    if args.temp_files:
        temp_files = [
            "sample.png", "output.png", "sample.pdf", "output.pdf",
            "feature_matches.png", "edges_original.png", "edges_processed.png",
            "visual_diff.png"
        ]
        
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed: {file}")
    
    if args.test_documents and os.path.exists(test_documents_dir):
        if args.keep_reports:
            for file in os.listdir(test_documents_dir):
                file_path = os.path.join(test_documents_dir, file)
                if os.path.isfile(file_path):
                    if file in ["test_summary.csv", "test_report.html"]:
                        continue
                    if args.keep_json and file.endswith("_results.json"):
                        continue
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
        else:
            shutil.rmtree(test_documents_dir)
            print(f"Removed directory: {test_documents_dir}")
    
    if args.test_results and os.path.exists(test_results_dir):
        if args.keep_reports:
            for file in os.listdir(test_results_dir):
                file_path = os.path.join(test_results_dir, file)
                if os.path.isfile(file_path):
                    if file in ["test_summary.csv", "test_report.html"]:
                        continue
                    if args.keep_json and file.endswith("_results.json"):
                        continue
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
        else:
            shutil.rmtree(test_results_dir)
            print(f"Removed directory: {test_results_dir}")
    
    if args.cups_output and os.path.exists(pdf_dir):
        if args.remove_cups_dir:
            shutil.rmtree(pdf_dir)
            print(f"Removed directory: {pdf_dir}")
        else:
            for file in os.listdir(pdf_dir):
                file_path = os.path.join(pdf_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
    
    total_files_after = sum(len(files) for _, _, files in os.walk(workspace_dir))
    
    print(f"\nCleanup summary:")
    print(f"Files before cleanup: {total_files_before}")
    print(f"Files after cleanup: {total_files_after}")
    print(f"Removed {total_files_before - total_files_after} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up the OpenPrinting workspace after testing")
    
    parser.add_argument("--temp-files", action="store_true", 
                        help="Remove temporary files in root directory (sample.png, output.png, etc.)")
    parser.add_argument("--test-documents", action="store_true", 
                        help="Clean files in test_documents directory")
    parser.add_argument("--test-results", action="store_true",
                        help="Clean files in test_results directory")
    parser.add_argument("--cups-output", action="store_true", 
                        help="Clean CUPS output files in PDF directory")
    parser.add_argument("--keep-reports", action="store_true", 
                        help="Keep HTML and CSV reports when cleaning test directories")
    parser.add_argument("--keep-json", action="store_true", 
                        help="Keep JSON result files when cleaning test directories")
    parser.add_argument("--remove-cups-dir", action="store_true", 
                        help="Remove the entire PDF directory rather than just its contents")
    parser.add_argument("--backup", action="store_true", 
                        help="Create a backup of test results before cleaning")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to store backup (default: current directory)")
    parser.add_argument("--all", action="store_true", 
                        help="Clean everything (equivalent to --temp-files --test-documents --test-results --cups-output)")
    
    args = parser.parse_args()
    
    if args.all:
        args.temp_files = True
        args.test_documents = True
        args.test_results = True
        args.cups_output = True
    
    if not (args.temp_files or args.test_documents or args.test_results or args.cups_output or args.backup):
        parser.print_help()
    else:
        cleanup_workspace(args)