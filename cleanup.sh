#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/clean_workspace.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: clean_workspace.py not found at $PYTHON_SCRIPT${NC}"
    exit 1
fi

print_banner() {
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}       OpenPrinting Cleanup Tool       ${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""
}

print_help() {
    echo -e "Usage: $0 [OPTION]"
    echo -e "Clean up workspace files from OpenPrinting tests.\n"
    echo -e "Options:"
    echo -e "  ${GREEN}quick${NC}             Quick cleanup (temp files only)"
    echo -e "  ${GREEN}full${NC}              Full cleanup (everything except reports)"
    echo -e "  ${GREEN}all${NC}               Remove everything (including reports)"
    echo -e "  ${GREEN}backup${NC}            Backup reports before cleaning"
    echo -e "  ${GREEN}reports-only${NC}      Keep only the reports, remove everything else"
    echo -e "  ${GREEN}cups${NC}              Clean only the CUPS output directory"
    echo -e "  ${GREEN}check${NC}             Show what would be cleaned (dry run)"
    echo -e "  ${GREEN}help${NC}              Display this help and exit"
    echo -e "\nThis script is a wrapper around clean_workspace.py."
    echo -e "For more options: python3 clean_workspace.py --help"
}

check_disk_space() {
    echo -e "${BLUE}Current disk space:${NC}"
    df -h . | tail -n 1
}

confirm_action() {
    echo -e "${YELLOW}$1${NC}"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Operation cancelled.${NC}"
        exit 1
    fi
}

main() {
    print_banner
    
    if [ $# -eq 0 ]; then
        print_help
        exit 0
    fi

    case "$1" in
        quick)
            confirm_action "This will remove temporary files in the root directory."
            python3 "$PYTHON_SCRIPT" --temp-files
            ;;
        full)
            confirm_action "This will clean temporary files, test documents, test results, and CUPS output (keeping reports)."
            python3 "$PYTHON_SCRIPT" --all --keep-reports
            ;;
        all)
            confirm_action "WARNING: This will remove ALL files, including reports!"
            python3 "$PYTHON_SCRIPT" --all
            ;;
        backup)
            confirm_action "This will backup reports and then clean everything."
            python3 "$PYTHON_SCRIPT" --backup --all
            ;;
        reports-only)
            confirm_action "This will keep only reports, removing all other files."
            python3 "$PYTHON_SCRIPT" --all --keep-reports --keep-json
            ;;
        cups)
            confirm_action "This will clean only the CUPS output directory."
            python3 "$PYTHON_SCRIPT" --cups-output
            ;;
        check)
            echo -e "${BLUE}Checking workspace...${NC}"
            echo -e "Test documents directory: ${YELLOW}$(find test_documents -type f 2>/dev/null | wc -l)${NC} files"
            echo -e "Test results directory: ${YELLOW}$(find test_results -type f 2>/dev/null | wc -l)${NC} files"
            echo -e "Temporary files: ${YELLOW}$(find . -maxdepth 1 -name "*.png" -o -name "*.pdf" | wc -l)${NC} files"
            echo -e "CUPS output: ${YELLOW}$(find PDF -type f 2>/dev/null | wc -l)${NC} files"
            echo -e "\nRun with another option to clean files."
            ;;
        help|--help|-h)
            print_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac

}

main "$@"