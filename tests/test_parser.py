#!/usr/bin/env python3
"""
Test script for document parser functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataroom.tools.parser import parse_csv, parse_pdf

def test_csv_parsing():
    """Test CSV parsing with pandas."""
    print("\n" + "="*50)
    print("Testing CSV Parsing")
    print("="*50)
    
    # Use the existing test CSV file
    csv_file = Path(__file__).parent.parent / "data" / "test_data.csv"
    
    if not csv_file.exists():
        print(f"❌ Test CSV file not found: {csv_file}")
        return False
    
    print(f"Testing CSV file: {csv_file}")
    result = parse_csv(str(csv_file))
    
    if "error" in result:
        print(f"❌ Error parsing CSV: {result['error']}")
        return False
    else:
        print(f"✅ Successfully parsed CSV: {result['filename']}")
        print(f"   Rows: {result['rows']}")
        print(f"   Columns: {result['columns']}")
        print(f"   Column names: {result['column_names']}")
        print(f"   Output format: {result.get('output_format', 'Unknown')}")
        
        # Save markdown content to outputs folder
        if 'markdown_content' in result:
            # Create outputs directory
            output_dir = Path(__file__).parent.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / "test_data_output.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['markdown_content'])
            print(f"💾 Markdown content saved to: {output_file}")
            
            # Show preview
            print(f"\n📄 Markdown Content Preview:")
            print("-" * 40)
            lines = result['markdown_content'].split('\n')
            for i, line in enumerate(lines[:15]):  # Show first 15 lines
                print(line)
            if len(lines) > 15:
                print(f"... (showing first 15 lines of {len(lines)} total)")
            print("-" * 40)
        
        return True

def test_pdf_parsing():
    """Test PDF parsing with PyMuPDF4LLM."""
    print("\n" + "="*50)
    print("Testing PDF Parsing")
    print("="*50)
    
    # Use the existing PDF file
    pdf_file = Path(__file__).parent.parent / "data" /  "TASK_DESCRIPTION.pdf"
    
    if not pdf_file.exists():
        print(f"❌ Test PDF file not found: {pdf_file}")
        return False
    
    print(f"Testing PDF file: {pdf_file}")
    result = parse_pdf(str(pdf_file))
    
    if "error" in result:
        print(f"❌ Error parsing PDF: {result['error']}")
        return False
    else:
        print(f"✅ Successfully parsed PDF")
        print(f"   Status: {result.get('status')}")
        print(f"   Method: {result.get('method')}")
        print(f"   Total pages: {result.get('total_pages')}")
        
    # Show structure of page_chunks
        page_chunks = result.get('page_chunks', [])
        if page_chunks:
            print(f"\n📄 Page chunks structure:")
            print(f"   Number of pages: {len(page_chunks)}")
            print(f"   First page keys: {list(page_chunks[0].keys())}")
            
            # Show metadata of the first page
            first_page = page_chunks[0]
            metadata = first_page.get('metadata', {})
            print(f"   First page metadata keys: {list(metadata.keys())}")
            print(f"   File path: {metadata.get('file_path', 'N/A')}")
            print(f"   Page number: {metadata.get('page_number', 'N/A')}")
            
            # Save full page_chunks data to a JSON file
            output_dir = Path(__file__).parent.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            import json
            json_output_file = output_dir / "TASK_DESCRIPTION_page_chunks_full.json"
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(page_chunks, f, indent=2, ensure_ascii=False)
            print(f"💾 Full page_chunks data saved to: {json_output_file}")
            
            # Save first page content preview
            text_content = first_page.get('text', '')
            preview_file = output_dir / "TASK_DESCRIPTION_page_1_preview.md"
            with open(preview_file, 'w', encoding='utf-8') as f:
                f.write(f"# Page {metadata.get('page_number', 1)} - {metadata.get('file_path', 'unknown.pdf')}\n\n")
                f.write(text_content)
            print(f"💾 First page content saved to: {preview_file}")
            
            # Display first page content preview
            print(f"\n📄 First page content preview:")
            print("-" * 40)
            lines = text_content.split('\n')
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print(line)
            if len(lines) > 10:
                print(f"... (showing first 10 lines of {len(lines)} total)")
            print("-" * 40)
        
        return True

def test_environment():
    """Test environment setup."""
    print("\n" + "="*50)
    print("Environment Check")
    print("="*50)
    
    # Check Python packages
    try:
        import pandas as pd
        print(f"✅ pandas version: {pd.__version__}")
    except ImportError:
        print("❌ pandas not installed")
        return False
    
    try:
        import pymupdf4llm
        print(f"✅ pymupdf4llm available")
    except ImportError:
        print("❌ pymupdf4llm not installed")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Document Parser Test Suite")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test CSV parsing
    csv_ok = test_csv_parsing()
    
    # Test PDF parsing
    pdf_ok = test_pdf_parsing()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Environment: {'✅' if env_ok else '❌'}")
    print(f"CSV Parsing: {'✅' if csv_ok else '❌'}")
    print(f"PDF Parsing: {'✅' if pdf_ok else '❌'}")
    
    if csv_ok:
        print("\n🎉 CSV parsing is working! You can now test with your own CSV files.")
    else:
        print("\n❌ CSV parsing failed. Check the error messages above.")
    
    if pdf_ok:
        print("\n🎉 PDF parsing is working! PyMuPDF4LLM integration successful.")
    else:
        print("\n❌ PDF parsing failed. Check the error messages above.")

if __name__ == "__main__":
    main()