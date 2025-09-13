#!/usr/bin/env python3
"""
Test script for document parser functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataroom.tools.parser import parse_pdf, parse_csv

def create_test_csv():
    """Create a test CSV file for testing."""
    test_csv_path = Path(__file__).parent.parent / "test_data.csv"
    
    if not test_csv_path.exists():
        print("Creating test CSV file...")
        csv_content = """name,age,city,salary
张三,25,北京,50000
李四,30,上海,60000
王五,28,广州,55000
赵六,35,深圳,70000
钱七,22,杭州,45000"""
        
        test_csv_path.write_text(csv_content, encoding='utf-8')
        print(f"Test CSV file created: {test_csv_path}")
    
    return str(test_csv_path)

def test_csv_parsing():
    """Test CSV parsing with pandas."""
    print("\n" + "="*50)
    print("Testing CSV Parsing")
    print("="*50)
    
    # Create test CSV file
    csv_file = create_test_csv()
    
    print(f"Testing CSV file: {csv_file}")
    result = parse_csv(csv_file)
    
    if "error" in result:
        print(f"❌ Error parsing CSV: {result['error']}")
        return False
    else:
        print(f"✅ Successfully parsed CSV: {result['file_name']}")
        print(f"   Rows: {result['rows']}")
        print(f"   Columns: {result['columns']}")
        print(f"   Column names: {result['column_names']}")
        print(f"   Output format: {result.get('output_format', 'Unknown')}")
        
        # Show markdown content preview
        if 'markdown_content' in result:
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
    """Test PDF parsing with MinerU API."""
    print("\n" + "="*50)
    print("Testing PDF Parsing")
    print("="*50)
    
    # Check if API key is set
    if not os.getenv("MINERU_API_KEY"):
        print("⚠️  Warning: MINERU_API_KEY not found in environment variables")
        print("   Please set your MinerU API key in the .env file")
        return False
    
    # Use a real test PDF URL
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print(f"Testing PDF URL: {pdf_url}")
    print("This is a W3C test PDF file for accessibility testing")
    
    result = parse_pdf(pdf_url)
    
    if "error" in result:
        print(f"❌ Error parsing PDF: {result['error']}")
        return False
    else:
        print(f"✅ Successfully parsed PDF")
        print(f"   Status: {result.get('status')}")
        print(f"   Task ID: {result.get('task_id')}")
        print(f"   Data ID: {result.get('data_id')}")
        
        if 'download_url' in result:
            print(f"   Download URL: {result['download_url']}")
        
        if 'markdown_content' in result:
            print(f"\n📄 Markdown Content Preview:")
            print("-" * 40)
            lines = result['markdown_content'].split('\n')
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
        import requests
        print(f"✅ requests available")
    except ImportError:
        print("❌ requests not installed")
        return False
    
    # Check environment variables
    mineru_key = os.getenv("MINERU_API_KEY")
    if mineru_key:
        print(f"✅ MINERU_API_KEY: {'*' * (len(mineru_key) - 4) + mineru_key[-4:]}")
    else:
        print("⚠️  MINERU_API_KEY not set")
    
    return True

def main():
    """Run all tests."""
    print("Document Parser Test Suite")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test CSV parsing (should work regardless of API keys)
    csv_ok = test_csv_parsing()
    
    # Test PDF parsing (requires API key)
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
        print("\n🎉 PDF parsing is working! MinerU API integration successful.")
    elif not os.getenv("MINERU_API_KEY"):
        print("\n📝 To test PDF parsing:")
        print("   1. Set MINERU_API_KEY in your .env file")
        print("   2. Run the test again")
    else:
        print("\n❌ PDF parsing failed. Check the error messages above.")

if __name__ == "__main__":
    main()