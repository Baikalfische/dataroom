import pymupdf4llm
import pandas as pd
from pathlib import Path
from typing import Dict, Any

def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Parse a local PDF file using PyMuPDF4LLM and return page-level chunks.
    
    Args:
        file_path: Local PDF file path
        
    Returns:
        Dictionary containing page parsing results
    """
    try:
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Validate file extension
        if file_path.suffix.lower() != '.pdf':
            return {"error": f"Not a PDF file: {file_path}"}

        # Use PyMuPDF4LLM with page_chunks enabled
        print(f"🔍 Parsing PDF with PyMuPDF4LLM: {file_path.name}")
        page_chunks = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True
        )

        result = {
            "status": "success",
            "page_chunks": page_chunks,  # Directly return page-level data
            "filename": file_path.name,
            "file_type": "pdf",
            "method": "pymupdf4llm_page_chunks",
            "total_pages": len(page_chunks),
            "file_path": str(file_path)
        }

        print(f"✅ PDF parsing successful: {len(page_chunks)} pages")
        if page_chunks:
            first_page = page_chunks[0]
            print(f" Page structure keys: {list(first_page.keys())}")
            print(f"📄 First page metadata keys: {list(first_page.get('metadata', {}).keys())}")

        return result

    except Exception as e:
        return {"error": f"PDF parsing failed: {str(e)}"}

def parse_csv(file_path: str) -> Dict[str, Any]:
    """
    Parse CSV document using pandas and convert to Markdown format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed content and metadata in Markdown format
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Validate file extension
        if file_path.suffix.lower() != '.csv':
            return {"error": f"Not a CSV file: {file_path}"}

        # Read CSV with pandas
        df = pd.read_csv(file_path)

        # Convert to Markdown
        markdown_content = f"# {file_path.name}\n\n"
        markdown_content += f"**Rows**: {len(df)}\n"
        markdown_content += f"**Columns**: {len(df.columns)}\n\n"
        markdown_content += "## Full Data\n\n"
        markdown_content += df.to_markdown(index=False)

        info = {
            "filename": file_path.name,
            "file_type": "csv",
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "sample_data": df.head(5).to_dict('records'),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "markdown_content": markdown_content,
            "output_format": "Markdown",
            "file_path": str(file_path)
        }

        return info

    except Exception as e:
        return {"error": f"Error parsing CSV: {str(e)}"}