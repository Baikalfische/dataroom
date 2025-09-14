"""
Document chunking utilities for CSV files.
Handles splitting CSV documents into manageable chunks with metadata.
"""

from typing import List, Dict, Any
from pathlib import Path

class DocumentChunker:
    """Document chunker supporting CSV document segmentation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunker.

        Args:
            chunk_size: chunk size in characters
            chunk_overlap: overlap size in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_csv_markdown(self, 
                          markdown_content: str, 
                          document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk CSV markdown content.

        Args:
            markdown_content: Markdown converted from CSV
            document_metadata: document level metadata

        Returns:
            List of chunk metadata dicts
        """
        chunks = []
        # Split by rows
        chunks = self._split_csv_by_rows(markdown_content, document_metadata)
        print(f"📊 CSV chunking complete: {len(chunks)} chunks")
        return chunks

    def _split_csv_by_rows(self, 
                          markdown_content: str, 
                          document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split CSV markdown by table rows.

        Args:
            markdown_content: CSV markdown content
            document_metadata: document metadata

        Returns:
            List of chunk dicts
        """
        chunks = []
        lines = markdown_content.split('\n')
        # Locate table lines
        table_lines = []
        header_line = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and '|' in line[1:]:
                if header_line is None:
                    header_line = i
                table_lines.append((i, line))
        
        # Iterate rows as chunks
        for i, (line_num, line) in enumerate(table_lines):
            if i == 0:  # skip header
                continue
            chunk = {
                "chunk_id": f"{document_metadata['document_id']}_row_{i}",
                "document_id": document_metadata["document_id"],
                "content": line.strip(),
                "metadata": {
                    "file_type": "csv",
                    "filename": document_metadata.get("filename", "unknown.csv"),
                    "row": i,
                    "chunk_type": "row",
                    "total_rows": len(table_lines) - 1,  # minus header
                    "chunk_size": len(line)
                }
            }
            chunks.append(chunk)
            
        return chunks
    
    
    # TODO: PDF chunking functionality - commented out pending page_chunks integration
    # def chunk_pdf_page_chunks(self, page_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     处理PDF的page_chunks数据，进行二次分块（如果需要）
    #     
    #     Args:
    #         page_chunks: PyMuPDF4LLM返回的page_chunks数据
    #         
    #     Returns:
    #         处理后的分块列表
    #     """
    #     chunks = []
    #     
    #     for page_chunk in page_chunks:
    #         # 直接使用页面内容作为chunk，或者进一步分割
    #         chunk = {
    #             "chunk_id": f"{page_chunk['metadata']['file_path']}_page_{page_chunk['metadata']['page_number']}",
    #             "content": page_chunk["text"],
    #             "metadata": page_chunk["metadata"]
    #         }
    #         chunks.append(chunk)
    #     
    #     return chunks
