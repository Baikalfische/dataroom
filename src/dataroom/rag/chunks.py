"""
Document chunking utilities for PDF and CSV files.
Handles splitting documents into manageable chunks with metadata.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import json

class DocumentChunker:
    """
    文档分块器，支持PDF和CSV文档的分块处理
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 pdf_page_chunk: bool = True,
                 csv_row_chunk: bool = True):
        """
        初始化分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小（字符数）
            pdf_page_chunk: PDF是否按页面分块
            csv_row_chunk: CSV是否按行分块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_page_chunk = pdf_page_chunk
        self.csv_row_chunk = csv_row_chunk
    
    def chunk_pdf_markdown(self, 
                          markdown_content: str, 
                          document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分块PDF的Markdown内容
        
        Args:
            markdown_content: PDF转换后的Markdown内容
            document_metadata: 文档元数据
            
        Returns:
            分块列表
        """
        chunks = []
        
        if self.pdf_page_chunk:
            # 按页面分块（假设页面之间有分隔符）
            pages = self._split_by_pages(markdown_content)
            
            for page_num, page_content in enumerate(pages, 1):
                if page_content.strip():
                    chunk = {
                        "chunk_id": f"{document_metadata['document_id']}_page_{page_num}",
                        "document_id": document_metadata["document_id"],
                        "content": page_content.strip(),
                        "metadata": {
                            "file_type": "pdf",
                            "filename": document_metadata.get("filename", "unknown.pdf"),
                            "page": page_num,
                            "chunk_type": "page",
                            "total_pages": len(pages),
                            "chunk_size": len(page_content)
                        }
                    }
                    chunks.append(chunk)
        else:
            # 按固定大小分块
            chunks = self._split_by_size(
                markdown_content, 
                document_metadata, 
                "pdf"
            )
        
        print(f"📄 PDF分块完成: {len(chunks)} 个块")
        return chunks
    
    def chunk_csv_markdown(self, 
                          markdown_content: str, 
                          document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分块CSV的Markdown内容
        
        Args:
            markdown_content: CSV转换后的Markdown内容
            document_metadata: 文档元数据
            
        Returns:
            分块列表
        """
        chunks = []
        
        if self.csv_row_chunk:
            # 按行分块
            chunks = self._split_csv_by_rows(markdown_content, document_metadata)
        else:
            # 按固定大小分块
            chunks = self._split_by_size(
                markdown_content, 
                document_metadata, 
                "csv"
            )
        
        print(f"📊 CSV分块完成: {len(chunks)} 个块")
        return chunks
    
    def _split_by_pages(self, markdown_content: str) -> List[str]:
        """
        按页面分割Markdown内容
        
        Args:
            markdown_content: Markdown内容
            
        Returns:
            页面列表
        """
        # 尝试多种页面分隔符
        page_separators = [
            r'\n---\n',  # 标准页面分隔符
            r'\n\n---\n\n',  # 带空行的分隔符
            r'\n# Page \d+\n',  # 页面标题
            r'\n## Page \d+\n',  # 页面子标题
        ]
        
        pages = [markdown_content]
        
        for separator in page_separators:
            new_pages = []
            for page in pages:
                split_pages = re.split(separator, page)
                new_pages.extend(split_pages)
            pages = new_pages
        
        # 如果没有找到页面分隔符，尝试按标题分割
        if len(pages) == 1:
            pages = self._split_by_headers(pages[0])
        
        return pages
    
    def _split_by_headers(self, markdown_content: str) -> List[str]:
        """
        按标题分割Markdown内容
        
        Args:
            markdown_content: Markdown内容
            
        Returns:
            分割后的内容列表
        """
        # 按一级标题分割
        sections = re.split(r'\n# ', markdown_content)
        
        # 重新添加标题
        if len(sections) > 1:
            sections[0] = sections[0]  # 第一部分可能没有标题
            for i in range(1, len(sections)):
                sections[i] = '# ' + sections[i]
        
        return sections
    
    def _split_csv_by_rows(self, 
                          markdown_content: str, 
                          document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        按行分割CSV的Markdown内容
        
        Args:
            markdown_content: CSV的Markdown内容
            document_metadata: 文档元数据
            
        Returns:
            分块列表
        """
        chunks = []
        lines = markdown_content.split('\n')
        
        # 找到表格行
        table_lines = []
        header_line = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and '|' in line[1:]:
                if header_line is None:
                    header_line = i
                table_lines.append((i, line))
        
        if not table_lines:
            # 如果没有找到表格，按普通文本处理
            return self._split_by_size(markdown_content, document_metadata, "csv")
        
        # 按行分块
        for i, (line_num, line) in enumerate(table_lines):
            if i == 0:  # 跳过表头
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
                    "total_rows": len(table_lines) - 1,  # 减去表头
                    "chunk_size": len(line)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_size(self, 
                      content: str, 
                      document_metadata: Dict[str, Any], 
                      file_type: str) -> List[Dict[str, Any]]:
        """
        按固定大小分割内容
        
        Args:
            content: 要分割的内容
            document_metadata: 文档元数据
            file_type: 文件类型
            
        Returns:
            分块列表
        """
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # 尝试在句子边界分割
            if end < len(content):
                # 向后查找句号、换行符等分割点
                for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                    if content[i] in '.\n!?':
                        end = i + 1
                        break
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = {
                    "chunk_id": f"{document_metadata['document_id']}_chunk_{chunk_num}",
                    "document_id": document_metadata["document_id"],
                    "content": chunk_content,
                    "metadata": {
                        "file_type": file_type,
                        "filename": document_metadata.get("filename", f"unknown.{file_type}"),
                        "chunk_num": chunk_num,
                        "chunk_type": "size_based",
                        "chunk_size": len(chunk_content),
                        "start_pos": start,
                        "end_pos": end
                    }
                }
                chunks.append(chunk)
                chunk_num += 1
            
            # 设置下一个块的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            if start < 0:
                start = end
        
        return chunks
    
    def chunk_document(self, 
                      content: str, 
                      file_type: str, 
                      document_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        通用文档分块接口
        
        Args:
            content: 文档内容
            file_type: 文件类型 (pdf/csv)
            document_metadata: 文档元数据
            
        Returns:
            分块列表
        """
        if file_type.lower() == "pdf":
            return self.chunk_pdf_markdown(content, document_metadata)
        elif file_type.lower() == "csv":
            return self.chunk_csv_markdown(content, document_metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        保存分块到文件
        
        Args:
            chunks: 分块列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"💾 分块已保存到: {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载分块
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            分块列表
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"📂 从 {input_path} 加载了 {len(chunks)} 个分块")
        return chunks

def test_chunker():
    """测试分块器功能"""
    print("🧪 Testing DocumentChunker...")
    
    # 创建分块器
    chunker = DocumentChunker()
    
    # 测试PDF分块
    pdf_content = """# Document Title

This is page 1 content.

---

# Page 2

This is page 2 content.

---

# Page 3

This is page 3 content."""
    
    pdf_metadata = {
        "document_id": "test_pdf_001",
        "filename": "test.pdf"
    }
    
    pdf_chunks = chunker.chunk_pdf_markdown(pdf_content, pdf_metadata)
    print(f"✅ PDF分块: {len(pdf_chunks)} 个块")
    
    # 测试CSV分块
    csv_content = """# test_data.csv

**行数**: 5
**列数**: 4

## 完整数据

| name   |   age | city   |   salary |
|:-------|------:|:-------|---------:|
| 张三     |    25 | 北京     |    50000 |
| 李四     |    30 | 上海     |    60000 |
| 王五     |    28 | 广州     |    55000 |"""
    
    csv_metadata = {
        "document_id": "test_csv_001", 
        "filename": "test.csv"
    }
    
    csv_chunks = chunker.chunk_csv_markdown(csv_content, csv_metadata)
    print(f"✅ CSV分块: {len(csv_chunks)} 个块")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_chunker()
