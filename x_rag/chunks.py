"""
文本分块模块，用于将英文医学文本分割成适合向量存储的小块。
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

def get_chunks(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    将文本分割成小块
    
    Args:
        text: 要分割的文本内容
        metadata: 文本的元数据，默认为空字典
        yaml_path: 配置文件路径，如果为None则使用默认路径
        chunk_size: 块大小，如果为None则从配置文件读取
        chunk_overlap: 块重叠大小，如果为None则从配置文件读取
        separators: 自定义分隔符列表，为None时使用英文医学文本推荐分隔符
    
    Returns:
        分割后的Document对象列表
    """
    # 设置默认元数据
    metadata = metadata or {}
    
    # 加载配置文件
    if yaml_path is None:
        yaml_path = str(Path(__file__).parent / "rag_config.yaml")
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 从配置文件获取参数，如果没有提供的话
        text_config = config.get("text_processing_config", {})
        if chunk_size is None:
            chunk_size = text_config.get("chunk_size", 800)  # 医学文本推荐800字符
        if chunk_overlap is None:
            chunk_overlap = text_config.get("chunk_overlap", 100)  # 推荐100字符重叠
    except FileNotFoundError:
        print(f"Warning: {yaml_path} not found, using default chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # 使用英文医学文本推荐分隔符
    if separators is None:
        separators = ["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""]
    
    # 创建分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )
    
    # 创建文档对象
    doc = Document(page_content=text, metadata=metadata)
    
    # 分割文本
    chunks = text_splitter.split_documents([doc])
    return chunks

if __name__ == "__main__":
    # 简单的测试代码
    test_text = """
    Title: Pneumonia Case Study
    
    Diagnosis: Community-acquired pneumonia with right lower lobe consolidation.
    
    History: A 45-year-old male patient presented with fever, cough, and shortness of breath for 3 days.
    
    Image Finding: Chest X-ray shows right lower lobe consolidation with air bronchograms.
    
    Discussion: The patient was treated with antibiotics and showed significant improvement within 48 hours.
    """
    
    print("Testing chunking function...")
    print(f"Original text length: {len(test_text)} characters")
    
    # 进行分块
    chunks = get_chunks(
        text=test_text,
        metadata={"source": "test", "type": "medical_case"},
        chunk_size=200,
        chunk_overlap=50
    )
    
    print(f"Chunking result: Generated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: Length {len(chunk.page_content)} characters")
        print(f"  Content: {chunk.page_content}")
        print(f"  Metadata: {chunk.metadata}")
        print()