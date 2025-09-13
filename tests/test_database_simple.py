#!/usr/bin/env python3
"""
Simple test for database structure without OpenAI API.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb
from chromadb.config import Settings

def test_chroma_setup():
    """测试ChromaDB设置"""
    print("🧪 Testing ChromaDB setup with separate databases...")
    
    try:
        # 创建基础目录
        base_dir = "./chroma_db"
        pdf_db_path = os.path.join(base_dir, "pdf_db")
        csv_db_path = os.path.join(base_dir, "csv_db")
        
        Path(pdf_db_path).mkdir(parents=True, exist_ok=True)
        Path(csv_db_path).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created directories:")
        print(f"   PDF DB: {pdf_db_path}")
        print(f"   CSV DB: {csv_db_path}")
        
        # 创建PDF数据库客户端
        pdf_client = chromadb.PersistentClient(
            path=pdf_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 创建CSV数据库客户端
        csv_client = chromadb.PersistentClient(
            path=csv_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 创建集合
        pdf_collection = pdf_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"description": "PDF documents with page-level chunks"}
        )
        
        csv_collection = csv_client.get_or_create_collection(
            name="csv_documents",
            metadata={"description": "CSV documents with row-level chunks"}
        )
        
        print(f"✅ PDF Collection created: {pdf_collection.name}")
        print(f"✅ CSV Collection created: {csv_collection.name}")
        
        # 测试添加一些模拟数据
        test_chunks = [
            {
                "id": "test_pdf_chunk_1",
                "document": "This is a test PDF page content.",
                "metadata": {
                    "file_type": "pdf",
                    "filename": "test.pdf",
                    "page": 1,
                    "chunk_type": "page"
                }
            },
            {
                "id": "test_csv_chunk_1", 
                "document": "张三, 25, 北京, 50000",
                "metadata": {
                    "file_type": "csv",
                    "filename": "test.csv",
                    "row": 1,
                    "chunk_type": "row"
                }
            }
        ]
        
        # 添加到PDF集合
        pdf_collection.add(
            ids=[test_chunks[0]["id"]],
            documents=[test_chunks[0]["document"]],
            metadatas=[test_chunks[0]["metadata"]]
        )
        
        # 添加到CSV集合
        csv_collection.add(
            ids=[test_chunks[1]["id"]],
            documents=[test_chunks[1]["document"]],
            metadatas=[test_chunks[1]["metadata"]]
        )
        
        print("✅ Test data added to both collections")
        
        # 获取统计信息
        pdf_count = pdf_collection.count()
        csv_count = csv_collection.count()
        
        print(f"📊 PDF database ({pdf_db_path}): {pdf_count} chunks")
        print(f"📊 CSV database ({csv_db_path}): {csv_count} chunks")
        
        # 测试搜索
        print("\n🔍 Testing search...")
        
        pdf_results = pdf_collection.query(
            query_texts=["test PDF content"],
            n_results=1
        )
        
        csv_results = csv_collection.query(
            query_texts=["张三"],
            n_results=1
        )
        
        print(f"   PDF search results: {len(pdf_results['documents'][0])} found")
        print(f"   CSV search results: {len(csv_results['documents'][0])} found")
        
        if pdf_results['documents'][0]:
            print(f"   PDF result: {pdf_results['documents'][0][0]}")
        
        if csv_results['documents'][0]:
            print(f"   CSV result: {csv_results['documents'][0][0]}")
        
        # 清理测试数据
        pdf_client.delete_collection("pdf_documents")
        csv_client.delete_collection("csv_documents")
        
        print("\n🎉 ChromaDB separate databases test completed successfully!")
        print(f"📁 Database structure:")
        print(f"   {base_dir}/")
        print(f"   ├── pdf_db/")
        print(f"   └── csv_db/")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chroma_setup()