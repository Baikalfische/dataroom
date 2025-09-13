"""
Build and manage vector databases for PDF and CSV documents.
Implements dual database architecture with CLIP embeddings.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from .embedder import DocumentEmbedder
from .chunks import DocumentChunker
from .config import config
from ..tools.parser import parse_pdf, parse_csv

class VectorDatabaseManager:
    """
    向量数据库管理器，支持PDF和CSV的独立存储
    使用CLIP嵌入模型进行文本向量化
    """
    
    def __init__(self):
        """初始化数据库管理器"""
        # 从配置文件获取设置
        vector_config = config.get_vector_store_config()
        
        # 初始化嵌入器和分块器
        self.embedder = DocumentEmbedder()
        self.chunker = DocumentChunker()
        
        # 初始化PDF数据库客户端
        pdf_db_path = vector_config.get("pdf_database", {}).get("persist_directory", "./chroma_db/pdf_db")
        self.pdf_client = chromadb.PersistentClient(
            path=pdf_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 初始化CSV数据库客户端
        csv_db_path = vector_config.get("csv_database", {}).get("persist_directory", "./chroma_db/csv_db")
        self.csv_client = chromadb.PersistentClient(
            path=csv_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 创建集合
        self.pdf_collection = self.pdf_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"description": "PDF documents with page-level chunks"}
        )
        
        self.csv_collection = self.csv_client.get_or_create_collection(
            name="csv_documents", 
            metadata={"description": "CSV documents with row-level chunks"}
        )
        
        print("✅ VectorDatabaseManager initialized with CLIP embeddings")
        print(f"   📁 PDF Database: {pdf_db_path}")
        print(f"   📁 CSV Database: {csv_db_path}")
        print(f"   🧠 Embedding Model: CLIP (512 dimensions)")
    
    def add_pdf_document(self, 
                        document_url: str, 
                        document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """添加PDF文档到向量数据库"""
        try:
            print(f"📄 Processing PDF document: {document_url}")
            
            # 解析PDF文档
            parse_result = parse_pdf(document_url)
            if "error" in parse_result:
                return {"error": f"Failed to parse PDF: {parse_result['error']}"}
            
            # 生成文档ID
            if not document_id:
                document_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 准备文档元数据
            doc_metadata = {
                "document_id": document_id,
                "filename": os.path.basename(document_url),
                "file_type": "pdf",
                "url": document_url,
                "upload_time": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # 获取Markdown内容
            if "markdown_content" not in parse_result:
                return {"error": "No markdown content found in parse result"}
            
            markdown_content = parse_result["markdown_content"]
            
            # 分块
            chunks = self.chunker.chunk_pdf_markdown(markdown_content, doc_metadata)
            
            if not chunks:
                return {"error": "No chunks generated from PDF"}
            
            # 生成嵌入向量
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedder.embed(texts)
            
            # 将嵌入向量添加到块中
            chunks_with_embeddings = []
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
                chunks_with_embeddings.append(chunk)
            
            # 准备ChromaDB数据
            ids = [chunk["chunk_id"] for chunk in chunks_with_embeddings]
            documents = [chunk["content"] for chunk in chunks_with_embeddings]
            embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
            metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
            
            # 添加到PDF集合
            self.pdf_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            result = {
                "status": "success",
                "document_id": document_id,
                "database": "pdf_db",
                "collection": "pdf_documents",
                "chunks_added": len(chunks),
                "metadata": doc_metadata
            }
            
            print(f"✅ PDF document added to pdf_db: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            return {"error": f"Error adding PDF document: {str(e)}"}
    
    def add_csv_document(self, 
                        file_path: str, 
                        document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """添加CSV文档到向量数据库"""
        try:
            print(f"📊 Processing CSV document: {file_path}")
            
            # 解析CSV文档
            parse_result = parse_csv(file_path)
            if "error" in parse_result:
                return {"error": f"Failed to parse CSV: {parse_result['error']}"}
            
            # 生成文档ID
            if not document_id:
                document_id = f"csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 准备文档元数据
            doc_metadata = {
                "document_id": document_id,
                "filename": os.path.basename(file_path),
                "file_type": "csv",
                "file_path": file_path,
                "upload_time": datetime.now().isoformat(),
                "rows": parse_result.get("rows", 0),
                "columns": parse_result.get("columns", 0),
                **(metadata or {})
            }
            
            # 获取Markdown内容
            if "markdown_content" not in parse_result:
                return {"error": "No markdown content found in parse result"}
            
            markdown_content = parse_result["markdown_content"]
            
            # 分块
            chunks = self.chunker.chunk_csv_markdown(markdown_content, doc_metadata)
            
            if not chunks:
                return {"error": "No chunks generated from CSV"}
            
            # 生成嵌入向量
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedder.embed(texts)
            
            # 将嵌入向量添加到块中
            chunks_with_embeddings = []
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
                chunks_with_embeddings.append(chunk)
            
            # 准备ChromaDB数据
            ids = [chunk["chunk_id"] for chunk in chunks_with_embeddings]
            documents = [chunk["content"] for chunk in chunks_with_embeddings]
            embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
            metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
            
            # 添加到CSV集合
            self.csv_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            result = {
                "status": "success",
                "document_id": document_id,
                "database": "csv_db",
                "collection": "csv_documents",
                "chunks_added": len(chunks),
                "metadata": doc_metadata
            }
            
            print(f"✅ CSV document added to csv_db: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            return {"error": f"Error adding CSV document: {str(e)}"}
    
    def search_pdf_documents(self, 
                           query: str, 
                           top_k: int = 5,
                           filters: Optional[Dict] = None) -> Dict[str, Any]:
        """在PDF文档中搜索"""
        try:
            # 使用CLIP嵌入器生成查询向量
            query_embedding = self.embedder.embed(query)
            
            results = self.pdf_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )
            
            return {
                "status": "success",
                "database": "pdf_db",
                "collection": "pdf_documents",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            return {"error": f"Error searching PDF documents: {str(e)}"}
    
    def search_csv_documents(self, 
                           query: str, 
                           top_k: int = 5,
                           filters: Optional[Dict] = None) -> Dict[str, Any]:
        """在CSV文档中搜索"""
        try:
            # 使用CLIP嵌入器生成查询向量
            query_embedding = self.embedder.embed(query)
            
            results = self.csv_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )
            
            return {
                "status": "success",
                "database": "csv_db",
                "collection": "csv_documents",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            return {"error": f"Error searching CSV documents: {str(e)}"}
    
    def search_all_documents(self, 
                           query: str, 
                           top_k: int = 5,
                           file_type: Optional[str] = None) -> Dict[str, Any]:
        """在所有文档中搜索"""
        results = {
            "status": "success",
            "query": query,
            "pdf_results": None,
            "csv_results": None
        }
        
        if file_type is None or file_type == "pdf":
            pdf_results = self.search_pdf_documents(query, top_k)
            if "error" not in pdf_results:
                results["pdf_results"] = pdf_results
        
        if file_type is None or file_type == "csv":
            csv_results = self.search_csv_documents(query, top_k)
            if "error" not in csv_results:
                results["csv_results"] = csv_results
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        pdf_count = self.pdf_collection.count()
        csv_count = self.csv_collection.count()
        
        vector_config = config.get_vector_store_config()
        
        return {
            "pdf_database": {
                "path": vector_config.get("pdf_database", {}).get("persist_directory", "./chroma_db/pdf_db"),
                "collection_name": "pdf_documents",
                "total_chunks": pdf_count,
                "description": "PDF documents with page-level chunks"
            },
            "csv_database": {
                "path": vector_config.get("csv_database", {}).get("persist_directory", "./chroma_db/csv_db"),
                "collection_name": "csv_documents", 
                "total_chunks": csv_count,
                "description": "CSV documents with row-level chunks"
            },
            "embedding_model": "CLIP (512 dimensions)",
            "total_chunks": pdf_count + csv_count
        }
    
    def reset_databases(self):
        """重置所有数据库"""
        # 重置PDF数据库
        self.pdf_client.delete_collection("pdf_documents")
        self.pdf_collection = self.pdf_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"description": "PDF documents with page-level chunks"}
        )
        
        # 重置CSV数据库
        self.csv_client.delete_collection("csv_documents")
        self.csv_collection = self.csv_client.get_or_create_collection(
            name="csv_documents",
            metadata={"description": "CSV documents with row-level chunks"}
        )
        
        vector_config = config.get_vector_store_config()
        
        print("🔄 Databases reset successfully")
        print(f"   📁 PDF Database: {vector_config.get('pdf_database', {}).get('persist_directory', './chroma_db/pdf_db')}")
        print(f"   📁 CSV Database: {vector_config.get('csv_database', {}).get('persist_directory', './chroma_db/csv_db')}")

def test_database_manager():
    """测试数据库管理器"""
    print("🧪 Testing VectorDatabaseManager with CLIP embeddings...")
    
    try:
        # 创建数据库管理器
        db_manager = VectorDatabaseManager()
        
        # 测试CSV文档添加
        csv_file = "test_data.csv"
        if os.path.exists(csv_file):
            result = db_manager.add_csv_document(csv_file)
            if "error" not in result:
                print(f"✅ CSV document added: {result['chunks_added']} chunks")
            else:
                print(f"❌ CSV document failed: {result['error']}")
        
        # 获取统计信息
        stats = db_manager.get_collection_stats()
        print(f"📊 Database stats: {stats}")
        
        # 测试搜索
        if stats["csv_database"]["total_chunks"] > 0:
            search_results = db_manager.search_csv_documents("张三")
            print(f"🔍 Search test: {len(search_results.get('results', {}).get('documents', [[]])[0])} results")
        
        print("🎉 Database manager test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_manager()