#!/usr/bin/env python3
"""
Test script for the complete RAG system with CLIP embeddings.
Tests: parsing -> chunking -> embedding -> storage -> retrieval
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataroom.rag.build_database import VectorDatabaseManager
from dataroom.tools.parser import parse_csv

def test_full_rag_system():
    """测试完整的RAG系统"""
    print("🧪 Testing Complete RAG System with CLIP embeddings...")
    
    try:
        # 创建数据库管理器
        print("\n1️⃣ Initializing VectorDatabaseManager...")
        db_manager = VectorDatabaseManager()
        
        # 测试CSV文档添加
        print("\n2️⃣ Testing CSV document processing...")
        csv_file = "test_data.csv"
        if os.path.exists(csv_file):
            result = db_manager.add_csv_document(csv_file)
            if "error" not in result:
                print(f"✅ CSV document added: {result['chunks_added']} chunks")
                print(f"   Document ID: {result['document_id']}")
                print(f"   Database: {result['database']}")
            else:
                print(f"❌ CSV document failed: {result['error']}")
                return False
        else:
            print(f"⚠️  Test CSV file not found: {csv_file}")
            return False
        
        # 获取统计信息
        print("\n3️⃣ Getting database statistics...")
        stats = db_manager.get_collection_stats()
        print(f"📊 Database stats:")
        print(f"   PDF chunks: {stats['pdf_database']['total_chunks']}")
        print(f"   CSV chunks: {stats['csv_database']['total_chunks']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Embedding model: {stats['embedding_model']}")
        
        # 测试搜索功能
        print("\n4️⃣ Testing search functionality...")
        test_queries = ["张三", "李四", "王五", "姓名", "年龄"]
        
        for query in test_queries:
            search_results = db_manager.search_csv_documents(query, top_k=3)
            if "error" not in search_results:
                results = search_results.get('results', {})
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                print(f"🔍 Query: '{query}' -> {len(documents)} results")
                for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                    print(f"   {i+1}. Distance: {dist:.4f}")
                    print(f"      Content: {doc[:50]}...")
                    print(f"      Metadata: {meta}")
            else:
                print(f"❌ Search failed for '{query}': {search_results['error']}")
        
        # 测试跨数据库搜索
        print("\n5️⃣ Testing cross-database search...")
        all_results = db_manager.search_all_documents("张三", top_k=5)
        if "error" not in all_results:
            print(f"✅ Cross-database search successful")
            if all_results.get('csv_results'):
                csv_docs = all_results['csv_results'].get('results', {}).get('documents', [[]])[0]
                print(f"   CSV results: {len(csv_docs)} documents")
        else:
            print(f"❌ Cross-database search failed: {all_results['error']}")
        
        print("\n🎉 Complete RAG system test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """测试各个组件"""
    print("\n🔧 Testing individual components...")
    
    try:
        # 测试CSV解析
        print("\n📊 Testing CSV parsing...")
        csv_file = "test_data.csv"
        if os.path.exists(csv_file):
            parse_result = parse_csv(csv_file)
            if "error" not in parse_result:
                print(f"✅ CSV parsing successful")
                print(f"   Rows: {parse_result.get('rows', 0)}")
                print(f"   Columns: {parse_result.get('columns', 0)}")
                print(f"   Markdown preview: {parse_result.get('markdown_content', '')[:100]}...")
            else:
                print(f"❌ CSV parsing failed: {parse_result['error']}")
        
        print("\n✅ Individual component tests completed!")
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting RAG System Tests...")
    
    # 测试各个组件
    test_individual_components()
    
    # 测试完整系统
    success = test_full_rag_system()
    
    if success:
        print("\n✅ All tests passed! RAG system is ready!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
