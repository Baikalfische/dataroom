#!/usr/bin/env python3
"""
Test script for vector database functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataroom.rag.build_database import VectorDatabaseManager

def test_database_manager():
    """测试数据库管理器"""
    print("🧪 Testing VectorDatabaseManager...")
    
    try:
        # 创建数据库管理器
        db_manager = VectorDatabaseManager()
        
        # 测试CSV文档添加
        csv_file = "test_data.csv"
        if os.path.exists(csv_file):
            print(f"📊 Adding CSV document: {csv_file}")
            result = db_manager.add_csv_document(csv_file)
            if "error" not in result:
                print(f"✅ CSV document added: {result['chunks_added']} chunks")
            else:
                print(f"❌ CSV document failed: {result['error']}")
        else:
            print(f"⚠️  CSV file not found: {csv_file}")
        
        # 获取统计信息
        stats = db_manager.get_collection_stats()
        print(f"📊 Database stats:")
        print(f"   PDF chunks: {stats['pdf_documents']['total_chunks']}")
        print(f"   CSV chunks: {stats['csv_documents']['total_chunks']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        
        # 测试搜索
        if stats["csv_documents"]["total_chunks"] > 0:
            print("\n🔍 Testing CSV search...")
            search_results = db_manager.search_csv_documents("张三")
            if "error" not in search_results:
                results = search_results.get("results", {})
                documents = results.get("documents", [[]])[0]
                print(f"   Found {len(documents)} results")
                if documents:
                    print(f"   First result: {documents[0][:100]}...")
            else:
                print(f"   Search failed: {search_results['error']}")
        
        print("\n🎉 Database manager test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_manager()