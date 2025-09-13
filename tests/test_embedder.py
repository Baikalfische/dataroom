#!/usr/bin/env python3
"""
Test script for Google Gemini embedder functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataroom.rag.embedder import DocumentEmbedder

def test_embedder():
    """测试Google Gemini嵌入器功能"""
    print("🧪 Testing DocumentEmbedder with Google Gemini...")
    
    # 检查API密钥
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
        print("   Please set your Google API key in the .env file")
        return False
    
    try:
        # 创建嵌入器
        embedder = DocumentEmbedder()
        
        # 测试单个文本嵌入
        test_text = "This is a test document for embedding."
        embedding = embedder.embed_text(test_text)
        print(f"✅ Single text embedding: {len(embedding)} dimensions")
        
        # 测试查询嵌入
        query_text = "What is this document about?"
        query_embedding = embedder.embed_query(query_text)
        print(f"✅ Query embedding: {len(query_embedding)} dimensions")
        
        # 测试批量文本嵌入
        test_texts = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        embeddings = embedder.embed_texts(test_texts)
        print(f"✅ Batch embeddings: {len(embeddings)} vectors")
        
        # 测试相似度计算
        similarity = embedder.calculate_similarity(embeddings[0], embeddings[1])
        print(f"✅ Similarity between first two texts: {similarity:.4f}")
        
        # 测试查询相似度
        query_similarity = embedder.calculate_similarity(query_embedding, embeddings[0])
        print(f"✅ Query similarity with first document: {query_similarity:.4f}")
        
        # 测试最相似搜索
        most_similar = embedder.find_most_similar(query_embedding, embeddings, top_k=2)
        print(f"✅ Most similar search: {len(most_similar)} results")
        for result in most_similar:
            print(f"   Index {result['index']}: similarity {result['similarity']:.4f}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedder()
    if success:
        print("\n✅ Google Gemini embedder is ready to use!")
    else:
        print("\n❌ Please check your Google API key configuration.")
