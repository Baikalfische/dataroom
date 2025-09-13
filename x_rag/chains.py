import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from .embedder import MultimodalEmbedder

class MultimodalRAGChain:
    """
    A custom chain for performing multimodal RAG.
    This chain handles text and image inputs, retrieves relevant documents from a 
    multimodal ChromaDB, and generates a response using a language model.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the RAG chain components.
        
        Args:
            config_path (str, optional): Path to the YAML config file. 
                                         Defaults to 'rag_config.yaml' in the same directory.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "rag_config.yaml")
        
        print("--- Initializing Multimodal RAG Chain ---")
        
        # 1. Load Configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        llm_cfg = config["model_config"]["llm_model"]
        vs_cfg = config["vector_store_config"]
        prompt_cfg = config["model_config"]["prompt_template"]

        # 2. Initialize Language Model
        # 意图识别用小模型
        self.intent_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.1,  # 低温度确保一致性
            max_output_tokens=100,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.llm = ChatGoogleGenerativeAI(
            model=llm_cfg["name"],
            temperature=llm_cfg["temperature"],
            max_output_tokens=llm_cfg["max_length"],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(f"LLM initialized: {llm_cfg['name']}, intent_llm initialized: gemini-2.0-flash-lite")

        # 3. Initialize Multimodal Embedder
        self.embedder = MultimodalEmbedder()
        self.embedding_function = self.embedder.get_embedding_function()
        print("Multimodal embedder initialized.")

        # 4. Initialize ChromaDB Clients
        case_cfg = vs_cfg["case_level"]
        chunk_cfg = vs_cfg["chunk_level"]
        
        # Case级别数据库
        case_db_path = case_cfg["persist_directory"]
        self.case_client = chromadb.PersistentClient(path=case_db_path)
        self.case_collection = self.case_client.get_collection(name=case_cfg["collection_name"])
        print(f"Case-level database connected: {case_db_path}")
        
        # Chunk级别数据库
        chunk_db_path = chunk_cfg["persist_directory"]
        self.chunk_client = chromadb.PersistentClient(path=chunk_db_path)
        self.chunk_collection = self.chunk_client.get_collection(name=chunk_cfg["collection_name"])
        print(f"Chunk-level database connected: {chunk_db_path}")
        
        # 5. Initialize settings
        self.case_k = case_cfg["search_kwargs"]["k"]
        self.chunk_k = chunk_cfg["search_kwargs"]["k"]
        self.prompt = PromptTemplate(
            template=prompt_cfg["template"],
            input_variables=prompt_cfg["input_variables"]
        )
        print("Two-Stage RAG chain components are ready.")

        # 6. 意图识别prompt
        self.intent_prompt = PromptTemplate(
            template="""分析用户查询意图，返回检索策略。

        文本: {query_text}
        图片: {has_image}

        规则:
        - 文本是通用指令("检索图片信息"等)且有图片 → image_only
        - 文本是具体医学问题且有图片 → mixed_query  
        - 只有文本 → text_only
        - 只有图片 → image_only

        返回: image_only / text_only / mixed_query""",
                    input_variables=["query_text", "has_image"]
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the two-stage RAG chain.
        """
        query_text = inputs.get("input")
        query_image = inputs.get("image")

        if not query_text and not query_image:
            raise ValueError("Query must contain at least 'input' text or an 'image'.")

        # LLM意图识别
        intent_chain = self.intent_prompt | self.intent_llm
        response = intent_chain.invoke({
            "query_text": query_text if query_text else "",
            "has_image": "是" if query_image else "否"
        })
        
        intent = response.content.strip().lower()
        print(f"🎯 意图识别结果: {intent}")
        
        # 根据意图调整查询
        if intent == "image_only":
            print(f"🖼️ 意图识别: 纯图片检索")
            query_text = None
        elif intent == "text_only":
            print(f"📝 意图识别: 纯文本检索")
            query_image = None
        elif intent == "mixed_query":
            print(f"🔄 意图识别: 混合查询")
        else:
            print(f"⚠️ 未知意图，使用混合查询")
            intent = "mixed_query"
            
        # Stage 1: Case-level检索
        print("🔍 Stage 1: Case-level检索")
        case_results = self._case_level_retrieval(query_text, query_image)
        
        # Stage 2: Chunk-level检索
        print("🔍 Stage 2: Chunk-level检索")
        chunk_results = self._chunk_level_retrieval(query_text, query_image, case_results)
        
        # 生成答案
        context_docs = chunk_results['documents']

        print(f"Context数量: {len(context_docs)}")
        print(f"Context内容: {context_docs[:2]}...")  # 显示前两个文档

        context = "\n\n---\n\n".join(context_docs)
        print(f"合并后Context长度: {len(context)}")

        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "input": query_text if query_text else ""
        })

        # 提取图片路径
        image_paths = []
        for metadata_list in case_results['metadatas']:
            for metadata in metadata_list:
                if metadata and 'image_path' in metadata and metadata['image_path']:
                    image_paths.append(metadata['image_path'])

        # 调试信息
        try:
            self._print_debug_info_two_stage(query_text, query_image, case_results, chunk_results)
        except Exception as e:
            print(f"⚠️ 调试信息打印失败: {e}")
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈
        
        return {
            "context": context,
            "answer": response.content,
            "image_paths": image_paths,
            "case_results": case_results,
            "chunk_results": chunk_results
        }

    def _case_level_retrieval(self, query_text, query_image):
        """Case级别检索"""
        # 生成查询向量
        query_embedding = self.embedding_function(texts=[query_text], images=[query_image])
        
        # 在case级别数据库中检索
        case_results = self.case_collection.query(
            query_embeddings=query_embedding,
            n_results=self.case_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return case_results

    def _chunk_level_retrieval(self, query_text, query_image, case_results):
        """Chunk级别检索"""
        # 只获取相似度最高的case的ID
        if not case_results['metadatas'][0]:
            print("⚠️ 没有找到相关case，返回空结果")
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        # 只取第一个（相似度最高的）case
        top_case_metadata = case_results['metadatas'][0][0]
        top_case_id = top_case_metadata.get('case_id')
        
        if not top_case_id:
            print("⚠️ 没有找到相关case，返回空结果")
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        print(f"🎯 在相似度最高的case中检索: {top_case_id}")

        # 如果query_text为None，使用默认文本
        if query_text is None:
            query_text = "image analysis"
        
        # 生成查询向量（chunk级别不使用图片）
        query_embedding = self.embedding_function(texts=[query_text], images=[None])
        
        # 只在相似度最高的case的chunks中检索
        chunk_results = self.chunk_collection.query(
            query_embeddings=query_embedding,
            n_results=self.chunk_k,
            include=['documents', 'metadatas', 'distances'],
            where={"case_id": top_case_id}  # 只检索一个case
        )
        
        # 在返回前展平结果（如果需要）
        if isinstance(chunk_results['documents'], list) and len(chunk_results['documents']) > 0:
            if isinstance(chunk_results['documents'][0], list):
                # 嵌套结构，展平
                chunk_results['documents'] = [doc for doc_list in chunk_results['documents'] for doc in doc_list]
                chunk_results['metadatas'] = [meta for meta_list in chunk_results['metadatas'] for meta in meta_list]
                chunk_results['distances'] = [dist for dist_list in chunk_results['distances'] for dist in dist_list]
        
        return chunk_results

    def _print_debug_info_two_stage(self, query_text, query_image, case_results, chunk_results):
        """显示两阶段检索的调试信息"""
        print(f"\n{'='*50}")
        print(f"🔍 两阶段RAG检索调试信息")
        print(f"{'='*50}")
        
        # 查询信息
        print(f" 文本查询: '{query_text}'")
        print(f"🖼️ 图片查询: {'有' if query_image else '无'}")
        
        # Stage 1: Case级别结果
        print(f"\n--- Stage 1: Case级别检索结果 ---")
        for i, (distance, metadata, document) in enumerate(zip(
            case_results['distances'][0],
            case_results['metadatas'][0],
            case_results['documents'][0]
        )):
            similarity = 1.0 / (1.0 + distance)
            print(f"📸 Case {i+1}:")
            print(f"    📏 距离: {distance:.4f}")
            print(f"    相似度: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"    元数据: {metadata}")
            print(f"    📝 内容预览: {document[:100]}...")
            print()
        
        # Stage 2: Chunk级别结果
        print(f"\n--- Stage 2: Chunk级别检索结果 ---")
        for i, (distance, metadata, document) in enumerate(zip(
            chunk_results['distances'],
            chunk_results['metadatas'],
            chunk_results['documents']
        )):
            similarity = 1.0 / (1.0 + distance)
            print(f"📄 Chunk {i+1}:")
            print(f"    📏 距离: {distance:.4f}")
            print(f"    相似度: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"    元数据: {metadata}")
            print(f"    📝 内容预览: {document[:100]}...")
            print()
        
        # 统计信息
        print(f"📊 检索统计:")
        print(f"   📸 Case级别检索: {len(case_results['documents'][0])}个case")
        print(f"   Chunk级别检索: {len(chunk_results['documents'])}个chunk")
        print(f"{'='*50}")