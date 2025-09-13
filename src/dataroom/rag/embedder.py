"""
Text Embedding model using CLIP for document vectorization.
Uses CLIP text encoder for text embeddings only.
"""

import os
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Optional
from dotenv import load_dotenv

from .config import config

# Load environment variables
load_dotenv()

class DocumentEmbedder:
    """
    文本文档嵌入器，使用CLIP模型的文本编码器
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化CLIP文本嵌入器
        
        Args:
            model_name: CLIP模型名称，None则使用配置文件中的设置
            device: 运行设备，None则使用配置文件中的设置
        """
        # 从配置文件获取设置
        embed_config = config.get_embedding_config()
        self.model_name = model_name or embed_config.get("name", "openai/clip-vit-base-patch32")
        
        # 设备设置
        if device:
            self.device = device
        elif embed_config.get("device") == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = embed_config.get("device", "cpu")
        
        print(f"🚀 Initializing CLIP Text Embedder on device: {self.device}")
        
        # 设置模型缓存目录
        cache_dir = './model-weights/huggingface'
        clip_model_path = os.path.join(cache_dir, 'clip')
        clip_processor_path = os.path.join(cache_dir, 'clip_processor')
        
        # 创建缓存目录
        os.makedirs(clip_model_path, exist_ok=True)
        os.makedirs(clip_processor_path, exist_ok=True)
        
        # 加载CLIP模型和处理器
        self.model = CLIPModel.from_pretrained(
            self.model_name, 
            cache_dir=clip_model_path
        ).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name, 
            cache_dir=clip_processor_path
        )
        
        print(f"✅ CLIP model loaded successfully from {self.model_name}")
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        嵌入方法，处理单个文本或多个文本
        
        Args:
            texts: 单个文本字符串或文本列表
            
        Returns:
            单个文本返回嵌入向量，多个文本返回嵌入向量列表
        """
        # 统一处理：单个文本转为列表
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        if not texts:
            return [] if not return_single else []
        
        try:
            with torch.no_grad():
                text_inputs = self.processor(
                    text=texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                
                text_emb = self.model.get_text_features(**text_inputs)
                # 归一化
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                
                # 转换为列表格式
                embeddings = text_emb.cpu().numpy().tolist()
                
                # 如果输入是单个文本，返回单个向量
                if return_single:
                    return embeddings[0]
                else:
                    return embeddings
                
        except Exception as e:
            print(f"❌ Error embedding texts: {str(e)}")
            raise