"""
RAG configuration loader from YAML file.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class RAGConfig:
    """
    RAG系统配置类，从YAML文件读取配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，None则使用默认路径
        """
        if config_path is None:
            config_path = Path(__file__).parent / "rag_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 创建必要的目录
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ RAG config loaded from: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {}
    
    def _create_directories(self):
        """创建必要的目录"""
        base_dir = Path(self.config.get('base_dir', './'))
        
        # 创建数据库目录
        pdf_db_path = base_dir / self.config.get('vector_store_config', {}).get('pdf_database', {}).get('persist_directory', './chroma_db/pdf_db')
        csv_db_path = base_dir / self.config.get('vector_store_config', {}).get('csv_database', {}).get('persist_directory', './chroma_db/csv_db')
        
        pdf_db_path.mkdir(parents=True, exist_ok=True)
        csv_db_path.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        log_file = self.config.get('log_config', {}).get('file', 'logs/rag.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created directories: {pdf_db_path}, {csv_db_path}, {log_dir}")
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入配置"""
        return self.config.get('model_config', {}).get('embedding_model', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get('model_config', {}).get('llm_model', {})
    
    def get_prompt_template(self) -> Dict[str, Any]:
        """获取提示词模板"""
        return self.config.get('model_config', {}).get('prompt_template', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """获取向量存储配置"""
        return self.config.get('vector_store_config', {})
    
    def get_text_processing_config(self) -> Dict[str, Any]:
        """获取文本处理配置"""
        return self.config.get('text_processing_config', {})
    
    def get_mineru_config(self) -> Dict[str, Any]:
        """获取MinerU配置"""
        mineru_config = self.config.get('mineru_config', {})
        # 从环境变量读取API密钥
        if not mineru_config.get('api_key'):
            mineru_config['api_key'] = os.getenv('MINERU_API_KEY', '')
        return mineru_config
    
    def get_log_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('log_config', {})
    
    def print_config(self):
        """打印当前配置"""
        print("📋 RAG Configuration:")
        
        # 嵌入配置
        embed_config = self.get_embedding_config()
        print(f"   🧠 Embedding Model: {embed_config.get('name', 'N/A')}")
        print(f"   💻 Device: {embed_config.get('device', 'N/A')}")
        
        # 向量存储配置
        vector_config = self.get_vector_store_config()
        print(f"   🗄️  Database Type: {vector_config.get('database_type', 'N/A')}")
        
        pdf_db = vector_config.get('pdf_database', {})
        csv_db = vector_config.get('csv_database', {})
        print(f"   📁 PDF Database: {pdf_db.get('persist_directory', 'N/A')}")
        print(f"   📁 CSV Database: {csv_db.get('persist_directory', 'N/A')}")
        
        # 文本处理配置
        text_config = self.get_text_processing_config()
        print(f"   📄 Chunk Size: {text_config.get('chunk_size', 'N/A')}")
        print(f"   🔍 PDF Page Chunk: {text_config.get('pdf_page_chunk', 'N/A')}")
        print(f"   📊 CSV Row Chunk: {text_config.get('csv_row_chunk', 'N/A')}")

# 全局配置实例
config = RAGConfig()
