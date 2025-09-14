"""
Text Embedding model using CLIP for document vectorization.
Uses CLIP text encoder for text embeddings only.
"""

import os
import yaml
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class DocumentEmbedder:
    """Text document embedder using the CLIP text encoder."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize CLIP text embedder.

        Args:
            model_name: CLIP model name; if None use config file
            device: execution device; if None resolve from config
        """
    # Load settings from YAML config
        config_path = Path(__file__).parent / "rag_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        embed_config = config["model_config"]["embedding_model"]
        self.model_name = model_name or embed_config.get("name", "openai/clip-vit-base-patch32")
        
    # Device selection
        if device:
            self.device = device
        elif embed_config.get("device") == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = embed_config.get("device", "cpu")
        
        print(f"🚀 Initializing CLIP Text Embedder on device: {self.device}")
        
    # Model cache directory setup
        cache_dir = './model-weights/huggingface'
        clip_model_path = os.path.join(cache_dir, 'clip')
        clip_processor_path = os.path.join(cache_dir, 'clip_processor')
        
    # Ensure cache directories exist
        os.makedirs(clip_model_path, exist_ok=True)
        os.makedirs(clip_processor_path, exist_ok=True)
        
    # Load CLIP model & processor
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
        """Embed one or many texts.

        Args:
            texts: single string or list of strings

        Returns:
            Single embedding vector or list of vectors
        """
    # Normalize single input to list
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
                # Normalize
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                
                # Convert to list
                embeddings = text_emb.cpu().numpy().tolist()
                
                # Return single vector if original input was single
                if return_single:
                    return embeddings[0]
                else:
                    return embeddings
                
        except Exception as e:
            print(f"❌ Error embedding texts: {str(e)}")
            raise