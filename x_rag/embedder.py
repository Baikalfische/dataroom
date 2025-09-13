"""
Multimodal Embedder using CLIP model.
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List
import os

class MultimodalEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initializes the MultimodalEmbedder using a CLIP model.
        Args:
            model_name (str): The name of the CLIP model to use from Hugging Face.
            device (str): The device to run the model on ('cuda' or 'cpu'). 
                          If None, it will auto-detect CUDA availability.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing Multimodal Embedder on device: {self.device}")
        
        # Setting the cache directory. 
        cache_dir = './model-weights/huggingface'
        clip_model_path = os.path.join(cache_dir, 'clip')
        clip_processor_path = os.path.join(cache_dir, 'clip_processor')

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=clip_processor_path)
        print("CLIP model and processor loaded successfully.")

    def get_embedding_function(self):
        """
        Returns a function that can embed a single document, which may contain text and/or an image.
        This is designed to produce a single, unified vector for a text-image pair by concatenating (拼接) text and image embeddings.
        - If both text and image are provided, returns [text_embeds, image_embeds] concatenated.
        - If only text or only image is provided, returns the corresponding embedding.
        This allows flexible retrieval: text-only, image-only, or multimodal.
        """
        def embed_function(texts=None, images=None):
            text_input = texts[0] if texts and texts[0] is not None else None
            image_input = images[0] if images and images[0] is not None else None
            
            if not text_input and not image_input:
                raise ValueError("At least one of text or image must be provided.")

            dim = 512
            zero_emb = torch.zeros((1, dim), device=self.device)

            with torch.no_grad():
                # Compute text embedding if text is provided
                if text_input:
                    text_inputs = self.processor(text=[text_input], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_emb = self.model.get_text_features(**text_inputs)
                    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                else:
                    text_emb = zero_emb

                # Compute image embedding if image is provided
                if image_input:
                    image_inputs = self.processor(images=[image_input], return_tensors="pt").to(self.device)
                    image_emb = self.model.get_image_features(**image_inputs)
                    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                else:
                    image_emb = zero_emb

                # Always concatenate, so output is always 1024-dim
                final_embedding = torch.cat([text_emb, image_emb], dim=-1)

            return final_embedding.cpu().numpy().flatten().tolist()

        return embed_function
