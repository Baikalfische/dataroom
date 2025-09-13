"""
Data loading and preprocessing module for RAG.
Supports chunking and multimodal RAG for medical case data.
"""

import pandas as pd
from typing import List, Dict, Any, Generator
from pathlib import Path
import json
import os
from PIL import Image
from tqdm import tqdm
from .chunks import get_chunks


class DataLoader:
    def __init__(self, data_path, image_base_path, level="chunk"):
        """
        Initializes the DataLoader.
        Args:
            data_path (str): Path to the JSON file containing the dataset.
            image_base_path (str): The base directory where the case images are stored.
            level (str): "case" or "chunk" - 决定加载的级别
        """
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.level = level
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def load_data(self):
        """
        A generator that loads data and yields documents with metadata and image objects.
        
        Yields:
            tuple: A tuple containing (document_text, metadata_dict, image_object).
        """
        print(f"Loading data from {len(self.data)} cases at {self.level} level...")
        
        if self.level == "case":
            yield from self._load_case_level_data()
        else:
            yield from self._load_chunk_level_data()

    def _load_case_level_data(self):
        """加载Case级别的数据 - 包含图片信息"""
        for case_id, case_data in tqdm(self.data.items(), desc="Processing cases"):
            # 构建case级别的完整文档
            document = f"Title: {case_data.get('title', '')}\n" \
                       f"Diagnosis: {case_data.get('diagnosis', '')}\n" \
                       f"History: {case_data.get('history', '')}\n" \
                       f"Image Finding: {case_data.get('image_finding', '')}\n" \
                       f"Discussion: {case_data.get('discussion', '')}"

            metadata = {
                "case_id": case_id,
                "title": case_data.get('title', ''),
                "diagnosis": case_data.get('diagnosis', ''),
                "section": case_data.get('section', ''),
                "age": case_data.get('age', ''),
                "gender": case_data.get('gender', ''),
                "link": case_data.get('link', ''),
                "level": "case"
            }
            
            # 获取case的第一张图片
            image = None
            image_path = None
            if case_data.get('figures'):
                for figure in case_data['figures']:
                    for subfigure in figure.get('subfigures', []):
                        file_name = subfigure['number'].lower().replace(" ", "_") + ".jpg"
                        potential_path = os.path.join(self.image_base_path, case_id, file_name)
                        if os.path.exists(potential_path):
                            try:
                                image = Image.open(potential_path).convert("RGB")
                                image_path = potential_path
                                metadata["image_path"] = image_path
                                break
                            except Exception as e:
                                print(f"无法加载图片 {potential_path}: {e}")
                    if image:
                        break
            
            yield document, metadata, image

    def _load_chunk_level_data(self):
        """加载Chunk级别的数据 - 仅文本信息，不包含图片"""
        for case_id, case_data in tqdm(self.data.items(), desc="Processing cases"):
            # 构建case级别的完整文档
            document = f"Title: {case_data.get('title', '')}\n" \
                       f"Diagnosis: {case_data.get('diagnosis', '')}\n" \
                       f"History: {case_data.get('history', '')}\n" \
                       f"Image Finding: {case_data.get('image_finding', '')}\n" \
                       f"Discussion: {case_data.get('discussion', '')}"

            metadata = {
                "case_id": case_id,
                "title": case_data.get('title', ''),
                "link": case_data.get('link', ''),
                "section": case_data.get('section', ''),
                "age": case_data.get('age', ''),
                "gender": case_data.get('gender', ''),
                "level": "chunk"
            }
            
            # 应用分块
            chunks = get_chunks(text=document, metadata=metadata)
            
            # Yield每个chunk，不包含图片
            for chunk in chunks:
                yield chunk.page_content, chunk.metadata, None  # image=None 