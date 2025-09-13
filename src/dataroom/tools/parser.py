import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import time
import re
import zipfile
import io
import tempfile

# Load environment variables
load_dotenv()

def generate_data_id(file_path: str) -> str:
    """
    生成符合MinerU API要求的data_id
    
    Args:
        file_path: 文件路径
        
    Returns:
        符合要求的data_id字符串
    """
    # 从文件路径提取文件名（不含扩展名）
    file_name = Path(file_path).stem
    
    # 清理文件名，只保留允许的字符
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file_name)
    
    # 添加时间戳确保唯一性
    timestamp = str(int(time.time()))
    
    # 组合生成data_id
    data_id = f"{clean_name}_{timestamp}"
    
    # 确保不超过128字符
    return data_id[:128]

def parse_pdf(document_url: str) -> Dict[str, Any]:
    """
    使用MinerU API解析PDF文档，自动下载并提取Markdown内容
    
    Args:
        document_url: PDF文档的URL
        
    Returns:
        解析结果字典，包含Markdown内容
    """
    try:
        # 第一步：创建解析任务
        task_result = create_parse_task(document_url)
        if "error" in task_result:
            return task_result
        
        task_id = task_result["task_id"]
        
        # 第二步：获取解析结果
        result = get_parse_result(task_id)
        if "error" in result:
            return result
        
        # 第三步：下载并解压ZIP文件，提取Markdown内容
        if "download_url" in result:
            content_result = download_and_extract_content(result["download_url"])
            if "error" not in content_result:
                # 合并结果
                result.update(content_result)
            else:
                return content_result  # 返回错误信息
        
        return result
        
    except Exception as e:
        return {"error": f"Error parsing PDF: {str(e)}"}

def create_parse_task(document_url: str) -> Dict[str, Any]:
    """
    创建解析任务
    """
    try:
        if not (api_key := os.getenv("MINERU_API_KEY")):
            return {"error": "MINERU_API_KEY not found in environment variables"}
        
        # 生成data_id
        data_id = generate_data_id(document_url)
        
        # API端点
        url = "https://mineru.net/api/v4/extract/task"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 请求数据 - 写死页数范围，不指定语言
        data = {
            "url": document_url,           # 文档URL
            "is_ocr": True,               # OCR功能
            "enable_formula": True,       # 公式识别
            "enable_table": True,         # 表格识别
            "data_id": data_id,           # 数据ID
            "page_range": "1-50",         # 写死最大50页
            "model_version": "pipeline"   # 模型版本
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            # 修复：从data字段中获取task_id
            data_section = result.get("data", {})
            task_id = data_section.get("task_id")
            if task_id:
                return {
                    "status": "task_created",
                    "task_id": task_id,
                    "data_id": data_id
                }
            else:
                return {"error": "Failed to get task_id from response", "raw_response": result}
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        return {"error": f"Error creating task: {str(e)}"}

def get_parse_result(task_id: str, max_retries: int = 5, wait_seconds: int = 10) -> Dict[str, Any]:
    """
    获取解析结果
    """
    try:
        if not (api_key := os.getenv("MINERU_API_KEY")):
            return {"error": "MINERU_API_KEY not found"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 轮询获取结果
        for attempt in range(max_retries):
            url = f"https://mineru.net/api/v4/extract/task/{task_id}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查API调用是否成功
                if result.get("code") != 0:
                    return {
                        "error": f"API error: {result.get('msg', 'Unknown error')}",
                        "raw_response": result
                    }
                
                # 获取任务状态
                data = result.get("data", {})
                state = data.get("state")
                
                if state == "done":
                    # 解析完成，返回结果
                    return {
                        "status": "success",
                        "task_id": data.get("task_id"),
                        "data_id": data.get("data_id"),
                        "download_url": data.get("full_zip_url"),
                        "raw_response": result
                    }
                elif state == "failed":
                    return {
                        "error": f"Parsing failed: {data.get('err_msg', 'Unknown error')}",
                        "raw_response": result
                    }
                elif state in ["pending", "running", "converting"]:
                    # 任务还在处理中，显示进度信息
                    progress = data.get("extract_progress", {})
                    print(f"Task {state}: {progress.get('extracted_pages', 0)}/{progress.get('total_pages', 0)} pages processed")
                    
                    # 等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(wait_seconds)
                        continue
                    else:
                        return {"error": "Parsing timeout"}
                else:
                    return {
                        "error": f"Unknown state: {state}",
                        "raw_response": result
                    }
            else:
                return {
                    "error": f"Failed to get task result: {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {"error": f"Error getting result: {str(e)}"}

def download_and_extract_content(download_url: str) -> Dict[str, Any]:
    """
    下载ZIP文件并提取Markdown和JSON内容
    
    Args:
        download_url: ZIP文件的下载链接
        
    Returns:
        包含markdown_content和json_content的字典
    """
    try:
        # 下载ZIP文件
        print(f"📥 Downloading ZIP file from: {download_url}")
        response = requests.get(download_url)
        
        if response.status_code != 200:
            return {"error": f"Failed to download ZIP file: {response.status_code}"}
        
        # 在内存中处理ZIP文件
        zip_buffer = io.BytesIO(response.content)
        
        markdown_content = ""
        json_content = ""
        extracted_files = []
        
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            file_list = zip_file.namelist()
            print(f"📁 ZIP文件包含的文件: {file_list}")
            
            # 查找Markdown文件
            md_files = [f for f in file_list if f.endswith('.md')]
            json_files = [f for f in file_list if f.endswith('.json')]
            
            # 提取Markdown内容
            if md_files:
                # 优先使用full.md，如果没有则使用第一个md文件
                md_file = "full.md" if "full.md" in md_files else md_files[0]
                with zip_file.open(md_file) as f:
                    markdown_content = f.read().decode('utf-8')
                print(f"📄 提取Markdown文件: {md_file}")
                extracted_files.append(md_file)
            
            # 提取JSON内容（用于结构化数据）
            if json_files:
                # 优先使用layout.json，如果没有则使用第一个json文件
                json_file = "layout.json" if "layout.json" in json_files else json_files[0]
                with zip_file.open(json_file) as f:
                    json_content = f.read().decode('utf-8')
                print(f"📊 提取JSON文件: {json_file}")
                extracted_files.append(json_file)
        
        result = {
            "markdown_content": markdown_content,
            "json_content": json_content,
            "extracted_files": extracted_files,
            "output_format": "Markdown"
        }
        
        print(f"✅ 成功提取内容，Markdown长度: {len(markdown_content)} 字符")
        return result
        
    except Exception as e:
        return {"error": f"Error downloading and extracting content: {str(e)}"}

def parse_csv(file_path: str) -> Dict[str, Any]:
    """
    Parse CSV document using pandas and convert to Markdown format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed content and metadata in Markdown format
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Read CSV with pandas
        df = pd.read_csv(file_path)
        
        # 使用pandas内置方法转换为Markdown
        markdown_content = f"# {file_path.name}\n\n"
        markdown_content += f"**行数**: {len(df)}\n"
        markdown_content += f"**列数**: {len(df.columns)}\n\n"
        markdown_content += "## 完整数据\n\n"
        
        # 使用pandas内置的to_markdown()方法转换整个DataFrame
        markdown_content += df.to_markdown(index=False)
        
        # Extract basic information
        info = {
            "file_name": file_path.name,
            "file_type": "csv",
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "sample_data": df.head(5).to_dict('records'),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "markdown_content": markdown_content,  # 包含完整CSV数据的Markdown格式
            "output_format": "Markdown"
        }
        
        return info
        
    except Exception as e:
        return {"error": f"Error parsing CSV: {str(e)}"}