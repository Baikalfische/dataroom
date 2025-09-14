"""
Utility functions for dataroom.
"""

from pathlib import Path
from typing import Dict, Any

def load_txt_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load prompts from a text file.
    
    Args:
        file_path: Path to the prompt file
        
    Returns:
        Dictionary with loaded prompts
    """
    prompt_file = Path(file_path)
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    # Read the entire file content as system prompt
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    return {
        "system_prompt": content
    }
