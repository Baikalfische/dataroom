"""
Main application entry point for the Real Estate Dataroom.
"""

import os
import sys
import warnings
from typing import *
from pathlib import Path
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataroom.agent.agent import Agent
from dataroom.tools import RealEstateRAGTool
from dataroom.utils.utils import load_txt_prompts_from_file

warnings.filterwarnings("ignore")
_ = load_dotenv()


def initialize_real_estate_agent(
    prompt_file="docs/system_prompt.txt",
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    top_p=0.95,
    gemini_kwargs={}
) -> Agent:
    """Initialize the Real Estate Agent with RAG tool.

    Args:
        prompt_file (str): Path to file containing system prompts
        model (str): Model to use. Defaults to "gemini-2.5-flash-lite".
        temperature (float): Temperature for the model. Defaults to 0.7.
        top_p (float): Top P for the model. Defaults to 0.95.
        gemini_kwargs (dict): Additional keyword arguments for Gemini API.

    Returns:
        Agent: Initialized real estate agent
    """
    print("🚀 Initializing Real Estate Agent...")
    
    # Load system prompt
    prompts = load_txt_prompts_from_file(prompt_file)
    system_prompt = prompts["system_prompt"]
    
    # Initialize RAG tool
    print("🔧 Initializing RAG Tool...")
    rag_tool = RealEstateRAGTool()
    print("✅ RAG Tool initialized")
    
    # Initialize LLM
    print("🧠 Initializing LLM...")
    model = ChatGoogleGenerativeAI(
        model=model, 
        temperature=temperature, 
        top_p=top_p,
        **gemini_kwargs
    )
    print("✅ LLM initialized")
    
    # Initialize checkpointer for conversation memory
    checkpointer = MemorySaver()
    
    # Initialize Real Estate Agent
    print("🏠 Initializing Real Estate Agent...")
    agent = Agent(
        model=model,
        tools=[rag_tool], 
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        log_tools=True,
        log_dir="logs"
    )
    print("✅ Real Estate Agent initialized successfully")
    
    return agent


if __name__ == "__main__":
    """
    This is the main entry point for the Real Estate Dataroom application.
    It initializes the agent and starts the Gradio web interface.
    """
    print("🏠 Starting Real Estate Dataroom...")

    # Collect the ENV variables
    gemini_kwargs = {}
    if google_api_key := os.getenv("GOOGLE_API_KEY"):
        gemini_kwargs["api_key"] = google_api_key

    # Initialize the agent
    agent = initialize_real_estate_agent(
        "docs/system_prompt.txt",
        model="gemini-2.5-flash-lite",
        temperature=0.7,
        top_p=0.95,
        gemini_kwargs=gemini_kwargs
    )
    print("🏠 Real Estate Agent initialized successfully")
    
    # Start Gradio web interface
    print("🌐 Starting Gradio web interface...")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        from dataroom.ui.interface import create_demo
        
        demo = create_demo(agent)
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False
        )
        
    except ImportError as e:
        print(f"❌ Error importing Gradio interface: {e}")
        print("Make sure gradio is installed: pip install gradio")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")