"""
Main application entry point for the Chat-Enabled Dataroom.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from dataroom.config import Config
from dataroom.ui.gradio_app import create_app
from dataroom.utils.logging import setup_logging

def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config()
    
    # Create and launch Gradio app
    app = create_app(config)
    app.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        debug=config.DEBUG
    )

if __name__ == "__main__":
    main()
