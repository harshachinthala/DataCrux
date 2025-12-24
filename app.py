"""
Excel RAG Pipeline - Main Entry Point
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.app_ui import ExcelRAGApp
from src.config import settings


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)


def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("Starting Excel RAG Assistant")
    logger.info("="*80)
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Chunking Strategy: {settings.chunking_strategy}")
    logger.info(f"Vector DB Path: {settings.vector_db_dir}")
    logger.info("="*80)
    
    try:
        # Create and launch app
        app = ExcelRAGApp()
        app.launch()
    
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
