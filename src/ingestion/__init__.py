"""
Data ingestion module for Excel file processing
"""

from .excel_parser import ExcelParser
from .chunker import ExcelChunker

__all__ = ["ExcelParser", "ExcelChunker"]
