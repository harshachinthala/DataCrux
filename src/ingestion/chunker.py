"""
Semantic chunking strategies for Excel data
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import uuid
from langchain_core.documents import Document

from .excel_parser import ExcelFileData, SheetData

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Container for a data chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class ExcelChunker:
    """Convert Excel data into semantic chunks for embedding"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "row"
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks (for sliding strategy)
            strategy: Chunking strategy ('row', 'group', 'sheet', 'sliding')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        self.strategy_map = {
            "row": self._chunk_by_row,
            "group": self._chunk_by_group,
            "sheet": self._chunk_by_sheet,
            "sliding": self._chunk_sliding_window
        }
        
        if strategy not in self.strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def chunk_excel_data(self, excel_data: ExcelFileData) -> List[Chunk]:
        """
        Chunk Excel file data using the configured strategy
        
        Args:
            excel_data: Parsed Excel file data
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Chunking {excel_data.filename} using '{self.strategy}' strategy")
        
        all_chunks = []
        chunk_func = self.strategy_map[self.strategy]
        
        for sheet_name, sheet_data in excel_data.sheets.items():
            chunks = chunk_func(sheet_data, excel_data.filename)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {excel_data.filename}")
        return all_chunks

    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """Convert LangChain Document objects into our Chunk model."""
        chunks: List[Chunk] = []
        for doc in docs:
            # Use the same metadata handling you already have for Excel sheets
            metadata = doc.metadata.copy()
            metadata.setdefault("filename", doc.metadata.get("source", "unknown"))
            metadata["chunk_type"] = "document"
            
            # Simple chunking: each document becomes one chunk (you can add further splitting if desired)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=doc.page_content,
                metadata=metadata
            )
            chunks.append(chunk)
        return chunks
    
    def _chunk_by_row(self, sheet_data: SheetData, filename: str) -> List[Chunk]:
        """
        Create one chunk per row with column context
        
        Each chunk contains: column names + row values
        """
        chunks = []
        df = sheet_data.data
        
        for idx, row in df.iterrows():
            # Format row as "Column: Value" pairs
            row_parts = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value):  # Skip NaN values
                    row_parts.append(f"{col}: {value}")
            
            content = " | ".join(row_parts)
            
            # Truncate if exceeds chunk size
            if len(content) > self.chunk_size:
                content = content[:self.chunk_size - 3] + "..."
            
            chunk = Chunk(
                content=content,
                metadata={
                    "filename": filename,
                    "sheet_name": sheet_data.name,
                    "row_index": int(idx),
                    "chunk_type": "row",
                    "columns": list(df.columns)
                },
                chunk_id=f"{filename}_{sheet_data.name}_row_{idx}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_group(
        self,
        sheet_data: SheetData,
        filename: str,
        group_size: int = 10
    ) -> List[Chunk]:
        """
        Create chunks by grouping multiple rows together
        
        Useful for reducing total chunk count while preserving context
        """
        chunks = []
        df = sheet_data.data
        
        for start_idx in range(0, len(df), group_size):
            end_idx = min(start_idx + group_size, len(df))
            group_df = df.iloc[start_idx:end_idx]
            
            # Format as table-like text
            content_parts = [
                f"Sheet: {sheet_data.name}",
                f"Rows {start_idx} to {end_idx - 1}",
                f"Columns: {', '.join(df.columns)}",
                ""
            ]
            
            for idx, row in group_df.iterrows():
                row_str = " | ".join(
                    f"{col}: {row[col]}"
                    for col in df.columns
                    if pd.notna(row[col])
                )
                content_parts.append(row_str)
            
            content = "\n".join(content_parts)
            
            # Truncate if needed
            if len(content) > self.chunk_size:
                content = content[:self.chunk_size - 3] + "..."
            
            chunk = Chunk(
                content=content,
                metadata={
                    "filename": filename,
                    "sheet_name": sheet_data.name,
                    "row_start": start_idx,
                    "row_end": end_idx - 1,
                    "chunk_type": "group",
                    "columns": list(df.columns)
                },
                chunk_id=f"{filename}_{sheet_data.name}_group_{start_idx}_{end_idx}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sheet(self, sheet_data: SheetData, filename: str) -> List[Chunk]:
        """
        Create summary chunks per sheet
        
        Includes: sheet name, columns, data types, sample rows, basic stats
        """
        df = sheet_data.data
        
        # Create summary
        summary_parts = [
            f"Sheet: {sheet_data.name}",
            f"Rows: {sheet_data.row_count}",
            f"Columns: {', '.join(sheet_data.columns)}",
            "",
            "Data Types:"
        ]
        
        for col, dtype in sheet_data.dtypes.items():
            summary_parts.append(f"  - {col}: {dtype}")
        
        summary_parts.append("\nSample Data (first 5 rows):")
        
        for idx, row in df.head(5).iterrows():
            row_str = " | ".join(
                f"{col}: {row[col]}"
                for col in df.columns
                if pd.notna(row[col])
            )
            summary_parts.append(row_str)
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nNumeric Column Statistics:")
            for col in numeric_cols:
                stats = df[col].describe()
                summary_parts.append(
                    f"  - {col}: mean={stats['mean']:.2f}, "
                    f"min={stats['min']:.2f}, max={stats['max']:.2f}"
                )
        
        content = "\n".join(summary_parts)
        
        chunk = Chunk(
            content=content,
            metadata={
                "filename": filename,
                "sheet_name": sheet_data.name,
                "chunk_type": "sheet_summary",
                "columns": list(df.columns),
                "row_count": sheet_data.row_count
            },
            chunk_id=f"{filename}_{sheet_data.name}_summary"
        )
        
        return [chunk]
    
    def _chunk_sliding_window(
        self,
        sheet_data: SheetData,
        filename: str
    ) -> List[Chunk]:
        """
        Create overlapping chunks using sliding window
        
        Useful for preserving context across row boundaries
        """
        chunks = []
        df = sheet_data.data
        
        # Convert entire sheet to text first
        full_text_parts = []
        for idx, row in df.iterrows():
            row_str = " | ".join(
                f"{col}: {row[col]}"
                for col in df.columns
                if pd.notna(row[col])
            )
            full_text_parts.append(row_str)
        
        full_text = "\n".join(full_text_parts)
        
        # Create sliding windows
        start = 0
        chunk_num = 0
        
        while start < len(full_text):
            end = start + self.chunk_size
            content = full_text[start:end]
            
            chunk = Chunk(
                content=content,
                metadata={
                    "filename": filename,
                    "sheet_name": sheet_data.name,
                    "chunk_type": "sliding_window",
                    "chunk_number": chunk_num,
                    "columns": list(df.columns)
                },
                chunk_id=f"{filename}_{sheet_data.name}_window_{chunk_num}"
            )
            chunks.append(chunk)
            
            # Move window with overlap
            start += (self.chunk_size - self.chunk_overlap)
            chunk_num += 1
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "strategy": self.strategy,
            "sheets": len(set(chunk.metadata["sheet_name"] for chunk in chunks))
        }
