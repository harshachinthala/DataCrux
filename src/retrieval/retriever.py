"""
Query processing and context retrieval
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_core.documents import Document

from ..embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Handle query processing and context retrieval"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.5
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store instance
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            k: Number of results (overrides default)
            filters: Metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or self.top_k
        
        logger.info(f"Retrieving top {k} chunks for query: {query[:100]}...")
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_dict=filters,
            score_threshold=self.score_threshold
        )
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results
    
    def format_context(
        self,
        results: List[Tuple[Document, float]],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved chunks into context string for LLM
        
        Args:
            results: List of (Document, score) tuples
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant data found in the uploaded Excel files."
        
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            if include_metadata:
                metadata = doc.metadata
                header = (
                    f"[CONTEXT CHUNK {i}]\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"Source File: {metadata.get('filename', 'Unknown')}\n"
                    f"Sheet: {metadata.get('sheet_name', 'Unknown')}\n"
                    f"Chunk Type: {metadata.get('chunk_type', 'Unknown')}\n"
                )
                context_parts.append(header + "\n" + doc.page_content)
            else:
                context_parts.append(doc.page_content)
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def retrieve_and_format(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Retrieve chunks and format them into a single context string
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            filters: Metadata filters
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k, filters)
        
        if not results:
            return "No relevant documents found."
            
        context_parts = []
        for i, (doc, score) in enumerate(results, 1): # Keep score for potential future use or consistency, though not used in formatting here
            content = doc.page_content
            
            if include_metadata and doc.metadata:
                # Add source info if available
                source = doc.metadata.get("filename", "Unknown")
                sheet = doc.metadata.get("sheet_name", "")
                if sheet:
                    source += f" (Sheet: {sheet})"
                
                # Format as: [Source] Content
                context_parts.append(f"[{source}]\n{content}")
            else:
                context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def get_retrieval_stats(
        self,
        results: List[Tuple[Document, float]]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results
        
        Args:
            results: Retrieval results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"count": 0}
        
        scores = [score for _, score in results]
        sources = set()
        sheets = set()
        
        for doc, _ in results:
            metadata = doc.metadata
            sources.add(metadata.get('filename', 'Unknown'))
            sheets.add(metadata.get('sheet_name', 'Unknown'))
        
        return {
            "count": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "unique_sources": len(sources),
            "unique_sheets": len(sheets),
            "sources": list(sources),
            "sheets": list(sheets)
        }
