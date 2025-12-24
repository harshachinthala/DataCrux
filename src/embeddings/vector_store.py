"""
FAISS vector store for efficient similarity search
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss

from .embedding_generator import EmbeddingGenerator
from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for chunk embeddings"""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        index_path: Optional[Path] = None
    ):
        """
        Initialize vector store
        
        Args:
            embedding_generator: Embedding generator instance
            index_path: Path to save/load FAISS index
        """
        self.embedding_generator = embedding_generator
        self.index_path = index_path
        self.vector_store: Optional[FAISS] = None
        
        # Load existing index if path provided and exists
        if index_path and index_path.exists():
            self.load(index_path)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Convert chunks to LangChain Documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id
                }
            )
            for chunk in chunks
        ]
        
        # Create or update vector store
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_generator.embeddings
            )
            logger.info("Created new FAISS index")
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added documents to existing index")
        
        # Auto-save if path is set
        if self.index_path:
            self.save(self.index_path)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar chunks
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"sheet_name": "Sales"})
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        # Log vector store stats
        stats = self.get_stats()
        logger.info(f"Vector store has {stats.get('total_vectors', 0)} vectors")
        logger.debug(f"Searching for top {k} similar chunks with threshold {score_threshold}")
        
        # Perform similarity search with scores
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Error during similarity search (possible index corruption): {e}")
            # Identify if it is the specific KeyError for index sync issues
            if "KeyError" in str(e) or isinstance(e, KeyError):
                logger.critical("Vector store index is corrupted (ID mismatch). Please clear data and re-upload.")
            return []
        
        logger.info(f"FAISS returned {len(results)} results before filtering")
        
        # Convert all scores from L2 distance to similarity (1 / (1 + distance))
        converted_results = [(doc, 1 / (1 + score)) for doc, score in results]
        
        # Filter by score threshold only if threshold > 0
        if score_threshold is not None and score_threshold > 0.0:
            filtered_results = [
                (doc, sim_score)
                for doc, sim_score in converted_results
                if sim_score >= score_threshold
            ]
            logger.info(f"After threshold filtering ({score_threshold}): {len(filtered_results)} results")
            results = filtered_results
        else:
            logger.info(f"No threshold filtering (threshold={score_threshold})")
            results = converted_results
        
        logger.info(f"Returning {len(results)} results")
        return results
    
    def get_relevant_context(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> str:
        """
        Get formatted context string from relevant chunks
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            filter_dict: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            Formatted context string
        """
        results = self.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
            score_threshold=score_threshold
        )
        
        if not results:
            return "No relevant context found."
        
        # Format context
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            context_parts.append(
                f"[Context {i}] (Relevance: {score:.2f})\n"
                f"Source: {metadata.get('filename', 'Unknown')} - "
                f"Sheet: {metadata.get('sheet_name', 'Unknown')}\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def save(self, path: Path) -> None:
        """
        Save vector store to disk
        
        Args:
            path: Directory path to save index
        """
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving vector store to {path}")
        self.vector_store.save_local(str(path))
        logger.info("Vector store saved successfully")
    
    def load(self, path: Path) -> None:
        """
        Load vector store from disk
        
        Args:
            path: Directory path containing saved index
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")
        
        logger.info(f"Loading vector store from {path}")
        
        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(path),
                embeddings=self.embedding_generator.embeddings,
                allow_dangerous_deserialization=True  # Required for pickle
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def delete(self) -> None:
        """Delete the vector store"""
        self.vector_store = None
        logger.info("Vector store deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        if self.vector_store is None:
            return {
                "status": "empty",
                "total_vectors": 0,
                "dimension": self.embedding_generator.dimension,
                "index_type": "None"
            }
        
        # Get index stats
        index = self.vector_store.index
        
        return {
            "status": "active",
            "total_vectors": index.ntotal,
            "dimension": self.embedding_generator.dimension,
            "index_type": type(index).__name__
        }
