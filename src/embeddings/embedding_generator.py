"""
Embedding generation using HuggingFace and OpenAI models
"""

from abc import ABC, abstractmethod
from typing import List, Union
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings as LangChainHFEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass


class HuggingFaceEmbeddings(EmbeddingGenerator):
    """HuggingFace embedding generator using sentence-transformers"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str = None
    ):
        """
        Initialize HuggingFace embeddings
        
        Args:
            model_name: Name of the HuggingFace model
            cache_folder: Directory to cache the model
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Use LangChain wrapper for consistency
        self.embeddings = LangChainHFEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Also load sentence-transformers model for dimension info
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddings(EmbeddingGenerator):
    """OpenAI embedding generator"""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None
    ):
        """
        Initialize OpenAI embeddings
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (if not set in environment)
        """
        self.model = model
        
        logger.info(f"Initializing OpenAI embeddings: {model}")
        
        kwargs = {"model": model}
        if api_key:
            kwargs["openai_api_key"] = api_key
        
        self.embeddings = LangChainOpenAIEmbeddings(**kwargs)
        
        # Dimension mapping for OpenAI models
        self._dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Model initialized. Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents with OpenAI")
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self._dimension_map.get(self.model, 1536)


def create_embedding_generator(
    provider: str = "huggingface",
    model_name: str = None,
    api_key: str = None
) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator
    
    Args:
        provider: 'huggingface' or 'openai'
        model_name: Model name (provider-specific)
        api_key: API key for OpenAI
        
    Returns:
        EmbeddingGenerator instance
    """
    if provider.lower() == "huggingface":
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model)
    
    elif provider.lower() == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model, api_key=api_key)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
