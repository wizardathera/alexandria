"""
Embedding services for the DBC application.

This module provides embedding generation using OpenAI's APIs with
proper error handling, rate limiting, and cost tracking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from src.utils.logger import get_logger
from src.utils.config import get_settings

# Initialize logger early for dependency import logging
logger = get_logger(__name__)

# Import OpenAI dependencies separately for better error diagnosis
try:
    import openai
    from openai import OpenAI
except ImportError as e:
    logger.warning(f"OpenAI library not available: {e}")
    openai = None
    OpenAI = None

# Define Document protocol for type safety
class DocumentProtocol(Protocol):
    """Protocol for Document-like objects."""
    page_content: str
    metadata: Dict[str, Any]

# Import LangChain dependencies separately for better error diagnosis
try:
    from langchain.schema import Document
    # Make sure Document class is available as a proper type
    DocumentClass = Document
except ImportError as e:
    logger.warning(f"LangChain library not available: {e}")
    # Create a fallback Document class that matches the protocol
    class DocumentClass:
        def __init__(self, page_content: str = "", metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


@dataclass
class EmbeddingMetrics:
    """Metrics for tracking embedding generation."""
    total_tokens: int = 0
    total_requests: int = 0
    total_documents: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    def add_request(self, tokens: int, success: bool = True, error: str = None):
        """Add metrics for a single request."""
        self.total_requests += 1
        if success:
            self.total_tokens += tokens
        else:
            self.errors.append(error or "Unknown error")
    
    def finish(self):
        """Mark metrics as complete."""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_cost_estimate(self, model: str = "text-embedding-ada-002") -> float:
        """
        Estimate cost in USD based on OpenAI pricing.
        
        Args:
            model: Embedding model name
            
        Returns:
            float: Estimated cost in USD
        """
        # OpenAI pricing as of 2024 (per 1K tokens)
        pricing = {
            "text-embedding-ada-002": 0.0001,
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013
        }
        
        price_per_1k = pricing.get(model, 0.0001)
        return (self.total_tokens / 1000) * price_per_1k


class EmbeddingProviderInterface(ABC):
    """
    Abstract interface for embedding providers.
    
    This interface allows easy switching between different embedding providers
    (OpenAI, Anthropic, local models, etc.).
    """
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], EmbeddingMetrics]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the dimension of embeddings for the specified model."""
        pass
    
    @abstractmethod
    def get_max_tokens(self, model: Optional[str] = None) -> int:
        """Get maximum tokens per request for the specified model."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProviderInterface):
    """
    OpenAI embedding provider using their API.
    
    Handles rate limiting, batching, and error recovery for OpenAI embeddings.
    """
    
    def __init__(self):
        """Initialize OpenAI embedding provider."""
        self.settings = get_settings()
        self.client = None
        self._initialize_client()
        
        # Model configurations
        self.model_configs = {
            "text-embedding-ada-002": {
                "dimension": 1536,
                "max_tokens": 8191,
                "batch_size": 100
            },
            "text-embedding-3-small": {
                "dimension": 1536,
                "max_tokens": 8191,
                "batch_size": 100
            },
            "text-embedding-3-large": {
                "dimension": 3072,
                "max_tokens": 8191,
                "batch_size": 100
            }
        }
        
        self.default_model = self.settings.embedding_model
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key."""
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
        
        try:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            logger.info("OpenAI embedding client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], EmbeddingMetrics]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            model: Optional model name (defaults to text-embedding-ada-002)
            
        Returns:
            Tuple[List[List[float]], EmbeddingMetrics]: Embeddings and metrics
        """
        model = model or self.default_model
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}")
        
        config = self.model_configs[model]
        metrics = EmbeddingMetrics()
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {model}")
        
        try:
            # Filter out empty texts
            valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
            if not valid_texts:
                logger.warning("No valid texts to embed")
                return [[] for _ in texts], metrics
            
            embeddings = [[] for _ in texts]  # Initialize with empty lists
            
            # Process in batches to respect rate limits
            batch_size = config["batch_size"]
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_texts = [text for _, text in batch]
                batch_indices = [idx for idx, _ in batch]
                
                # Generate embeddings for batch
                batch_embeddings, batch_metrics = await self._generate_batch_embeddings(
                    batch_texts, model
                )
                
                # Place embeddings in correct positions
                for j, embedding in enumerate(batch_embeddings):
                    original_idx = batch_indices[j]
                    embeddings[original_idx] = embedding
                
                metrics.total_tokens += batch_metrics.total_tokens
                metrics.total_requests += batch_metrics.total_requests
                metrics.errors.extend(batch_metrics.errors)
                
                # Rate limiting delay
                await asyncio.sleep(0.1)
            
            metrics.total_documents = len(valid_texts)
            metrics.finish()
            
            logger.info(f"Generated embeddings: {metrics.total_documents} documents, "
                       f"{metrics.total_tokens} tokens, ${metrics.get_cost_estimate(model):.4f} estimated cost")
            
            return embeddings, metrics
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            metrics.add_request(0, False, str(e))
            metrics.finish()
            raise ValueError(f"Embedding generation failed: {e}")
    
    async def _generate_batch_embeddings(
        self,
        texts: List[str],
        model: str
    ) -> Tuple[List[List[float]], EmbeddingMetrics]:
        """
        Generate embeddings for a single batch.
        
        Args:
            texts: Batch of texts to embed
            model: Model name
            
        Returns:
            Tuple[List[List[float]], EmbeddingMetrics]: Batch embeddings and metrics
        """
        metrics = EmbeddingMetrics()
        
        try:
            # Calculate token count (approximate)
            total_tokens = sum(len(text.split()) for text in texts)
            
            # Make API request - OpenAI client is sync, so we need to run in executor
            import asyncio
            import functools
            
            # Run synchronous OpenAI call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    self.client.embeddings.create,
                    input=texts,
                    model=model
                )
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Update metrics - handle both response types
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.total_tokens
            else:
                actual_tokens = total_tokens
            metrics.add_request(actual_tokens, True)
            
            return embeddings, metrics
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            metrics.add_request(0, False, str(e))
            
            # Return empty embeddings for failed batch
            dimension = self.get_embedding_dimension(model)
            empty_embeddings = [[0.0] * dimension for _ in texts]
            return empty_embeddings, metrics
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get embedding dimension for the specified model.
        
        Args:
            model: Model name
            
        Returns:
            int: Embedding dimension
        """
        model = model or self.default_model
        return self.model_configs.get(model, {}).get("dimension", 1536)
    
    def get_max_tokens(self, model: Optional[str] = None) -> int:
        """
        Get maximum tokens per request for the specified model.
        
        Args:
            model: Model name
            
        Returns:
            int: Maximum tokens
        """
        model = model or self.default_model
        return self.model_configs.get(model, {}).get("max_tokens", 8191)


class CachedEmbeddingProvider:
    """
    Wrapper that adds caching to any embedding provider.
    
    Caches embeddings based on text hash to avoid regenerating
    embeddings for identical text content.
    """
    
    def __init__(self, provider: EmbeddingProviderInterface, cache_size: int = 10000):
        """
        Initialize cached embedding provider.
        
        Args:
            provider: Underlying embedding provider
            cache_size: Maximum number of cached embeddings
        """
        self.provider = provider
        self.cache = {}
        self.cache_size = cache_size
        self.hit_count = 0
        self.miss_count = 0
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], EmbeddingMetrics]:
        """
        Generate embeddings with caching.
        
        Args:
            texts: List of texts to embed
            model: Optional model name
            
        Returns:
            Tuple[List[List[float]], EmbeddingMetrics]: Cached/generated embeddings and metrics
        """
        model = model or "text-embedding-ada-002"
        
        # Check cache for each text
        cached_embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, model)
            
            if cache_key in self.cache:
                cached_embeddings.append((i, self.cache[cache_key]))
                self.hit_count += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self.miss_count += 1
        
        logger.info(f"Cache stats: {self.hit_count} hits, {self.miss_count} misses, "
                   f"{len(uncached_texts)} texts need embedding")
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings, metrics = await self.provider.generate_embeddings(uncached_texts, model)
            
            # Cache new embeddings
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                if embedding:  # Only cache non-empty embeddings
                    cache_key = self._get_cache_key(text, model)
                    self._add_to_cache(cache_key, embedding)
        else:
            new_embeddings = []
            metrics = EmbeddingMetrics()
        
        # Combine cached and new embeddings
        final_embeddings = [[] for _ in texts]
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            final_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, embedding in enumerate(new_embeddings):
            original_idx = uncached_indices[i]
            final_embeddings[original_idx] = embedding
        
        return final_embeddings, metrics
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: List[float]):
        """Add embedding to cache with size limit."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size
        }


class EmbeddingService:
    """
    High-level embedding service for the DBC application.
    
    Provides a simple interface for generating embeddings from documents
    with automatic provider selection and optimization.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize embedding service.
        
        Args:
            use_cache: Whether to use embedding cache
        """
        self.settings = get_settings()
        
        # Initialize provider
        base_provider = OpenAIEmbeddingProvider()
        
        if use_cache:
            self.provider = CachedEmbeddingProvider(base_provider)
        else:
            self.provider = base_provider
    
    async def embed_documents(
        self,
        documents: List[Union[DocumentProtocol, Dict[str, Any]]],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], EmbeddingMetrics]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
            model: Optional model name
            
        Returns:
            Tuple[List[List[float]], EmbeddingMetrics]: Embeddings and metrics
        """
        if not documents:
            return [], EmbeddingMetrics()
        
        # Extract text content from documents (handle both Document objects and dicts)
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                texts.append(doc['page_content'])
            else:
                # Fallback for plain text or other formats
                texts.append(str(doc))
        
        logger.info(f"Embedding {len(documents)} documents")
        
        try:
            embeddings, metrics = await self.provider.generate_embeddings(texts, model)
            
            logger.info(f"Embedding complete: {metrics.total_documents} documents processed")
            
            return embeddings, metrics
            
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise
    
    async def embed_query(self, query: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            model: Optional model name
            
        Returns:
            List[float]: Query embedding
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        embeddings, _ = await self.provider.generate_embeddings([query], model)
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model."""
        if hasattr(self.provider, 'provider'):
            # CachedEmbeddingProvider wraps another provider
            return self.provider.provider.get_embedding_dimension(model)
        else:
            # Direct provider (OpenAIEmbeddingProvider)
            return self.provider.get_embedding_dimension(model)
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if hasattr(self.provider, 'get_cache_stats'):
            return self.provider.get_cache_stats()
        return None