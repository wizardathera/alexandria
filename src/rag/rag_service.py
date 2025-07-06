"""
RAG (Retrieval-Augmented Generation) service for the Alexandria application.

This module provides intelligent Q&A functionality using vector similarity search
and LLM response generation with proper context management and source citations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
import json

try:
    import openai
    from openai import OpenAI
    from langchain.schema import Document
except ImportError:
    # Fallback for testing without full installation
    openai = None
    OpenAI = None
    Document = dict

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.database import get_database
from src.utils.embeddings import EmbeddingService

logger = get_logger(__name__)


@dataclass
class QueryContext:
    """Context information for a query."""
    documents: List[Dict[str, Any]] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    total_retrieved: int = 0
    query_embedding_time: float = 0.0
    vector_search_time: float = 0.0


@dataclass
class RAGResponse:
    """Response from RAG query processing."""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: QueryContext
    token_usage: Dict[str, int]
    processing_time: float
    confidence_score: Optional[float] = None


class LLMProviderInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    This interface allows easy switching between different LLM providers
    (OpenAI, Anthropic, local models, etc.).
    """
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def get_context_limit(self, model: Optional[str] = None) -> int:
        """Get maximum context length for the model."""
        pass


class OpenAILLMProvider(LLMProviderInterface):
    """
    OpenAI LLM provider for response generation.
    
    Handles chat completion API calls with proper error handling and token tracking.
    """
    
    def __init__(self):
        """Initialize OpenAI LLM provider."""
        self.settings = get_settings()
        self.client = None
        self._initialize_client()
        
        # Model configurations
        self.model_configs = {
            "gpt-3.5-turbo": {
                "context_limit": 16385,
                "max_output_tokens": 4096,
                "cost_per_1k_input": 0.0005,
                "cost_per_1k_output": 0.0015
            },
            "gpt-4": {
                "context_limit": 8192,
                "max_output_tokens": 4096,
                "cost_per_1k_input": 0.03,
                "cost_per_1k_output": 0.06
            },
            "gpt-4-turbo": {
                "context_limit": 128000,
                "max_output_tokens": 4096,
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03
            }
        }
        
        self.default_model = "gpt-3.5-turbo"
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key."""
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
        
        try:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            logger.info("OpenAI LLM client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate response using OpenAI Chat Completion API.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            model: Model name to use
            
        Returns:
            Tuple[str, Dict[str, int]]: Generated response and token usage
        """
        model = model or self.default_model
        if model not in self.model_configs:
            raise ValueError(f"Unsupported model: {model}")
        
        config = self.model_configs[model]
        max_tokens = max_tokens or min(1000, config["max_output_tokens"])
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "text"}
            )
            
            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"LLM response generated: {token_usage['total_tokens']} tokens used")
            
            return answer, token_usage
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise ValueError(f"Failed to generate response: {e}")
    
    def get_context_limit(self, model: Optional[str] = None) -> int:
        """
        Get maximum context length for the model.
        
        Args:
            model: Model name
            
        Returns:
            int: Maximum context length in tokens
        """
        model = model or self.default_model
        return self.model_configs.get(model, {}).get("context_limit", 16385)


class RAGService:
    """
    High-level RAG service for intelligent question answering.
    
    Combines vector similarity search with LLM response generation to provide
    contextual answers with proper source citations and confidence scoring.
    """
    
    def __init__(self):
        """Initialize RAG service with default providers."""
        self.settings = get_settings()
        self.embedding_service = EmbeddingService(use_cache=True)
        self.llm_provider = OpenAILLMProvider()
        self.vector_db = None
    
    async def initialize(self) -> bool:
        """
        Initialize RAG service components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize vector database
            self.vector_db = await get_database()
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False
    
    async def query(
        self,
        question: str,
        book_id: Optional[str] = None,
        context_limit: int = 5,
        model: Optional[str] = None
    ) -> RAGResponse:
        """
        Process a query using RAG to generate contextual answers.
        
        Args:
            question: User's question
            book_id: Optional book ID filter
            context_limit: Maximum number of context documents to retrieve
            model: Optional LLM model to use
            
        Returns:
            RAGResponse: Complete response with answer, sources, and metadata
        """
        if not self.vector_db:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        start_time = datetime.now()
        logger.info(f"Processing RAG query: {question[:100]}...")
        
        try:
            # Step 1: Generate query embedding
            embedding_start = datetime.now()
            query_embedding = await self.embedding_service.embed_query(question)
            embedding_time = (datetime.now() - embedding_start).total_seconds()
            
            # Step 2: Perform vector similarity search
            search_start = datetime.now()
            context = await self._retrieve_context(
                question, book_id, context_limit, query_embedding
            )
            search_time = (datetime.now() - search_start).total_seconds()
            
            context.query_embedding_time = embedding_time
            context.vector_search_time = search_time
            
            # Step 3: Generate response using LLM
            response_start = datetime.now()
            answer, token_usage = await self._generate_answer(question, context, model)
            response_time = (datetime.now() - response_start).total_seconds()
            
            # Step 4: Format sources for response
            sources = self._format_sources(context)
            
            # Step 5: Calculate confidence score
            confidence_score = self._calculate_confidence(context, token_usage)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            response = RAGResponse(
                answer=answer,
                sources=sources,
                context_used=context,
                token_usage=token_usage,
                processing_time=total_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"RAG query completed in {total_time:.2f}s: "
                       f"{context.total_retrieved} docs retrieved, "
                       f"{token_usage['total_tokens']} tokens used")
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise ValueError(f"Failed to process query: {e}")
    
    async def _retrieve_context(
        self,
        question: str,
        book_id: Optional[str],
        context_limit: int,
        query_embedding: Optional[List[float]] = None
    ) -> QueryContext:
        """
        Retrieve relevant context documents using vector similarity search.
        
        Args:
            question: User's question
            book_id: Optional book ID filter
            context_limit: Maximum number of documents to retrieve
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            QueryContext: Retrieved context information
        """
        try:
            # Build metadata filter
            where_clause = {}
            if book_id:
                where_clause["book_id"] = book_id
            
            # Use default collection name from settings
            collection_name = self.settings.chroma_collection_name
            
            # Perform vector search
            results = await self.vector_db.query(
                collection_name=collection_name,
                query_text=question,
                n_results=context_limit,
                where=where_clause if where_clause else None
            )
            
            context = QueryContext(
                documents=results["documents"],
                metadatas=results["metadatas"],
                distances=results["distances"],
                total_retrieved=len(results["documents"])
            )
            
            logger.info(f"Retrieved {context.total_retrieved} context documents")
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            # Return empty context on failure
            return QueryContext()
    
    async def _generate_answer(
        self,
        question: str,
        context: QueryContext,
        model: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context information
            model: Optional LLM model to use
            
        Returns:
            Tuple[str, Dict[str, int]]: Generated answer and token usage
        """
        if not context.documents:
            # No context available, provide general response
            prompt = f"""
You are a helpful reading companion AI. The user asked: "{question}"

I don't have access to specific book content to answer this question. Please provide a helpful general response that:
1. Acknowledges that you need more context
2. Suggests how the user might find the information they're looking for
3. Offers general guidance if applicable

Keep your response concise and helpful.
"""
        else:
            # Build context-aware prompt
            context_text = ""
            for i, (doc, metadata) in enumerate(zip(context.documents, context.metadatas)):
                source_info = ""
                if metadata:
                    if "book_title" in metadata:
                        source_info += f"From '{metadata['book_title']}'"
                    if "page" in metadata:
                        source_info += f" (page {metadata['page']})"
                    if "chapter" in metadata:
                        source_info += f" - Chapter: {metadata['chapter']}"
                
                context_text += f"\n--- Context {i+1} {source_info} ---\n{doc}\n"
            
            prompt = f"""
You are a helpful reading companion AI. Answer the user's question based on the provided context from their books.

User Question: {question}

Context from books:
{context_text}

Instructions:
1. Answer the question directly and concisely based on the context provided
2. If the context doesn't fully answer the question, acknowledge what you can answer and what you cannot
3. Use specific references to book content when relevant
4. If multiple books are referenced, clearly distinguish between sources
5. Keep your response focused and helpful for a reader

Answer:
"""
        
        try:
            answer, token_usage = await self.llm_provider.generate_response(
                prompt=prompt,
                model=model,
                temperature=0.1,
                max_tokens=800
            )
            
            return answer.strip(), token_usage
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response to your question. Please try again.", {}
    
    def _format_sources(self, context: QueryContext) -> List[Dict[str, Any]]:
        """
        Format retrieved context as source citations.
        
        Args:
            context: Retrieved context information
            
        Returns:
            List[Dict[str, Any]]: Formatted source information
        """
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            context.documents, context.metadatas, context.distances
        )):
            source = {
                "id": i + 1,
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "similarity_score": 1.0 - distance if distance else 0.0,
                "metadata": metadata or {}
            }
            
            # Add readable source description
            source_desc = "Unknown source"
            if metadata:
                if "book_title" in metadata:
                    source_desc = metadata["book_title"]
                    if "page" in metadata:
                        source_desc += f", page {metadata['page']}"
                    if "chapter" in metadata:
                        source_desc += f" (Chapter: {metadata['chapter']})"
            
            source["source_description"] = source_desc
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(
        self,
        context: QueryContext,
        token_usage: Dict[str, int]
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            context: Retrieved context information
            token_usage: Token usage statistics
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not context.documents:
            return 0.1  # Low confidence without context
        
        # Base confidence on similarity scores
        if context.distances:
            avg_similarity = sum(1.0 - d for d in context.distances) / len(context.distances)
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost slightly but cap at 1.0
        else:
            confidence = 0.5  # Medium confidence if no distance info
        
        # Adjust based on number of retrieved documents
        if context.total_retrieved >= 3:
            confidence *= 1.1  # Boost for multiple sources
        elif context.total_retrieved == 1:
            confidence *= 0.9  # Slight penalty for single source
        
        return min(confidence, 1.0)
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for maintaining conversation flow.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List[Dict[str, Any]]: Recent conversation messages
        """
        try:
            # Import conversation service
            from src.services.conversation_service import get_conversation_service
            
            # Get conversation service and retrieve context
            conversation_service = await get_conversation_service()
            context = await conversation_service.get_conversation_context(
                conversation_id=conversation_id,
                limit=limit
            )
            
            logger.info(f"Retrieved {len(context)} context messages for conversation: {conversation_id}")
            return context
            
        except Exception as e:
            logger.warning(f"Failed to retrieve conversation context for {conversation_id}: {e}")
            # Return empty context if retrieval fails
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RAG service components.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        status = {
            "rag_service": "healthy",
            "vector_database": "unknown",
            "embedding_service": "unknown", 
            "llm_provider": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check vector database
            if self.vector_db:
                collection_info = await self.vector_db.get_collection_info(
                    self.settings.chroma_collection_name
                )
                if collection_info:
                    status["vector_database"] = "healthy"
                    status["document_count"] = collection_info.get("document_count", 0)
                else:
                    status["vector_database"] = "error"
            else:
                status["vector_database"] = "not_initialized"
            
            # Check embedding service cache stats
            cache_stats = self.embedding_service.get_cache_stats()
            if cache_stats:
                status["embedding_service"] = "healthy"
                status["embedding_cache"] = cache_stats
            else:
                status["embedding_service"] = "healthy_no_cache"
            
            # Check LLM provider (simple test)
            status["llm_provider"] = "healthy"  # Assume healthy if no errors during init
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            status["rag_service"] = "error"
            status["error"] = str(e)
        
        return status


# Global RAG service instance
_rag_service: Optional[RAGService] = None


async def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance with lazy initialization.
    
    Returns:
        RAGService: Initialized RAG service instance
    """
    global _rag_service
    
    if _rag_service is None:
        _rag_service = RAGService()
        if not await _rag_service.initialize():
            raise RuntimeError("Failed to initialize RAG service")
    
    return _rag_service