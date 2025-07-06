"""
Chat/Q&A API endpoints for the DBC application.

This module provides endpoints for querying books using RAG (Retrieval-Augmented Generation)
and managing conversation history.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.rag.rag_service import get_rag_service
from src.utils.hybrid_search import get_hybrid_search_engine, HybridSearchConfig, SearchStrategy

logger = get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""
    id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    book_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., min_length=1, max_length=1000)
    book_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None  # Prepared for Phase 2 multi-user support
    context_limit: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    message_id: str
    timestamp: datetime
    token_usage: Optional[Dict[str, int]] = None


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    id: str
    title: str
    message_count: int
    last_message_at: datetime
    book_id: Optional[str] = None
    user_id: Optional[str] = None


class ConversationHistory(BaseModel):
    """Conversation history model."""
    conversation: ConversationSummary
    messages: List[ChatMessage]


class HybridSearchRequest(BaseModel):
    """Hybrid search request model."""
    query: str = Field(..., min_length=1, max_length=1000)
    book_id: Optional[str] = None
    strategy: str = Field(default="auto", pattern="^(vector|bm25|graph|vector_bm25|vector_graph|bm25_graph|all|auto)$")
    fusion_method: str = Field(default="rrf", pattern="^(rrf|weighted|combsum)$")
    max_results: int = Field(default=10, ge=1, le=50)
    max_results_per_strategy: int = Field(default=20, ge=1, le=100)


class HybridSearchResponse(BaseModel):
    """Hybrid search response model."""
    query: str
    results: List[Dict[str, Any]]
    strategy_used: str
    fusion_method: str
    total_time: float
    vector_time: float = 0.0
    bm25_time: float = 0.0
    graph_time: float = 0.0
    fusion_time: float = 0.0
    stats: Dict[str, Any]


@router.post("/chat/query", response_model=QueryResponse)
async def query_books(request: QueryRequest):
    """
    Query books using RAG to get contextual answers.
    
    Args:
        request: Query request with question and optional context
        
    Returns:
        QueryResponse: AI-generated answer with sources
    """
    logger.info(f"Processing query: {request.question[:50]}...")
    
    try:
        # Get RAG service instance
        rag_service = await get_rag_service()
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        # Process query using RAG
        rag_response = await rag_service.query(
            question=request.question,
            book_id=request.book_id,
            context_limit=request.context_limit
        )
        
        # Format response
        response = QueryResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            conversation_id=conversation_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            token_usage=rag_response.token_usage
        )
        
        logger.info(f"Query processed successfully: {message_id}, "
                   f"processing_time={rag_response.processing_time:.2f}s, "
                   f"confidence={rag_response.confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        
        # Generate fallback response
        conversation_id = request.conversation_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        response = QueryResponse(
            answer=f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or contact support if the issue persists.",
            sources=[],
            conversation_id=conversation_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
        
        return response


@router.get("/chat/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    user_id: Optional[str] = None,  # Prepared for Phase 2
    skip: int = 0,
    limit: int = 50
):
    """
    List conversation summaries for a user.
    
    Args:
        user_id: User ID filter (Phase 2)
        skip: Number of conversations to skip
        limit: Maximum number of conversations to return
        
    Returns:
        List[ConversationSummary]: List of conversation summaries
    """
    logger.info(f"Listing conversations: skip={skip}, limit={limit}")
    
    # TODO: Implement actual database query
    # For now, return empty list as placeholder
    
    return []


@router.get("/chat/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """
    Get full conversation history including all messages.
    
    Args:
        conversation_id: Unique conversation identifier
        
    Returns:
        ConversationHistory: Complete conversation with messages
    """
    logger.info(f"Getting conversation: {conversation_id}")
    
    # TODO: Implement actual database query
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: Unique conversation identifier
        
    Returns:
        Dict: Deletion confirmation
    """
    logger.info(f"Deleting conversation: {conversation_id}")
    
    # TODO: Implement actual conversation deletion
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("/chat/conversations/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """
    Clear all messages from a conversation while keeping the conversation record.
    
    Args:
        conversation_id: Unique conversation identifier
        
    Returns:
        Dict: Clear operation confirmation
    """
    logger.info(f"Clearing conversation: {conversation_id}")
    
    # TODO: Implement actual conversation clearing
    return {
        "message": "Conversation cleared successfully",
        "conversation_id": conversation_id,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/chat/health")
async def get_rag_health():
    """
    Get health status of RAG service components.
    
    Returns:
        Dict: Health status information
    """
    logger.info("Checking RAG service health")
    
    try:
        rag_service = await get_rag_service()
        health_status = await rag_service.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {
            "rag_service": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/chat/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search using multiple retrieval strategies.
    
    Args:
        request: Hybrid search request with query and configuration
        
    Returns:
        HybridSearchResponse: Search results from multiple strategies
    """
    logger.info(f"Performing hybrid search: {request.query[:50]}... (strategy: {request.strategy})")
    
    try:
        # Get hybrid search engine
        search_engine = await get_hybrid_search_engine()
        
        # Convert string strategy to enum
        strategy_map = {
            "vector": SearchStrategy.VECTOR_ONLY,
            "bm25": SearchStrategy.BM25_ONLY,
            "graph": SearchStrategy.GRAPH_ONLY,
            "vector_bm25": SearchStrategy.VECTOR_BM25,
            "vector_graph": SearchStrategy.VECTOR_GRAPH,
            "bm25_graph": SearchStrategy.BM25_GRAPH,
            "all": SearchStrategy.ALL_STRATEGIES,
            "auto": SearchStrategy.AUTO
        }
        
        # Create search configuration
        config = HybridSearchConfig(
            strategy=strategy_map.get(request.strategy, SearchStrategy.AUTO),
            fusion_method=request.fusion_method,
            max_results=request.max_results,
            max_results_per_strategy=request.max_results_per_strategy
        )
        
        # Perform hybrid search
        search_results = await search_engine.search(
            query=request.query,
            config=config,
            book_id=request.book_id
        )
        
        # Convert results to response format
        results_data = []
        for result in search_results.results:
            result_data = {
                "doc_id": result.doc_id,
                "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "final_score": result.final_score,
                "metadata": result.metadata,
                "contributing_strategies": result.contributing_strategies,
                "vector_score": result.vector_score,
                "bm25_score": result.bm25_score,
                "graph_score": result.graph_score,
                "fusion_explanation": result.fusion_explanation,
                "original_ranks": result.original_ranks
            }
            results_data.append(result_data)
        
        response = HybridSearchResponse(
            query=search_results.query,
            results=results_data,
            strategy_used=search_results.strategy_used.value,
            fusion_method=search_results.fusion_method,
            total_time=search_results.total_time,
            vector_time=search_results.vector_time,
            bm25_time=search_results.bm25_time,
            graph_time=search_results.graph_time,
            fusion_time=search_results.fusion_time,
            stats=search_results.stats
        )
        
        logger.info(f"Hybrid search completed: {len(results_data)} results "
                   f"(total_time: {search_results.total_time:.3f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.get("/chat/search-suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get search suggestions for partial queries.
    
    Args:
        q: Partial query string
        limit: Maximum number of suggestions
        
    Returns:
        Dict: Search suggestions
    """
    logger.info(f"Getting search suggestions for: {q}")
    
    try:
        search_engine = await get_hybrid_search_engine()
        suggestions = await search_engine.get_search_suggestions(q, limit)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        return {
            "query": q,
            "suggestions": [],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/chat/hybrid-search/stats")
async def get_hybrid_search_stats():
    """
    Get statistics and configuration information for hybrid search.
    
    Returns:
        Dict: Hybrid search engine statistics
    """
    logger.info("Getting hybrid search engine statistics")
    
    try:
        search_engine = await get_hybrid_search_engine()
        stats = search_engine.get_engine_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Getting hybrid search stats failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }