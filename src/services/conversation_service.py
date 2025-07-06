"""
Conversation history service for managing chat conversations and messages.

This service provides database operations for storing and retrieving
conversation history across all platform modules.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from src.utils.logger import get_logger
from src.models import (
    Conversation, 
    ChatMessage, 
    ConversationHistory,
    ConversationStatus,
    MessageRole,
    ModuleType
)

logger = get_logger(__name__)


class ConversationService:
    """
    Service for managing conversation history.
    
    This service handles storage and retrieval of conversations and messages
    with support for all platform modules and user permissions.
    """
    
    def __init__(self):
        """Initialize conversation service."""
        # In-memory storage for Phase 1 (single user)
        # TODO: Replace with proper database in Phase 2
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[ChatMessage]] = {}
        
        logger.info("ConversationService initialized with in-memory storage")
    
    async def create_conversation(
        self,
        content_id: Optional[str] = None,
        module_type: Optional[ModuleType] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            content_id: Associated content ID
            module_type: Module context for the conversation
            user_id: User who owns this conversation
            organization_id: Organization context
            title: Optional conversation title
            settings: Conversation-specific settings
        
        Returns:
            Conversation: The created conversation
        """
        conversation = Conversation(
            content_id=content_id,
            module_type=module_type,
            user_id=user_id,
            organization_id=organization_id,
            title=title or "New Conversation",
            settings=settings or {}
        )
        
        # Store conversation
        self._conversations[conversation.conversation_id] = conversation
        self._messages[conversation.conversation_id] = []
        
        logger.info(f"Created conversation: {conversation.conversation_id}")
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Unique conversation identifier
        
        Returns:
            Optional[Conversation]: The conversation if found
        """
        return self._conversations.get(conversation_id)
    
    async def get_conversation_history(
        self, 
        conversation_id: str,
        include_deleted: bool = False
    ) -> Optional[ConversationHistory]:
        """
        Get complete conversation history including all messages.
        
        Args:
            conversation_id: Unique conversation identifier
            include_deleted: Whether to include deleted messages
        
        Returns:
            Optional[ConversationHistory]: Complete conversation history
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        messages = self._messages.get(conversation_id, [])
        
        # Filter out deleted messages unless requested
        if not include_deleted:
            messages = [msg for msg in messages if not msg.is_deleted]
        
        return ConversationHistory(
            conversation=conversation,
            messages=messages
        )
    
    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        content_id: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Parent conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            content_id: Associated content ID if relevant
            sources: RAG sources for assistant messages
            token_usage: Token usage for this message
            processing_time: Processing time in seconds
            confidence_score: Confidence score for assistant responses
            metadata: Additional message metadata
        
        Returns:
            Optional[ChatMessage]: The created message
        """
        # Check if conversation exists
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            logger.error(f"Conversation not found: {conversation_id}")
            return None
        
        # Create message
        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            content_id=content_id,
            sources=sources or [],
            token_usage=token_usage,
            processing_time=processing_time,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        # Add message to storage
        if conversation_id not in self._messages:
            self._messages[conversation_id] = []
        
        self._messages[conversation_id].append(message)
        
        # Update conversation metadata
        conversation.add_message()
        
        # Generate title from first user message if needed
        if (not conversation.title or conversation.title == "New Conversation") and role == MessageRole.USER:
            conversation.title = conversation.generate_title_from_first_message(content)
        
        logger.info(f"Added {role.value} message to conversation {conversation_id}")
        return message
    
    async def get_recent_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
        status: Optional[ConversationStatus] = None
    ) -> List[Conversation]:
        """
        Get recent conversations for a user.
        
        Args:
            user_id: User ID filter (None for all users in Phase 1)
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip
            status: Filter by conversation status
        
        Returns:
            List[Conversation]: List of conversations
        """
        conversations = list(self._conversations.values())
        
        # Filter by user ID if provided
        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]
        
        # Filter by status if provided
        if status:
            conversations = [c for c in conversations if c.status == status]
        
        # Sort by last message time (most recent first)
        conversations.sort(
            key=lambda c: c.last_message_at or c.created_at,
            reverse=True
        )
        
        # Apply pagination
        return conversations[skip:skip + limit]
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for maintaining conversation flow.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List[Dict[str, Any]]: Recent conversation messages formatted for LLM
        """
        history = await self.get_conversation_history(conversation_id)
        if not history:
            return []
        
        return history.get_context_messages(limit)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: Unique conversation identifier
        
        Returns:
            bool: True if deleted successfully
        """
        if conversation_id not in self._conversations:
            return False
        
        # Mark conversation as deleted (soft delete)
        conversation = self._conversations[conversation_id]
        conversation.status = ConversationStatus.DELETED
        conversation.update_timestamp()
        
        # Mark all messages as deleted
        if conversation_id in self._messages:
            for message in self._messages[conversation_id]:
                message.is_deleted = True
        
        logger.info(f"Deleted conversation: {conversation_id}")
        return True
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear all messages from a conversation while keeping the conversation record.
        
        Args:
            conversation_id: Unique conversation identifier
        
        Returns:
            bool: True if cleared successfully
        """
        if conversation_id not in self._conversations:
            return False
        
        # Mark all messages as deleted
        if conversation_id in self._messages:
            for message in self._messages[conversation_id]:
                message.is_deleted = True
        
        # Reset conversation message count
        conversation = self._conversations[conversation_id]
        conversation.message_count = 0
        conversation.last_message_at = None
        conversation.update_timestamp()
        
        logger.info(f"Cleared conversation: {conversation_id}")
        return True
    
    async def update_message_rating(
        self,
        message_id: str,
        rating: int
    ) -> bool:
        """
        Update quality rating for a message.
        
        Args:
            message_id: Message identifier
            rating: Quality rating (1-5)
        
        Returns:
            bool: True if updated successfully
        """
        # Find message across all conversations
        for messages in self._messages.values():
            for message in messages:
                if message.message_id == message_id:
                    message.set_quality_rating(rating)
                    logger.info(f"Updated rating for message {message_id}: {rating}")
                    return True
        
        return False
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """
        Get conversation service statistics.
        
        Returns:
            Dict[str, Any]: Service statistics
        """
        total_conversations = len(self._conversations)
        active_conversations = sum(
            1 for c in self._conversations.values() 
            if c.status == ConversationStatus.ACTIVE
        )
        total_messages = sum(len(messages) for messages in self._messages.values())
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages,
            "storage_type": "in_memory",
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance for dependency injection
_conversation_service: Optional[ConversationService] = None


async def get_conversation_service() -> ConversationService:
    """
    Get or create the conversation service instance.
    
    Returns:
        ConversationService: The service instance
    """
    global _conversation_service
    
    if _conversation_service is None:
        _conversation_service = ConversationService()
        logger.info("ConversationService singleton created")
    
    return _conversation_service


# Convenience functions for common operations
async def create_conversation_with_first_message(
    question: str,
    content_id: Optional[str] = None,
    module_type: Optional[ModuleType] = None,
    user_id: Optional[str] = None
) -> tuple[Conversation, ChatMessage]:
    """
    Create a new conversation with the first user message.
    
    Args:
        question: First user question
        content_id: Associated content ID
        module_type: Module context
        user_id: User ID
    
    Returns:
        tuple[Conversation, ChatMessage]: Created conversation and message
    """
    service = await get_conversation_service()
    
    # Create conversation
    conversation = await service.create_conversation(
        content_id=content_id,
        module_type=module_type,
        user_id=user_id
    )
    
    # Add first message
    message = await service.add_message(
        conversation_id=conversation.conversation_id,
        role=MessageRole.USER,
        content=question,
        content_id=content_id
    )
    
    return conversation, message


async def add_assistant_response(
    conversation_id: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    token_usage: Optional[Dict[str, int]] = None,
    processing_time: Optional[float] = None,
    confidence_score: Optional[float] = None
) -> Optional[ChatMessage]:
    """
    Add an assistant response to a conversation.
    
    Args:
        conversation_id: Parent conversation ID
        answer: Assistant's answer
        sources: RAG sources
        token_usage: Token usage information
        processing_time: Processing time in seconds
        confidence_score: Confidence score
    
    Returns:
        Optional[ChatMessage]: Created message
    """
    service = await get_conversation_service()
    
    return await service.add_message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer,
        sources=sources,
        token_usage=token_usage,
        processing_time=processing_time,
        confidence_score=confidence_score
    )