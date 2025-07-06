"""
Enhanced embedding service for the multi-module Alexandria platform.

This service integrates the enhanced vector database, content management,
and AI-powered embedding generation to support permission-aware search,
content relationships, and cross-module recommendations.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import uuid
import json
from dataclasses import asdict

from src.models import (
    ContentItem, EmbeddingMetadata, User, ContentRelationship,
    ModuleType, ContentType, ContentVisibility, UserRole,
    ContentRelationshipType, ProcessingStatus
)
from src.utils.enhanced_database import get_enhanced_database, EnhancedVectorDatabaseInterface
from src.utils.embeddings import EmbeddingService, EmbeddingMetrics
from src.utils.enhanced_chunking import EnhancedSemanticChunker, EnhancedChunkMetadata
from src.services.content_service import get_content_service, ContentService
from src.utils.config import get_settings, DEFAULT_COLLECTION_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedEmbeddingService:
    """
    Enhanced embedding service supporting multi-module content with 
    permission-aware search and AI-powered content relationships.
    
    This service coordinates between:
    - Content management (content_service)
    - Enhanced chunking (enhanced_chunking)
    - Embedding generation (embeddings)
    - Vector database (enhanced_database)
    """
    
    def __init__(self):
        """Initialize the enhanced embedding service."""
        self.settings = get_settings()
        
        # Service dependencies - initialized lazily
        self._embedding_service: Optional[EmbeddingService] = None
        self._chunking_service: Optional[EnhancedSemanticChunker] = None
        self._vector_db: Optional[EnhancedVectorDatabaseInterface] = None
        self._content_service: Optional[ContentService] = None
        
        # Processing caches
        self._semantic_tag_cache: Dict[str, List[str]] = {}
        self._relationship_discovery_cache: Dict[str, List[ContentRelationship]] = {}
        
        # Performance metrics
        self._processing_metrics = {
            "total_content_processed": 0,
            "total_embeddings_created": 0,
            "total_relationships_discovered": 0,
            "average_processing_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize all service dependencies.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize core services
            self._embedding_service = EmbeddingService(use_cache=True)
            self._chunking_service = EnhancedSemanticChunker()
            self._vector_db = await get_enhanced_database()
            self._content_service = await get_content_service()
            
            logger.info("Enhanced embedding service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced embedding service: {e}")
            return False
    
    # ========================================
    # Content Processing and Embedding
    # ========================================
    
    async def process_content_item(
        self,
        content: ContentItem,
        user: Optional[User] = None,
        force_reprocess: bool = False
    ) -> bool:
        """
        Process a content item with enhanced metadata and relationships.
        
        Args:
            content: Content item to process
            user: User performing the processing (for permissions)
            force_reprocess: Whether to reprocess if already processed
            
        Returns:
            bool: True if processing successful
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing content item: {content.content_id} ({content.title})")
            
            # Check if already processed
            if content.processing_status == ProcessingStatus.COMPLETED and not force_reprocess:
                logger.info(f"Content {content.content_id} already processed")
                return True
            
            # Check permissions
            if user and not user.can_access_content(content):
                logger.warning(f"User {user.user_id} lacks permission to process {content.content_id}")
                return False
            
            # Update processing status
            content.processing_status = ProcessingStatus.PROCESSING
            await self._content_service.update_content_item(content)
            
            # Read content file
            if not content.file_path:
                raise ValueError("Content item has no file path")
            
            # Enhanced chunking with semantic analysis
            chunks = await self._chunking_service.chunk_content(
                file_path=content.file_path,
                content_type=content.content_type,
                chunking_strategy="semantic"
            )
            
            if not chunks:
                raise ValueError("No chunks generated from content")
            
            # Generate semantic tags for the content
            semantic_tags = await self._extract_semantic_tags(chunks, content)
            
            # Generate embeddings with enhanced metadata
            embeddings_created = await self._create_enhanced_embeddings(
                content=content,
                chunks=chunks,
                semantic_tags=semantic_tags,
                user=user
            )
            
            # Discover content relationships
            relationships_created = await self._discover_content_relationships(
                content=content,
                chunks=chunks,
                user=user
            )
            
            # Update content item status
            content.processing_status = ProcessingStatus.COMPLETED
            content.text_length = sum(len(chunk.text) for chunk in chunks)
            content.chunk_count = len(chunks)
            content.topics = semantic_tags
            content.mark_processed()
            
            await self._content_service.update_content_item(content)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_metrics(1, embeddings_created, relationships_created, processing_time)
            
            logger.info(f"Content processing completed: {content.content_id}, "
                       f"{embeddings_created} embeddings, {relationships_created} relationships, "
                       f"{processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Content processing failed for {content.content_id}: {e}")
            
            # Mark as failed
            content.processing_status = ProcessingStatus.FAILED
            await self._content_service.update_content_item(content)
            
            return False
    
    async def _create_enhanced_embeddings(
        self,
        content: ContentItem,
        chunks: List[EnhancedChunkMetadata],
        semantic_tags: List[str],
        user: Optional[User] = None
    ) -> int:
        """
        Create enhanced embeddings with multi-module metadata.
        
        Args:
            content: Content item being processed
            chunks: Text chunks with metadata
            semantic_tags: Extracted semantic tags
            user: User performing the processing
            
        Returns:
            int: Number of embeddings created
        """
        try:
            # Prepare texts for embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks using model: {self._embedding_service.default_model}")
            embeddings, metrics = await self._embedding_service.embed_documents(
                documents=[type('Document', (), {'page_content': text})() for text in texts]
            )
            logger.info(f"Embeddings generated - dimensions: {len(embeddings[0]) if embeddings else 0}")
            
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding count mismatch with chunk count")
            
            # Create enhanced metadata for each embedding
            embedding_metadata_list = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if not embedding:  # Skip empty embeddings
                    continue
                
                embedding_metadata = EmbeddingMetadata(
                    embedding_id=str(uuid.uuid4()),
                    content_id=content.content_id,
                    chunk_index=i,
                    module_type=content.module_type,
                    content_type=content.content_type,
                    chunk_type=chunk.chunk_type,
                    visibility=content.visibility,
                    creator_id=content.created_by,
                    organization_id=content.organization_id,
                    semantic_tags=semantic_tags,
                    language=content.language,
                    reading_level=content.reading_level,
                    source_location=chunk.source_location,
                    text_content=chunk.text,
                    chunk_length=len(chunk.text),
                    embedding_model=self._embedding_service.default_model,
                    embedding_dimension=len(embedding),
                    importance_score=chunk.importance_score,
                    quality_score=chunk.quality_score
                )
                embedding_metadata_list.append(embedding_metadata)
            
            # Store embeddings in vector database
            success = await self._vector_db.add_documents_with_metadata(
                collection_name=DEFAULT_COLLECTION_NAME,
                documents=texts[:len(embedding_metadata_list)],
                embeddings=embeddings[:len(embedding_metadata_list)],
                embedding_metadata=embedding_metadata_list
            )
            
            if not success:
                raise ValueError("Failed to store embeddings in vector database")
            
            logger.info(f"Created {len(embedding_metadata_list)} enhanced embeddings for {content.content_id}")
            return len(embedding_metadata_list)
            
        except Exception as e:
            logger.error(f"Failed to create enhanced embeddings: {e}")
            return 0
    
    async def _extract_semantic_tags(
        self,
        chunks: List[EnhancedChunkMetadata],
        content: ContentItem
    ) -> List[str]:
        """
        Extract semantic tags from content chunks using AI analysis.
        
        Args:
            chunks: Text chunks to analyze
            content: Content item being processed
            
        Returns:
            List[str]: Extracted semantic tags
        """
        try:
            # Check cache first
            cache_key = f"{content.content_id}:{len(chunks)}"
            if cache_key in self._semantic_tag_cache:
                return self._semantic_tag_cache[cache_key]
            
            # Combine chunk texts for analysis
            combined_text = " ".join([chunk.text for chunk in chunks[:5]])  # Use first 5 chunks
            if len(combined_text) > 2000:
                combined_text = combined_text[:2000]  # Limit for analysis
            
            # Use OpenAI to extract semantic tags
            from openai import OpenAI
            client = OpenAI(api_key=self.settings.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract 5-10 semantic tags from the given text. Return only the tags as a JSON array. 
                        Tags should be:
                        - Single words or short phrases
                        - Relevant to the main topics and themes
                        - Useful for content discovery and recommendations
                        - Professional and appropriate for all audiences"""
                    },
                    {
                        "role": "user",
                        "content": f"Content Title: {content.title}\n\nContent Type: {content.content_type}\n\nText:\n{combined_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            try:
                semantic_tags = json.loads(response_text)
                if isinstance(semantic_tags, list):
                    # Clean and validate tags
                    semantic_tags = [tag.strip().lower() for tag in semantic_tags if tag.strip()]
                    semantic_tags = list(set(semantic_tags))  # Remove duplicates
                    
                    # Cache the result
                    self._semantic_tag_cache[cache_key] = semantic_tags
                    
                    logger.info(f"Extracted {len(semantic_tags)} semantic tags for {content.content_id}")
                    return semantic_tags
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse semantic tags JSON: {response_text}")
            
            # Fallback: use basic keyword extraction
            semantic_tags = self._extract_basic_keywords(combined_text)
            self._semantic_tag_cache[cache_key] = semantic_tags
            
            return semantic_tags
            
        except Exception as e:
            logger.error(f"Failed to extract semantic tags: {e}")
            # Return basic fallback tags
            return [content.content_type.value, "general", "content"]
    
    def _extract_basic_keywords(self, text: str) -> List[str]:
        """Extract basic keywords as fallback for semantic tag extraction."""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(8)]
        
        return keywords
    
    async def _discover_content_relationships(
        self,
        content: ContentItem,
        chunks: List[EnhancedChunkMetadata],
        user: Optional[User] = None
    ) -> int:
        """
        Discover relationships between this content and existing content.
        
        Args:
            content: Content item being processed
            chunks: Text chunks for analysis
            user: User performing the processing
            
        Returns:
            int: Number of relationships discovered
        """
        try:
            # Check cache first
            cache_key = content.content_id
            if cache_key in self._relationship_discovery_cache:
                relationships = self._relationship_discovery_cache[cache_key]
                return len(relationships)
            
            # Get similar content using vector search
            query_text = " ".join([chunk.text for chunk in chunks[:3]])  # Use first 3 chunks
            if len(query_text) > 1000:
                query_text = query_text[:1000]
            
            similar_results = await self._vector_db.query_with_permissions(
                collection_name=DEFAULT_COLLECTION_NAME,
                query_text=query_text,
                n_results=10,
                user=user,
                additional_filters={"content_id": {"$ne": content.content_id}}  # Exclude self
            )
            
            relationships_created = 0
            discovered_relationships = []
            
            for i, metadata in enumerate(similar_results["metadatas"]):
                if i >= 5:  # Limit to top 5 similar content items
                    break
                
                similar_content_id = metadata.get("content_id")
                if not similar_content_id or similar_content_id == content.content_id:
                    continue
                
                # Calculate relationship strength based on similarity
                distance = similar_results["distances"][i]
                similarity_score = 1.0 - distance  # Convert distance to similarity
                
                # Create relationship based on similarity threshold
                if similarity_score > 0.7:  # High similarity
                    relationship_type = ContentRelationshipType.SIMILARITY
                elif similarity_score > 0.5:  # Medium similarity
                    relationship_type = ContentRelationshipType.REFERENCE
                else:
                    continue  # Skip low similarity
                
                # Create content relationship
                relationship = ContentRelationship(
                    source_content_id=content.content_id,
                    target_content_id=similar_content_id,
                    relationship_type=relationship_type,
                    strength=similarity_score,
                    confidence=0.8,  # AI-discovered with high confidence
                    discovered_by="ai_embedding_similarity",
                    context=f"Discovered via vector similarity (score: {similarity_score:.3f})"
                )
                
                # Store relationship
                if await self._content_service.create_relationship(relationship):
                    relationships_created += 1
                    discovered_relationships.append(relationship)
            
            # Cache discovered relationships
            self._relationship_discovery_cache[cache_key] = discovered_relationships
            
            logger.info(f"Discovered {relationships_created} relationships for {content.content_id}")
            return relationships_created
            
        except Exception as e:
            logger.error(f"Failed to discover content relationships: {e}")
            return 0
    
    # ========================================
    # Enhanced Search and Retrieval
    # ========================================
    
    async def enhanced_search(
        self,
        query: str,
        user: Optional[User] = None,
        module_filter: Optional[ModuleType] = None,
        content_type_filter: Optional[ContentType] = None,
        n_results: int = 10,
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Perform enhanced search with permission filtering and relationship awareness.
        
        Args:
            query: Search query
            user: User performing the search
            module_filter: Optional module filter
            content_type_filter: Optional content type filter
            n_results: Number of results to return
            include_relationships: Whether to include content relationships
            
        Returns:
            Dict[str, Any]: Enhanced search results
        """
        try:
            logger.info(f"Enhanced search: '{query}' for user {user.user_id if user else 'anonymous'}")
            logger.info(f"Search parameters: n_results={n_results}, module_filter={module_filter}, content_type_filter={content_type_filter}, include_relationships={include_relationships}")
            
            # Basic permission-aware search
            search_results = await self._vector_db.query_with_permissions(
                collection_name=DEFAULT_COLLECTION_NAME,
                query_text=query,
                n_results=n_results * 2,  # Get more results for relationship processing
                user=user,
                module_filter=module_filter,
                content_type_filter=content_type_filter
            )
            
            logger.info(f"Vector DB search returned {len(search_results['documents'])} documents")
            if search_results['documents']:
                logger.info(f"First result preview: {search_results['documents'][0][:100]}...")
            
            # Log embedding model being used
            if self._embedding_service:
                logger.info(f"Using embedding model: {self._embedding_service.default_model}")
            
            if not search_results["documents"]:
                logger.warning(f"No documents found for query: '{query}'")
                return {
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                    "ids": [],
                    "enhanced_results": [],
                    "relationship_boost": []
                }
            
            # Get content relationships if requested
            if include_relationships:
                # Get content IDs from results
                content_ids = [meta.get("content_id") for meta in search_results["metadatas"]]
                content_ids = list(set([cid for cid in content_ids if cid]))
                
                # Get relationships for these content items
                all_relationships = []
                for content_id in content_ids:
                    relationships = await self._content_service.get_content_relationships(content_id)
                    all_relationships.extend(relationships)
                
                # Apply relationship-aware search
                enhanced_results = await self._vector_db.similarity_search_with_relationships(
                    collection_name=DEFAULT_COLLECTION_NAME,
                    query_text=query,
                    content_relationships=all_relationships,
                    n_results=n_results,
                    relationship_boost=0.15
                )
                
                # Add relationship information to results
                enhanced_results["enhanced_results"] = []
                for i, metadata in enumerate(enhanced_results["metadatas"]):
                    content_id = metadata.get("content_id")
                    
                    # Get content item details
                    content_item = await self._content_service.get_content_item(content_id, user)
                    
                    enhanced_result = {
                        "content_id": content_id,
                        "title": content_item.title if content_item else "Unknown",
                        "author": content_item.author if content_item else None,
                        "content_type": metadata.get("content_type"),
                        "module_type": metadata.get("module_type"),
                        "chunk_type": metadata.get("chunk_type"),
                        "semantic_tags": json.loads(metadata.get("semantic_tags", "[]")),
                        "source_location": json.loads(metadata.get("source_location", "{}")),
                        "importance_score": metadata.get("importance_score"),
                        "quality_score": metadata.get("quality_score"),
                        "similarity_score": 1.0 - enhanced_results["distances"][i],
                        "relationship_score": enhanced_results.get("relationship_scores", [0.0])[i] if i < len(enhanced_results.get("relationship_scores", [])) else 0.0
                    }
                    enhanced_results["enhanced_results"].append(enhanced_result)
                
                return enhanced_results
            else:
                # Return basic results without relationship processing
                basic_results = {
                    "documents": search_results["documents"][:n_results],
                    "metadatas": search_results["metadatas"][:n_results],
                    "distances": search_results["distances"][:n_results],
                    "ids": search_results["ids"][:n_results],
                    "enhanced_results": [],
                    "relationship_boost": []
                }
                
                return basic_results
                
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
                "enhanced_results": [],
                "relationship_boost": []
            }
    
    async def get_content_recommendations(
        self,
        content_id: str,
        user: Optional[User] = None,
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get content recommendations based on content relationships and similarity.
        
        Args:
            content_id: Content ID to get recommendations for
            user: User requesting recommendations
            n_recommendations: Number of recommendations to return
            
        Returns:
            List[Dict[str, Any]]: Content recommendations
        """
        try:
            # Get the content item
            content = await self._content_service.get_content_item(content_id, user)
            if not content:
                return []
            
            # Get content relationships
            relationships = await self._content_service.get_content_relationships(content_id)
            
            # Get related content via relationships
            related_content_ids = []
            for rel in relationships:
                if rel.source_content_id == content_id:
                    related_content_ids.append((rel.target_content_id, rel.strength))
                elif rel.bidirectional and rel.target_content_id == content_id:
                    related_content_ids.append((rel.source_content_id, rel.strength))
            
            # Sort by relationship strength
            related_content_ids.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            
            # Add relationship-based recommendations
            for related_id, strength in related_content_ids[:n_recommendations]:
                related_content = await self._content_service.get_content_item(related_id, user)
                if related_content:
                    recommendations.append({
                        "content_id": related_id,
                        "title": related_content.title,
                        "author": related_content.author,
                        "content_type": related_content.content_type.value,
                        "module_type": related_content.module_type.value,
                        "recommendation_score": strength,
                        "recommendation_type": "relationship",
                        "reason": f"Related content (strength: {strength:.2f})"
                    })
            
            # If we need more recommendations, use similarity search
            if len(recommendations) < n_recommendations:
                # Get content embeddings for similarity search
                content_embeddings = await self._vector_db.get_content_embeddings(content_id)
                if content_embeddings:
                    # Use first embedding for similarity search
                    first_embedding = content_embeddings[0]
                    
                    similar_results = await self._vector_db.query_with_permissions(
                        collection_name=DEFAULT_COLLECTION_NAME,
                        query_text=first_embedding.text_content,
                        n_results=n_recommendations * 2,
                        user=user,
                        additional_filters={"content_id": {"$ne": content_id}}
                    )
                    
                    # Add similarity-based recommendations
                    for i, metadata in enumerate(similar_results["metadatas"]):
                        if len(recommendations) >= n_recommendations:
                            break
                        
                        similar_content_id = metadata.get("content_id")
                        if similar_content_id and similar_content_id not in [r["content_id"] for r in recommendations]:
                            similar_content = await self._content_service.get_content_item(similar_content_id, user)
                            if similar_content:
                                similarity_score = 1.0 - similar_results["distances"][i]
                                recommendations.append({
                                    "content_id": similar_content_id,
                                    "title": similar_content.title,
                                    "author": similar_content.author,
                                    "content_type": similar_content.content_type.value,
                                    "module_type": similar_content.module_type.value,
                                    "recommendation_score": similarity_score,
                                    "recommendation_type": "similarity",
                                    "reason": f"Similar content (score: {similarity_score:.2f})"
                                })
            
            logger.info(f"Generated {len(recommendations)} recommendations for {content_id}")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get content recommendations: {e}")
            return []
    
    # ========================================
    # Performance and Metrics
    # ========================================
    
    def _update_processing_metrics(
        self,
        content_count: int,
        embeddings_count: int,
        relationships_count: int,
        processing_time: float
    ):
        """Update processing metrics."""
        self._processing_metrics["total_content_processed"] += content_count
        self._processing_metrics["total_embeddings_created"] += embeddings_count
        self._processing_metrics["total_relationships_discovered"] += relationships_count
        
        # Update average processing time
        total_processed = self._processing_metrics["total_content_processed"]
        current_avg = self._processing_metrics["average_processing_time"]
        self._processing_metrics["average_processing_time"] = (
            (current_avg * (total_processed - content_count) + processing_time) / total_processed
        )
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        return {
            **self._processing_metrics,
            "cache_stats": {
                "semantic_tags_cached": len(self._semantic_tag_cache),
                "relationships_cached": len(self._relationship_discovery_cache)
            },
            "embedding_cache_stats": self._embedding_service.get_cache_stats() if self._embedding_service else None
        }
    
    async def cleanup_caches(self):
        """Clean up internal caches to free memory."""
        self._semantic_tag_cache.clear()
        self._relationship_discovery_cache.clear()
        logger.info("Enhanced embedding service caches cleaned")


# ========================================
# Global Service Instance
# ========================================

_enhanced_embedding_service: Optional[EnhancedEmbeddingService] = None


async def get_enhanced_embedding_service() -> EnhancedEmbeddingService:
    """
    Get the global enhanced embedding service instance with lazy initialization.
    
    Returns:
        EnhancedEmbeddingService: Initialized enhanced embedding service instance
    """
    global _enhanced_embedding_service
    
    if _enhanced_embedding_service is None:
        _enhanced_embedding_service = EnhancedEmbeddingService()
        if not await _enhanced_embedding_service.initialize():
            raise RuntimeError("Failed to initialize enhanced embedding service")
    
    return _enhanced_embedding_service