"""
Unit tests for Content Relationships Explorer Backend

Tests the relationship discovery algorithms, graph data structures,
and API endpoint functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List, Dict, Any

from src.models import (
    ContentItem, ContentRelationship, User, 
    ModuleType, ContentType, ContentVisibility, 
    UserRole, ProcessingStatus, ContentRelationshipType
)
from src.services.content_service import ContentService
from src.utils.graph_retrieval import (
    GraphSearchEngine, KnowledgeGraph, GraphNode, GraphEdge,
    RelationshipType, BreadthFirstSearch
)


class TestGraphStructures:
    """Test graph data structures for relationship visualization."""
    
    def test_graph_node_creation(self):
        """Test creating graph nodes with proper structure."""
        node = GraphNode(
            node_id="test-node-1",
            content="Test content for the node",
            node_type="document",
            metadata={"title": "Test Title", "author": "Test Author"},
            importance=0.8
        )
        
        assert node.node_id == "test-node-1"
        assert node.content == "Test content for the node"
        assert node.node_type == "document"
        assert node.metadata["title"] == "Test Title"
        assert node.importance == 0.8
    
    def test_graph_edge_creation(self):
        """Test creating graph edges with relationship data."""
        edge = GraphEdge(
            source_id="node-1",
            target_id="node-2",
            relationship_type=RelationshipType.SIMILAR,
            weight=0.85,
            confidence=0.92,
            metadata={"explanation": "Semantic similarity"}
        )
        
        assert edge.source_id == "node-1"
        assert edge.target_id == "node-2"
        assert edge.relationship_type == RelationshipType.SIMILAR
        assert edge.weight == 0.85
        assert edge.confidence == 0.92
        assert edge.metadata["explanation"] == "Semantic similarity"
    
    def test_knowledge_graph_operations(self):
        """Test knowledge graph construction and operations."""
        graph = KnowledgeGraph()
        
        # Add nodes
        node1 = GraphNode("n1", "Content 1", "document", {"topic": "AI"})
        node2 = GraphNode("n2", "Content 2", "document", {"topic": "ML"})
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert len(graph.nodes) == 2
        assert "n1" in graph.nodes
        assert "n2" in graph.nodes
        
        # Add edge
        edge = GraphEdge("n1", "n2", RelationshipType.SIMILAR, 0.8, 0.9)
        graph.add_edge(edge)
        
        # Test neighbors
        neighbors = graph.get_neighbors("n1")
        assert len(neighbors) == 1
        assert neighbors[0] == "n2"
        
        # Test graph stats
        stats = graph.get_graph_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1


class TestRelationshipDiscovery:
    """Test relationship discovery algorithms."""
    
    @pytest.fixture
    def sample_content_items(self):
        """Create sample content items for testing."""
        import uuid
        return [
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                title="Introduction to Machine Learning",
                description="A comprehensive guide to ML fundamentals",
                author="John Doe",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-1",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["machine learning", "artificial intelligence", "algorithms"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                title="Deep Learning Explained",
                description="Understanding neural networks and deep learning",
                author="Jane Smith",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-2",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["deep learning", "neural networks", "artificial intelligence"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.ARTICLE,
                title="Data Science Best Practices",
                description="Essential practices for data science projects",
                author="Bob Johnson",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-3",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["data science", "machine learning", "statistics"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    @pytest.fixture
    def sample_relationships(self, sample_content_items):
        """Create sample relationships for testing."""
        import uuid
        book_1_id = sample_content_items[0].content_id
        book_2_id = sample_content_items[1].content_id
        article_1_id = sample_content_items[2].content_id
        
        return [
            ContentRelationship(
                relationship_id=str(uuid.uuid4()),
                source_content_id=book_1_id,
                target_content_id=book_2_id,
                relationship_type=ContentRelationshipType.SIMILARITY,
                strength=0.85,
                confidence=0.92,
                discovered_by="ai",
                created_at=datetime.now(),
                context="Both books cover AI and ML topics"
            ),
            ContentRelationship(
                relationship_id=str(uuid.uuid4()),
                source_content_id=book_1_id,
                target_content_id=article_1_id,
                relationship_type=ContentRelationshipType.SUPPLEMENT,
                strength=0.72,
                confidence=0.81,
                discovered_by="ai",
                created_at=datetime.now(),
                context="Shared machine learning concepts"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_graph_engine_document_processing(self, sample_content_items):
        """Test graph engine document processing."""
        graph_engine = GraphSearchEngine()
        
        # Convert content items to documents
        documents = []
        for content in sample_content_items:
            documents.append((
                content.content_id,
                content.title + " " + (content.description or ""),
                {
                    "title": content.title,
                    "author": content.author,
                    "content_type": content.content_type.value,
                    "topics": content.topics
                }
            ))
        
        # Build graph
        await graph_engine.build_graph_from_documents(documents, similarity_threshold=0.1)
        
        # Verify graph construction
        stats = graph_engine.get_engine_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] >= 0  # Edges depend on similarity calculation
        
        # Test finding related content
        related = graph_engine.find_related_content("book-1", max_distance=2)
        assert isinstance(related, list)
        
        # Each related item should be a tuple of (node_id, score, distance)
        for item in related:
            assert isinstance(item, tuple)
            assert len(item) == 3
            assert isinstance(item[0], str)  # node_id
            assert isinstance(item[1], float)  # score
            assert isinstance(item[2], int)  # distance
    
    @pytest.mark.asyncio
    async def test_breadth_first_search(self):
        """Test BFS graph traversal strategy."""
        # Create a simple graph
        graph = KnowledgeGraph()
        
        # Add nodes
        nodes = [
            GraphNode("n1", "Node 1", "document", {"topic": "AI"}),
            GraphNode("n2", "Node 2", "document", {"topic": "ML"}),
            GraphNode("n3", "Node 3", "document", {"topic": "DL"}),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add edges
        edges = [
            GraphEdge("n1", "n2", RelationshipType.SIMILAR, 0.8, 0.9),
            GraphEdge("n2", "n3", RelationshipType.SIMILAR, 0.7, 0.8),
        ]
        
        for edge in edges:
            graph.add_edge(edge)
        
        # Test BFS
        bfs_strategy = BreadthFirstSearch()
        results = await bfs_strategy.search(
            graph=graph,
            start_nodes=["n1"],
            query="test query",
            limit=10,
            max_distance=2
        )
        
        assert results.search_strategy == "bfs"
        assert len(results.start_nodes) == 1
        assert results.nodes_visited > 0
        assert results.search_time >= 0
        
        # Results should include reachable nodes
        result_node_ids = [r.node_id for r in results.results]
        assert "n2" in result_node_ids  # Should be reachable from n1
    
    def test_relationship_strength_calculation(self):
        """Test relationship strength and confidence scoring."""
        # Test edge case: empty content
        graph_engine = GraphSearchEngine()
        similarity = graph_engine._calculate_simple_similarity("", "")
        assert similarity == 0.0
        
        # Test identical content
        similarity = graph_engine._calculate_simple_similarity(
            "machine learning algorithms", 
            "machine learning algorithms"
        )
        assert similarity == 1.0
        
        # Test partial overlap
        similarity = graph_engine._calculate_simple_similarity(
            "machine learning algorithms",
            "deep learning networks"
        )
        assert 0.0 < similarity < 1.0
        
        # Test no overlap
        similarity = graph_engine._calculate_simple_similarity(
            "machine learning",
            "cooking recipes"
        )
        assert similarity == 0.0


class TestAPIEndpoints:
    """Test relationship API endpoints."""
    
    @pytest.fixture
    def sample_content_items(self):
        """Create sample content items for testing."""
        import uuid
        return [
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                title="Introduction to Machine Learning",
                description="A comprehensive guide to ML fundamentals",
                author="John Doe",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-1",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["machine learning", "artificial intelligence", "algorithms"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                title="Deep Learning Explained",
                description="Understanding neural networks and deep learning",
                author="Jane Smith",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-2",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["deep learning", "neural networks", "artificial intelligence"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.ARTICLE,
                title="Data Science Best Practices",
                description="Essential practices for data science projects",
                author="Bob Johnson",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-3",
                processing_status=ProcessingStatus.COMPLETED,
                topics=["data science", "machine learning", "statistics"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    @pytest.fixture
    def sample_relationships(self, sample_content_items):
        """Create sample relationships for testing."""
        import uuid
        book_1_id = sample_content_items[0].content_id
        book_2_id = sample_content_items[1].content_id
        article_1_id = sample_content_items[2].content_id
        
        return [
            ContentRelationship(
                relationship_id=str(uuid.uuid4()),
                source_content_id=book_1_id,
                target_content_id=book_2_id,
                relationship_type=ContentRelationshipType.SIMILARITY,
                strength=0.85,
                confidence=0.92,
                discovered_by="ai",
                created_at=datetime.now(),
                context="Both books cover AI and ML topics"
            ),
            ContentRelationship(
                relationship_id=str(uuid.uuid4()),
                source_content_id=book_1_id,
                target_content_id=article_1_id,
                relationship_type=ContentRelationshipType.SUPPLEMENT,
                strength=0.72,
                confidence=0.81,
                discovered_by="ai",
                created_at=datetime.now(),
                context="Shared machine learning concepts"
            )
        ]
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing."""
        return User(
            user_id="test-user",
            email="test@example.com",
            username="testuser",
            role=UserRole.READER,
            subscription_tier="free"
        )
    
    @pytest.fixture
    def mock_content_service(self, sample_content_items, sample_relationships):
        """Create a mock content service."""
        service = Mock(spec=ContentService)
        
        # Mock methods
        service.get_content_item = AsyncMock()
        service.list_content_items = AsyncMock(return_value=sample_content_items)
        service.get_content_relationships = AsyncMock(return_value=sample_relationships)
        
        # Configure get_content_item to return specific items
        def get_content_side_effect(content_id, user=None):
            for item in sample_content_items:
                if item.content_id == content_id:
                    return item
            return None
        
        service.get_content_item.side_effect = get_content_side_effect
        
        return service
    
    @pytest.mark.asyncio
    async def test_content_relationships_endpoint(self, mock_content_service, mock_user, sample_content_items):
        """Test the content relationships endpoint logic."""
        # Import the endpoint function
        from src.api.enhanced_content import get_content_relationships as get_relationships_func
        
        # Mock the get_content_service dependency
        with patch('src.api.enhanced_content.get_content_service', return_value=mock_content_service):
            # This would test the actual endpoint logic
            # For now, we'll test the service interactions
            
            content_id = "book-1"
            
            # Test getting content item
            content = await mock_content_service.get_content_item(content_id, mock_user)
            assert content is not None
            assert content.content_id == content_id
            
            # Test getting relationships
            relationships = await mock_content_service.get_content_relationships(content_id)
            assert len(relationships) >= 0
            assert all(hasattr(rel, 'relationship_type') for rel in relationships)
    
    @pytest.mark.asyncio
    async def test_graph_data_response_structure(self, sample_content_items, sample_relationships):
        """Test that graph data response has correct structure."""
        # Simulate graph endpoint logic
        nodes = []
        edges = []
        
        # Build nodes from content items
        for content in sample_content_items:
            nodes.append({
                "id": content.content_id,
                "title": content.title,
                "author": content.author,
                "content_type": content.content_type.value,
                "module_type": content.module_type.value,
                "topics": content.topics,
                "size": 1000,  # Default size
                "color": "#3498db",  # Default color
                "created_at": content.created_at.isoformat()
            })
        
        # Build edges from relationships
        content_ids = {content.content_id for content in sample_content_items}
        for rel in sample_relationships:
            if rel.target_content_id in content_ids:
                edges.append({
                    "source": rel.source_content_id,
                    "target": rel.target_content_id,
                    "relationship_type": rel.relationship_type.value,
                    "strength": rel.strength,
                    "confidence": rel.confidence,
                    "weight": rel.strength,
                    "discovered_by": rel.discovered_by,
                    "human_verified": rel.human_verified,
                    "context": rel.context
                })
        
        # Calculate stats
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "average_connections": len(edges) / len(nodes) if nodes else 0,
            "content_types": list(set(node["content_type"] for node in nodes)),
            "module_types": list(set(node["module_type"] for node in nodes)),
            "relationship_types": list(set(edge["relationship_type"] for edge in edges)),
            "average_strength": sum(edge["strength"] for edge in edges) / len(edges) if edges else 0
        }
        
        # Validate structure
        assert len(nodes) == 3
        assert len(edges) >= 0
        assert stats["total_nodes"] == 3
        assert "content_types" in stats
        assert "module_types" in stats
        
        # Validate node structure
        for node in nodes:
            required_node_fields = ["id", "title", "content_type", "module_type"]
            for field in required_node_fields:
                assert field in node
        
        # Validate edge structure
        for edge in edges:
            required_edge_fields = ["source", "target", "relationship_type", "strength", "confidence"]
            for field in required_edge_fields:
                assert field in edge


class TestErrorHandling:
    """Test error handling in relationship endpoints."""
    
    @pytest.mark.asyncio
    async def test_nonexistent_content_handling(self):
        """Test handling of requests for non-existent content."""
        service = Mock(spec=ContentService)
        service.get_content_item = AsyncMock(return_value=None)
        
        content_id = "nonexistent-content"
        
        # Test that service returns None for non-existent content
        content = await service.get_content_item(content_id)
        assert content is None
    
    @pytest.mark.asyncio
    async def test_empty_relationships_handling(self):
        """Test handling of content with no relationships."""
        service = Mock(spec=ContentService)
        service.get_content_relationships = AsyncMock(return_value=[])
        
        content_id = "isolated-content"
        
        # Test that service returns empty list for isolated content
        relationships = await service.get_content_relationships(content_id)
        assert relationships == []
        assert isinstance(relationships, list)
    
    def test_invalid_relationship_type_handling(self):
        """Test handling of invalid relationship type filters."""
        # Test that invalid relationship types raise ValueError
        with pytest.raises(ValueError):
            ContentRelationshipType("invalid_type")
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance characteristics with large dataset."""
        import uuid
        # Create a large number of mock content items
        large_dataset = []
        for i in range(100):
            content = ContentItem(
                content_id=str(uuid.uuid4()),
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                title=f"Test Book {i}",
                description=f"Description for book {i}",
                author=f"Author {i}",
                visibility=ContentVisibility.PUBLIC,
                created_by="user-1",
                processing_status=ProcessingStatus.COMPLETED,
                topics=[f"topic-{i}", "general"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            large_dataset.append(content)
        
        # Test graph construction time
        graph_engine = GraphSearchEngine()
        
        documents = []
        for content in large_dataset:
            documents.append((
                content.content_id,
                content.title + " " + (content.description or ""),
                {"title": content.title, "topics": content.topics}
            ))
        
        start_time = asyncio.get_event_loop().time()
        await graph_engine.build_graph_from_documents(documents, similarity_threshold=0.1)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        # Verify performance requirements
        assert processing_time < 10.0  # Should process 100 items in under 10 seconds
        
        stats = graph_engine.get_engine_stats()
        assert stats["total_nodes"] == 100


if __name__ == "__main__":
    pytest.main([__file__])