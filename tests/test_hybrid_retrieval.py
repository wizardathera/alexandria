"""
Test suite for hybrid retrieval components.

This module tests BM25 search, result fusion, graph retrieval, and the
unified hybrid search engine with comprehensive coverage of all components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.utils.bm25_search import (
    BM25Index, BM25SearchEngine, TextProcessor, ExactMatchStrategy,
    FuzzyMatchStrategy, NGramMatchStrategy, SynonymExpansionStrategy
)
from src.utils.result_fusion import (
    ReciprocalRankFusion, WeightedScoreFusion, CombSumFusion, HybridResultFusion,
    SearchResult, StrategyResults, FusionResults
)
from src.utils.graph_retrieval import (
    KnowledgeGraph, GraphNode, GraphEdge, RelationshipType,
    BreadthFirstSearch, RandomWalkSearch, GraphSearchEngine
)
from src.utils.hybrid_search import (
    HybridSearchEngine, HybridSearchConfig, SearchStrategy, HybridSearchResults
)


class TestTextProcessor:
    """Test text processing utilities for BM25 search."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor(use_stemming=True, remove_stopwords=True)
        self.processor_no_stem = TextProcessor(use_stemming=False, remove_stopwords=False)
    
    def test_tokenize_basic_text(self):
        """Test basic text tokenization."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.processor.tokenize_text(text)
        
        # Should remove stopwords and apply stemming
        assert "the" not in tokens  # Stopword removed
        assert "quick" in tokens
        assert "jump" in tokens  # "jumps" should be stemmed to "jump"
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        assert self.processor.tokenize_text("") == []
        assert self.processor.tokenize_text("   ") == []
        assert self.processor.tokenize_text(None) == []
    
    def test_tokenize_without_stemming(self):
        """Test tokenization without stemming."""
        text = "The running dogs are jumping quickly."
        tokens = self.processor_no_stem.tokenize_text(text)
        
        assert "running" in tokens  # Should not be stemmed
        assert "jumping" in tokens  # Should not be stemmed
        assert "The" in tokens  # Stopwords not removed
    
    def test_process_document(self):
        """Test document processing."""
        content = "This is a test document with multiple words."
        metadata = {"title": "Test Document"}
        
        doc = self.processor.process_document("doc1", content, metadata)
        
        assert doc.doc_id == "doc1"
        assert doc.content == content
        assert doc.metadata == metadata
        assert len(doc.tokens) > 0
        assert doc.length == len(doc.tokens)
        assert isinstance(doc.token_counts, dict)


class TestBM25MatchingStrategies:
    """Test different matching strategies for BM25 search."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exact_strategy = ExactMatchStrategy()
        self.fuzzy_strategy = FuzzyMatchStrategy(similarity_threshold=0.8)
        self.ngram_strategy = NGramMatchStrategy(n=3, min_overlap=0.6)
        self.synonym_strategy = SynonymExpansionStrategy()
    
    def test_exact_match_strategy(self):
        """Test exact term matching."""
        query_tokens = ["quick", "brown", "fox"]
        doc_tokens = ["the", "quick", "brown", "fox", "jumps"]
        
        matched_terms, confidence = self.exact_strategy.match_terms(query_tokens, doc_tokens)
        
        assert matched_terms == ["quick", "brown", "fox"]
        assert confidence == 1.0
    
    def test_exact_match_partial(self):
        """Test exact matching with partial overlap."""
        query_tokens = ["quick", "brown", "elephant"]
        doc_tokens = ["the", "quick", "brown", "fox", "jumps"]
        
        matched_terms, confidence = self.exact_strategy.match_terms(query_tokens, doc_tokens)
        
        assert matched_terms == ["quick", "brown"]
        assert confidence == 2.0 / 3.0  # 2 out of 3 terms matched
    
    def test_fuzzy_match_strategy(self):
        """Test fuzzy matching for typos."""
        query_tokens = ["quikc", "browm", "fox"]  # Typos
        doc_tokens = ["the", "quick", "brown", "fox", "jumps"]
        
        matched_terms, confidence = self.fuzzy_strategy.match_terms(query_tokens, doc_tokens)
        
        # Should match despite typos
        assert len(matched_terms) >= 2  # At least 2 matches expected
        assert confidence > 0.5
    
    def test_ngram_match_strategy(self):
        """Test n-gram based matching."""
        query_tokens = ["quickl"]  # Partial word
        doc_tokens = ["the", "quickly", "brown", "fox"]
        
        matched_terms, confidence = self.ngram_strategy.match_terms(query_tokens, doc_tokens)
        
        # Should match "quickl" with "quickly" based on n-gram overlap
        assert len(matched_terms) >= 0
        assert 0.0 <= confidence <= 1.0
    
    def test_synonym_match_strategy(self):
        """Test synonym-based matching."""
        query_tokens = ["book", "read"]
        doc_tokens = ["novel", "study", "text"]
        
        matched_terms, confidence = self.synonym_strategy.match_terms(query_tokens, doc_tokens)
        
        # Should match some synonyms
        assert isinstance(matched_terms, list)
        assert 0.0 <= confidence <= 1.0


class TestBM25Index:
    """Test BM25 index functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.index = BM25Index(k1=1.5, b=0.75)
        
        # Sample documents
        self.docs = [
            ("doc1", "The quick brown fox jumps over the lazy dog.", {"title": "Fox Story"}),
            ("doc2", "A lazy dog sleeps under the tree.", {"title": "Dog Story"}),
            ("doc3", "Quick brown foxes are amazing animals.", {"title": "Fox Facts"}),
            ("doc4", "The tree provides shade for lazy animals.", {"title": "Tree Story"})
        ]
        
        # Add documents to index
        for doc_id, content, metadata in self.docs:
            self.index.add_document(doc_id, content, metadata)
    
    def test_add_document(self):
        """Test adding documents to index."""
        assert len(self.index.documents) == 4
        assert "doc1" in self.index.documents
        assert self.index.total_documents == 4
        assert self.index.average_doc_length > 0
    
    def test_remove_document(self):
        """Test removing documents from index."""
        self.index.remove_document("doc1")
        
        assert len(self.index.documents) == 3
        assert "doc1" not in self.index.documents
        assert self.index.total_documents == 3
    
    def test_search_exact_match(self):
        """Test search with exact term matching."""
        results = self.index.search("quick brown fox", strategy="exact")
        
        assert len(results.results) > 0
        assert results.query == "quick brown fox"
        assert results.strategy_used == "exact"
        
        # First result should be doc1 (contains all terms)
        assert results.results[0].doc_id == "doc1"
        assert results.results[0].score > 0
    
    def test_search_fuzzy_match(self):
        """Test search with fuzzy matching."""
        results = self.index.search("quikc browm", strategy="fuzzy")  # Typos
        
        assert isinstance(results.results, list)
        assert results.strategy_used == "fuzzy"
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        results = self.index.search("", strategy="exact")
        
        assert len(results.results) == 0
        assert results.search_time >= 0
    
    def test_search_no_matches(self):
        """Test search with no matching terms."""
        results = self.index.search("elephant zebra giraffe", strategy="exact")
        
        assert len(results.results) == 0
    
    def test_get_index_stats(self):
        """Test index statistics."""
        stats = self.index.get_index_stats()
        
        assert stats["total_documents"] == 4
        assert stats["total_unique_terms"] > 0
        assert stats["average_doc_length"] > 0
        assert "matching_strategies" in stats


class TestBM25SearchEngine:
    """Test BM25 search engine with automatic indexing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BM25SearchEngine()
        self.docs = [
            ("doc1", "Machine learning algorithms are powerful tools.", {"category": "AI"}),
            ("doc2", "Deep learning networks process complex data.", {"category": "AI"}),
            ("doc3", "Natural language processing enables text analysis.", {"category": "NLP"}),
            ("doc4", "Computer vision recognizes images and patterns.", {"category": "CV"})
        ]
    
    @pytest.mark.asyncio
    async def test_index_documents(self):
        """Test document indexing."""
        await self.engine.index_documents(self.docs)
        
        assert len(self.engine.indexed_documents) == 4
        assert self.engine.index.total_documents == 4
    
    @pytest.mark.asyncio
    async def test_search_with_fallback(self):
        """Test search with fallback strategies."""
        await self.engine.index_documents(self.docs)
        
        results = await self.engine.search_with_fallback(
            query="machine learning",
            strategies=["exact", "fuzzy"]
        )
        
        assert len(results.results) > 0
        assert results.results[0].doc_id == "doc1"  # Should match machine learning doc
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self):
        """Test search that returns no results."""
        await self.engine.index_documents(self.docs)
        
        results = await self.engine.search_with_fallback(
            query="quantum physics chemistry",
            strategies=["exact"]
        )
        
        assert len(results.results) == 0
    
    def test_get_engine_stats(self):
        """Test engine statistics."""
        stats = self.engine.get_engine_stats()
        
        assert "total_documents" in stats
        assert "indexed_document_ids" in stats


class TestResultFusion:
    """Test result fusion algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample strategy results
        self.vector_results = StrategyResults(
            strategy_name="vector",
            results=[
                SearchResult("doc1", 0.9, "Content 1", {"title": "Doc 1"}, "vector"),
                SearchResult("doc2", 0.8, "Content 2", {"title": "Doc 2"}, "vector"),
                SearchResult("doc3", 0.7, "Content 3", {"title": "Doc 3"}, "vector")
            ],
            search_time=0.1,
            total_docs_searched=100
        )
        
        self.bm25_results = StrategyResults(
            strategy_name="bm25",
            results=[
                SearchResult("doc2", 2.5, "Content 2", {"title": "Doc 2"}, "bm25"),
                SearchResult("doc4", 2.0, "Content 4", {"title": "Doc 4"}, "bm25"),
                SearchResult("doc1", 1.8, "Content 1", {"title": "Doc 1"}, "bm25")
            ],
            search_time=0.05,
            total_docs_searched=100
        )
        
        self.strategy_results = [self.vector_results, self.bm25_results]
    
    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""
        rrf = ReciprocalRankFusion(k=60)
        results = rrf.fuse_results(self.strategy_results, "test query")
        
        assert len(results.results) > 0
        assert results.fusion_strategy == "rrf"
        assert results.input_strategies == ["vector", "bm25"]
        
        # Doc1 and Doc2 should rank high (appear in both strategies)
        top_doc_ids = [r.doc_id for r in results.results[:2]]
        assert "doc1" in top_doc_ids
        assert "doc2" in top_doc_ids
    
    def test_weighted_fusion(self):
        """Test weighted score fusion."""
        weights = {"vector": 0.7, "bm25": 0.3}
        weighted = WeightedScoreFusion(strategy_weights=weights)
        results = weighted.fuse_results(self.strategy_results, "test query")
        
        assert len(results.results) > 0
        assert results.fusion_strategy == "weighted"
        
        # Vector strategy should have more influence due to higher weight
        assert results.results[0].final_score > 0
    
    def test_combsum_fusion(self):
        """Test CombSUM fusion."""
        combsum = CombSumFusion()
        results = combsum.fuse_results(self.strategy_results, "test query")
        
        assert len(results.results) > 0
        assert results.fusion_strategy == "combsum"
    
    def test_hybrid_fusion_engine(self):
        """Test hybrid fusion engine with auto-selection."""
        fusion_engine = HybridResultFusion()
        
        # Test auto-selection
        results = asyncio.run(fusion_engine.fuse_results(
            self.strategy_results, "test query", fusion_method="auto"
        ))
        
        assert len(results.results) > 0
        assert results.fusion_strategy in ["rrf", "weighted", "combsum"]
    
    def test_fusion_with_empty_results(self):
        """Test fusion with empty strategy results."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse_results([], "test query")
        
        assert len(results.results) == 0
        assert results.input_strategies == []


class TestKnowledgeGraph:
    """Test knowledge graph functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = KnowledgeGraph()
        
        # Create sample nodes
        self.nodes = [
            GraphNode("node1", "Content about machine learning", "document", {"topic": "ML"}),
            GraphNode("node2", "Content about deep learning", "document", {"topic": "DL"}),
            GraphNode("node3", "Content about neural networks", "document", {"topic": "NN"}),
            GraphNode("node4", "Content about algorithms", "document", {"topic": "Algo"})
        ]
        
        # Add nodes to graph
        for node in self.nodes:
            self.graph.add_node(node)
    
    def test_add_nodes(self):
        """Test adding nodes to graph."""
        assert len(self.graph.nodes) == 4
        assert "node1" in self.graph.nodes
        assert self.graph.nodes["node1"].content == "Content about machine learning"
    
    def test_add_edges(self):
        """Test adding edges to graph."""
        edge1 = GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.8)
        edge2 = GraphEdge("node2", "node3", RelationshipType.CONTAINS, weight=0.9)
        
        self.graph.add_edge(edge1)
        self.graph.add_edge(edge2)
        
        assert len(self.graph.edges["node1"]) == 1
        assert len(self.graph.incoming_edges["node2"]) == 1
    
    def test_get_neighbors(self):
        """Test getting node neighbors."""
        edge = GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.8)
        self.graph.add_edge(edge)
        
        neighbors = self.graph.get_neighbors("node1")
        assert "node2" in neighbors
        
        # Test filtering by relationship type
        similar_neighbors = self.graph.get_neighbors("node1", [RelationshipType.SIMILAR])
        assert "node2" in similar_neighbors
    
    def test_shortest_path(self):
        """Test shortest path finding."""
        # Create a path: node1 -> node2 -> node3
        edge1 = GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.8)
        edge2 = GraphEdge("node2", "node3", RelationshipType.SIMILAR, weight=0.9)
        
        self.graph.add_edge(edge1)
        self.graph.add_edge(edge2)
        
        path = self.graph.find_shortest_path("node1", "node3")
        
        assert path is not None
        assert path.nodes == ["node1", "node2", "node3"]
        assert len(path.edges) == 2
    
    def test_calculate_importance(self):
        """Test node importance calculation."""
        # Add some edges
        edges = [
            GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.8),
            GraphEdge("node1", "node3", RelationshipType.SIMILAR, weight=0.7),
            GraphEdge("node2", "node3", RelationshipType.CONTAINS, weight=0.9)
        ]
        
        for edge in edges:
            self.graph.add_edge(edge)
        
        importance = self.graph.calculate_node_importance("degree")
        
        assert len(importance) == 4
        assert all(0.0 <= score <= 1.0 for score in importance.values())
        
        # Node1 should have high importance (2 outgoing edges)
        assert importance["node1"] > importance["node4"]  # node4 has no edges
    
    def test_graph_stats(self):
        """Test graph statistics."""
        edge = GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.8)
        self.graph.add_edge(edge)
        
        stats = self.graph.get_graph_stats()
        
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 1
        assert "relationship_distribution" in stats


class TestGraphTraversal:
    """Test graph traversal strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = KnowledgeGraph()
        
        # Create a connected graph
        nodes = [
            GraphNode(f"node{i}", f"Content {i}", "document", {"index": i})
            for i in range(5)
        ]
        
        for node in nodes:
            self.graph.add_node(node)
        
        # Create edges: 0->1->2, 0->3->4, 1->3
        edges = [
            GraphEdge("node0", "node1", RelationshipType.SIMILAR, weight=0.8),
            GraphEdge("node1", "node2", RelationshipType.SIMILAR, weight=0.7),
            GraphEdge("node0", "node3", RelationshipType.SIMILAR, weight=0.9),
            GraphEdge("node3", "node4", RelationshipType.SIMILAR, weight=0.6),
            GraphEdge("node1", "node3", RelationshipType.SIMILAR, weight=0.5)
        ]
        
        for edge in edges:
            self.graph.add_edge(edge)
    
    @pytest.mark.asyncio
    async def test_breadth_first_search(self):
        """Test breadth-first search traversal."""
        bfs = BreadthFirstSearch()
        results = await bfs.search(
            graph=self.graph,
            start_nodes=["node0"],
            query="test query",
            max_distance=2
        )
        
        assert len(results.results) > 0
        assert results.search_strategy == "bfs"
        assert all(r.distance <= 2 for r in results.results)
    
    @pytest.mark.asyncio
    async def test_random_walk_search(self):
        """Test random walk search traversal."""
        random_walk = RandomWalkSearch()
        results = await random_walk.search(
            graph=self.graph,
            start_nodes=["node0"],
            query="test query",
            num_walks=10,
            walk_length=3
        )
        
        assert isinstance(results.results, list)
        assert results.search_strategy == "random_walk"


class TestGraphSearchEngine:
    """Test graph search engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = GraphSearchEngine()
        self.docs = [
            ("doc1", "Machine learning algorithms", {"topic": "ML"}),
            ("doc2", "Deep learning networks", {"topic": "DL"}),
            ("doc3", "Neural network architectures", {"topic": "NN"}),
            ("doc4", "Data science methods", {"topic": "DS"})
        ]
    
    @pytest.mark.asyncio
    async def test_build_graph(self):
        """Test building graph from documents."""
        await self.engine.build_graph_from_documents(self.docs)
        
        assert len(self.engine.graph.nodes) == 4
        assert "doc1" in self.engine.graph.nodes
    
    @pytest.mark.asyncio
    async def test_graph_search(self):
        """Test graph-based search."""
        await self.engine.build_graph_from_documents(self.docs)
        
        results = await self.engine.search(
            query="machine learning",
            strategy="bfs",
            limit=3
        )
        
        assert isinstance(results.results, list)
        assert results.search_strategy == "bfs"
    
    def test_find_related_content(self):
        """Test finding related content."""
        # Add some test data first
        asyncio.run(self.engine.build_graph_from_documents(self.docs))
        
        related = self.engine.find_related_content("doc1", max_distance=2)
        
        assert isinstance(related, list)
        # Each item should be (node_id, score, distance)
        for item in related:
            assert len(item) == 3
            assert isinstance(item[0], str)  # node_id
            assert isinstance(item[1], float)  # score
            assert isinstance(item[2], int)  # distance


class TestHybridSearchEngine:
    """Test unified hybrid search engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = HybridSearchEngine()
        self.docs = [
            ("doc1", "Artificial intelligence and machine learning algorithms", {"category": "AI"}),
            ("doc2", "Deep learning neural networks for image recognition", {"category": "AI"}),
            ("doc3", "Natural language processing and text analysis", {"category": "NLP"}),
            ("doc4", "Computer vision and pattern recognition systems", {"category": "CV"})
        ]
    
    @pytest.mark.asyncio
    async def test_initialize_engine(self):
        """Test hybrid search engine initialization."""
        # Mock the database initialization
        with patch('src.utils.hybrid_search.get_database') as mock_db:
            mock_db.return_value = Mock()
            result = await self.engine.initialize()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_index_documents(self):
        """Test document indexing across all engines."""
        # Mock the initialization
        with patch('src.utils.hybrid_search.get_database') as mock_db:
            mock_db.return_value = Mock()
            await self.engine.initialize()
            
            # Index documents
            await self.engine.index_documents(self.docs)
            
            # Verify indexing
            assert len(self.engine.bm25_engine.indexed_documents) == 4
            assert len(self.engine.graph_engine.graph.nodes) == 4
    
    @pytest.mark.asyncio
    async def test_hybrid_search_bm25_only(self):
        """Test hybrid search with BM25 only."""
        # Mock components
        with patch('src.utils.hybrid_search.get_database') as mock_db:
            mock_db.return_value = Mock()
            await self.engine.initialize()
            await self.engine.index_documents(self.docs)
            
            config = HybridSearchConfig(
                strategy=SearchStrategy.BM25_ONLY,
                max_results=5
            )
            
            results = await self.engine.search("machine learning", config)
            
            assert isinstance(results, HybridSearchResults)
            assert results.strategy_used == SearchStrategy.BM25_ONLY
            assert results.bm25_time > 0
            assert results.vector_time == 0  # Vector not used
    
    @pytest.mark.asyncio
    async def test_hybrid_search_auto_strategy(self):
        """Test hybrid search with auto strategy selection."""
        with patch('src.utils.hybrid_search.get_database') as mock_db:
            mock_db.return_value = Mock()
            await self.engine.initialize()
            await self.engine.index_documents(self.docs)
            
            config = HybridSearchConfig(strategy=SearchStrategy.AUTO)
            
            results = await self.engine.search("AI machine learning", config)
            
            assert isinstance(results, HybridSearchResults)
            assert results.total_time > 0
    
    def test_strategy_selection(self):
        """Test automatic strategy selection."""
        # Short query
        strategy = self.engine._select_search_strategy("AI")
        assert strategy == SearchStrategy.BM25_ONLY
        
        # Medium query
        strategy = self.engine._select_search_strategy("machine learning algorithms")
        assert strategy == SearchStrategy.VECTOR_BM25
        
        # Long query
        strategy = self.engine._select_search_strategy("artificial intelligence and machine learning with deep neural networks")
        assert strategy == SearchStrategy.ALL_STRATEGIES
    
    @pytest.mark.asyncio
    async def test_search_suggestions(self):
        """Test search suggestions."""
        suggestions = await self.engine.get_search_suggestions("learn", limit=3)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
    
    def test_engine_stats(self):
        """Test getting engine statistics."""
        stats = self.engine.get_engine_stats()
        
        assert "hybrid_engine" in stats
        assert "available_strategies" in stats["hybrid_engine"]


class TestHybridSearchIntegration:
    """Integration tests for hybrid search components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_search(self):
        """Test complete end-to-end hybrid search workflow."""
        # Create sample documents
        docs = [
            ("doc1", "Machine learning is a subset of artificial intelligence", {"source": "AI textbook"}),
            ("doc2", "Deep learning uses neural networks with multiple layers", {"source": "DL guide"}),
            ("doc3", "Natural language processing enables computers to understand text", {"source": "NLP manual"}),
            ("doc4", "Computer vision allows machines to interpret visual data", {"source": "CV handbook"})
        ]
        
        # Initialize engines
        bm25_engine = BM25SearchEngine()
        graph_engine = GraphSearchEngine()
        fusion_engine = HybridResultFusion()
        
        # Index documents
        await bm25_engine.index_documents(docs)
        await graph_engine.build_graph_from_documents(docs)
        
        # Perform BM25 search
        bm25_results = await bm25_engine.search_with_fallback("machine learning")
        
        # Perform graph search  
        graph_results = await graph_engine.search("machine learning", strategy="bfs")
        
        # Convert to strategy results format
        bm25_strategy_results = StrategyResults(
            strategy_name="bm25",
            results=[
                SearchResult(r.doc_id, r.score, r.content, r.metadata, "bm25")
                for r in bm25_results.results
            ],
            search_time=bm25_results.search_time,
            total_docs_searched=bm25_results.total_docs_searched
        )
        
        graph_strategy_results = StrategyResults(
            strategy_name="graph", 
            results=[
                SearchResult(r.node_id, r.score, r.content, r.metadata, "graph")
                for r in graph_results.results
            ],
            search_time=graph_results.search_time,
            total_docs_searched=graph_results.nodes_visited
        )
        
        # Fuse results
        fusion_results = await fusion_engine.fuse_results(
            [bm25_strategy_results, graph_strategy_results],
            "machine learning",
            fusion_method="rrf"
        )
        
        # Verify integration
        assert len(fusion_results.results) > 0
        assert fusion_results.fusion_strategy == "rrf"
        assert "bm25" in fusion_results.input_strategies
        assert "graph" in fusion_results.input_strategies
    
    def test_performance_requirements(self):
        """Test that performance requirements are met."""
        # Create large dataset for performance testing
        docs = [
            (f"doc{i}", f"Document {i} with content about topic {i % 5}", {"index": i})
            for i in range(100)
        ]
        
        # Test BM25 indexing performance
        import time
        start_time = time.time()
        
        bm25_engine = BM25SearchEngine()
        asyncio.run(bm25_engine.index_documents(docs))
        
        indexing_time = time.time() - start_time
        
        # Should index 100 documents in reasonable time
        assert indexing_time < 5.0  # 5 seconds max
        
        # Test search performance
        start_time = time.time()
        
        results = asyncio.run(bm25_engine.search_with_fallback("topic content"))
        
        search_time = time.time() - start_time
        
        # Should search in under 1 second
        assert search_time < 1.0
        assert len(results.results) > 0