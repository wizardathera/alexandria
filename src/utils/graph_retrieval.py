"""
Graph Traversal Retrieval Foundation for DBC Platform.

This module implements basic graph-based retrieval capabilities, constructing
simple knowledge graphs from content relationships and providing graph traversal
algorithms for enhanced content discovery.
"""

import math
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import asyncio
import time
import json
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RelationshipType(Enum):
    """Types of relationships between content nodes."""
    SIMILAR = "similar"                 # Semantic similarity
    CONTAINS = "contains"               # Hierarchical containment (chapter -> section)
    REFERENCES = "references"           # One content references another
    FOLLOWS = "follows"                 # Sequential relationship
    CONTRADICTS = "contradicts"         # Opposing viewpoints
    SUPPORTS = "supports"               # Supporting evidence
    PREREQUISITE = "prerequisite"       # Required background knowledge
    RELATED_TOPIC = "related_topic"     # Topic-based relationship


@dataclass
class GraphNode:
    """Node in the knowledge graph representing a content item."""
    node_id: str
    content: str
    node_type: str  # 'document', 'chunk', 'concept', 'entity'
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance: float = 0.0  # Node importance score
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.node_id == other.node_id


@dataclass
class GraphEdge:
    """Edge in the knowledge graph representing a relationship."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0  # Relationship strength (0.0 to 1.0)
    confidence: float = 1.0  # Confidence in relationship (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relationship_type.value))


@dataclass
class GraphPath:
    """Path through the knowledge graph."""
    nodes: List[str]
    edges: List[GraphEdge]
    total_weight: float
    path_score: float
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSearchResult:
    """Result from graph-based search."""
    node_id: str
    content: str
    score: float
    distance: int  # Hop distance from starting nodes
    path: Optional[GraphPath] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSearchResults:
    """Complete results from graph search."""
    query: str
    results: List[GraphSearchResult]
    start_nodes: List[str]
    search_strategy: str
    search_time: float
    nodes_visited: int
    max_distance: int


class KnowledgeGraph:
    """
    Simple knowledge graph for content relationships.
    
    Maintains nodes (content items) and edges (relationships) with support
    for various graph algorithms and traversal strategies.
    """
    
    def __init__(self):
        """Initialize empty knowledge graph."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = defaultdict(list)  # Outgoing edges
        self.incoming_edges: Dict[str, List[GraphEdge]] = defaultdict(list)  # Incoming edges
        self.node_embeddings: Dict[str, List[float]] = {}
        
        logger.info("Knowledge graph initialized")
    
    def add_node(self, node: GraphNode):
        """
        Add a node to the graph.
        
        Args:
            node: GraphNode to add
        """
        self.nodes[node.node_id] = node
        
        if node.embedding:
            self.node_embeddings[node.node_id] = node.embedding
        
        logger.debug(f"Added node to graph: {node.node_id}")
    
    def add_edge(self, edge: GraphEdge):
        """
        Add an edge to the graph.
        
        Args:
            edge: GraphEdge to add
        """
        # Validate that both nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Cannot add edge: missing nodes {edge.source_id} -> {edge.target_id}")
            return
        
        # Add to outgoing edges
        self.edges[edge.source_id].append(edge)
        
        # Add to incoming edges
        self.incoming_edges[edge.target_id].append(edge)
        
        logger.debug(f"Added edge to graph: {edge.source_id} -> {edge.target_id} "
                    f"({edge.relationship_type.value})")
    
    def get_neighbors(self, node_id: str, relationship_types: List[RelationshipType] = None) -> List[str]:
        """
        Get neighboring nodes connected by specified relationship types.
        
        Args:
            node_id: Source node ID
            relationship_types: Optional filter for relationship types
            
        Returns:
            List[str]: List of neighboring node IDs
        """
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        for edge in self.edges[node_id]:
            if relationship_types is None or edge.relationship_type in relationship_types:
                neighbors.append(edge.target_id)
        
        return neighbors
    
    def get_incoming_neighbors(self, node_id: str, relationship_types: List[RelationshipType] = None) -> List[str]:
        """Get nodes that point to this node."""
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        for edge in self.incoming_edges[node_id]:
            if relationship_types is None or edge.relationship_type in relationship_types:
                neighbors.append(edge.source_id)
        
        return neighbors
    
    def get_edge(self, source_id: str, target_id: str, relationship_type: RelationshipType = None) -> Optional[GraphEdge]:
        """Get edge between two nodes."""
        for edge in self.edges.get(source_id, []):
            if (edge.target_id == target_id and 
                (relationship_type is None or edge.relationship_type == relationship_type)):
                return edge
        return None
    
    def calculate_node_importance(self, method: str = 'degree') -> Dict[str, float]:
        """
        Calculate importance scores for all nodes.
        
        Args:
            method: Importance calculation method ('degree', 'pagerank', 'betweenness')
            
        Returns:
            Dict[str, float]: Node importance scores
        """
        if method == 'degree':
            return self._calculate_degree_centrality()
        elif method == 'pagerank':
            return self._calculate_pagerank()
        else:
            logger.warning(f"Unknown importance method: {method}, using degree centrality")
            return self._calculate_degree_centrality()
    
    def _calculate_degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for all nodes."""
        centrality = {}
        
        for node_id in self.nodes:
            out_degree = len(self.edges[node_id])
            in_degree = len(self.incoming_edges[node_id])
            total_degree = out_degree + in_degree
            
            # Normalize by maximum possible degree
            max_degree = (len(self.nodes) - 1) * 2  # Bidirectional
            centrality[node_id] = total_degree / max_degree if max_degree > 0 else 0.0
        
        return centrality
    
    def _calculate_pagerank(self, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
        """Calculate PageRank scores for all nodes."""
        if not self.nodes:
            return {}
        
        # Initialize PageRank values
        num_nodes = len(self.nodes)
        pagerank = {node_id: 1.0 / num_nodes for node_id in self.nodes}
        
        for _ in range(iterations):
            new_pagerank = {}
            
            for node_id in self.nodes:
                # Base PageRank value
                pr_value = (1.0 - damping) / num_nodes
                
                # Add contributions from incoming links
                for edge in self.incoming_edges[node_id]:
                    source_id = edge.source_id
                    out_degree = len(self.edges[source_id])
                    
                    if out_degree > 0:
                        # Weight by edge weight and relationship strength
                        edge_weight = edge.weight * edge.confidence
                        pr_value += damping * pagerank[source_id] * edge_weight / out_degree
                
                new_pagerank[node_id] = pr_value
            
            # Check for convergence
            diff = sum(abs(new_pagerank[node_id] - pagerank[node_id]) for node_id in self.nodes)
            pagerank = new_pagerank
            
            if diff < tolerance:
                break
        
        return pagerank
    
    def find_shortest_path(self, start_id: str, end_id: str, max_hops: int = 5) -> Optional[GraphPath]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_hops: Maximum number of hops to consider
            
        Returns:
            Optional[GraphPath]: Shortest path if found
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        if start_id == end_id:
            return GraphPath(
                nodes=[start_id],
                edges=[],
                total_weight=0.0,
                path_score=1.0
            )
        
        # BFS to find shortest path
        queue = deque([(start_id, [start_id], [], 0.0)])
        visited = set()
        
        while queue:
            current_id, path, edges, total_weight = queue.popleft()
            
            if current_id in visited or len(path) > max_hops:
                continue
            
            visited.add(current_id)
            
            # Check neighbors
            for edge in self.edges[current_id]:
                next_id = edge.target_id
                
                if next_id == end_id:
                    # Found target
                    final_path = path + [next_id]
                    final_edges = edges + [edge]
                    final_weight = total_weight + edge.weight
                    
                    return GraphPath(
                        nodes=final_path,
                        edges=final_edges,
                        total_weight=final_weight,
                        path_score=final_weight / len(final_edges) if final_edges else 0.0
                    )
                
                if next_id not in visited and len(path) < max_hops:
                    queue.append((
                        next_id,
                        path + [next_id],
                        edges + [edge],
                        total_weight + edge.weight
                    ))
        
        return None
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        total_edges = sum(len(edges) for edges in self.edges.values())
        relationship_counts = defaultdict(int)
        
        for edges in self.edges.values():
            for edge in edges:
                relationship_counts[edge.relationship_type.value] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': total_edges,
            'average_degree': total_edges / len(self.nodes) if self.nodes else 0,
            'relationship_distribution': dict(relationship_counts),
            'node_types': list(set(node.node_type for node in self.nodes.values())),
            'has_embeddings': len(self.node_embeddings)
        }


class GraphTraversalStrategy(ABC):
    """Abstract base class for graph traversal strategies."""
    
    @abstractmethod
    async def search(
        self,
        graph: KnowledgeGraph,
        start_nodes: List[str],
        query: str,
        limit: int = 10,
        **kwargs
    ) -> GraphSearchResults:
        """
        Perform graph search starting from given nodes.
        
        Args:
            graph: Knowledge graph to search
            start_nodes: Starting node IDs
            query: Search query
            limit: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            GraphSearchResults: Search results
        """
        pass


class BreadthFirstSearch(GraphTraversalStrategy):
    """Breadth-first search traversal strategy."""
    
    async def search(
        self,
        graph: KnowledgeGraph,
        start_nodes: List[str],
        query: str,
        limit: int = 10,
        max_distance: int = 3,
        relationship_types: List[RelationshipType] = None
    ) -> GraphSearchResults:
        """
        Perform breadth-first search from starting nodes.
        
        Args:
            graph: Knowledge graph to search
            start_nodes: Starting node IDs
            query: Search query (used for scoring)
            limit: Maximum number of results
            max_distance: Maximum hop distance
            relationship_types: Optional filter for relationship types
            
        Returns:
            GraphSearchResults: BFS search results
        """
        start_time = time.time()
        
        # Validate start nodes
        valid_start_nodes = [node_id for node_id in start_nodes if node_id in graph.nodes]
        if not valid_start_nodes:
            return GraphSearchResults(
                query=query,
                results=[],
                start_nodes=start_nodes,
                search_strategy="bfs",
                search_time=time.time() - start_time,
                nodes_visited=0,
                max_distance=max_distance
            )
        
        # BFS traversal
        queue = deque([(node_id, 0, None) for node_id in valid_start_nodes])  # (node_id, distance, path)
        visited = set()
        results = []
        nodes_visited = 0
        
        while queue and len(results) < limit:
            current_id, distance, path = queue.popleft()
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            nodes_visited += 1
            
            # Add current node to results (unless it's a start node)
            if distance > 0:
                node = graph.nodes[current_id]
                score = self._calculate_node_score(node, query, distance)
                
                result = GraphSearchResult(
                    node_id=current_id,
                    content=node.content,
                    score=score,
                    distance=distance,
                    path=path,
                    metadata=node.metadata.copy(),
                    explanation={
                        'strategy': 'bfs',
                        'distance': distance,
                        'base_score': score,
                        'importance': node.importance
                    }
                )
                results.append(result)
            
            # Add neighbors to queue
            neighbors = graph.get_neighbors(current_id, relationship_types)
            for neighbor_id in neighbors:
                if neighbor_id not in visited and distance < max_distance:
                    # Create path information
                    edge = graph.get_edge(current_id, neighbor_id)
                    new_path = None
                    if edge:
                        if path:
                            new_path = GraphPath(
                                nodes=path.nodes + [neighbor_id],
                                edges=path.edges + [edge],
                                total_weight=path.total_weight + edge.weight,
                                path_score=0.0  # Will be calculated later
                            )
                        else:
                            new_path = GraphPath(
                                nodes=[current_id, neighbor_id],
                                edges=[edge],
                                total_weight=edge.weight,
                                path_score=edge.weight
                            )
                    
                    queue.append((neighbor_id, distance + 1, new_path))
        
        # Sort results by score
        results.sort(key=lambda r: r.score, reverse=True)
        final_results = results[:limit]
        
        search_time = time.time() - start_time
        
        return GraphSearchResults(
            query=query,
            results=final_results,
            start_nodes=valid_start_nodes,
            search_strategy="bfs",
            search_time=search_time,
            nodes_visited=nodes_visited,
            max_distance=max_distance
        )
    
    def _calculate_node_score(self, node: GraphNode, query: str, distance: int) -> float:
        """Calculate relevance score for a node."""
        # Simple scoring based on distance and importance
        distance_penalty = 1.0 / (1.0 + distance)
        importance_boost = node.importance
        
        # TODO: Add semantic similarity with query if embeddings available
        base_score = distance_penalty + importance_boost
        
        return min(base_score, 1.0)


class RandomWalkSearch(GraphTraversalStrategy):
    """Random walk search traversal strategy."""
    
    async def search(
        self,
        graph: KnowledgeGraph,
        start_nodes: List[str],
        query: str,
        limit: int = 10,
        num_walks: int = 100,
        walk_length: int = 5,
        restart_probability: float = 0.15
    ) -> GraphSearchResults:
        """
        Perform random walk search from starting nodes.
        
        Args:
            graph: Knowledge graph to search
            start_nodes: Starting node IDs
            query: Search query
            limit: Maximum number of results
            num_walks: Number of random walks to perform
            walk_length: Maximum length of each walk
            restart_probability: Probability of restarting walk at start node
            
        Returns:
            GraphSearchResults: Random walk search results
        """
        start_time = time.time()
        
        valid_start_nodes = [node_id for node_id in start_nodes if node_id in graph.nodes]
        if not valid_start_nodes:
            return GraphSearchResults(
                query=query,
                results=[],
                start_nodes=start_nodes,
                search_strategy="random_walk",
                search_time=time.time() - start_time,
                nodes_visited=0,
                max_distance=walk_length
            )
        
        # Track visit counts for each node
        visit_counts = defaultdict(int)
        nodes_visited = set()
        
        import random
        
        # Perform random walks
        for _ in range(num_walks):
            # Start from random start node
            current_node = random.choice(valid_start_nodes)
            
            for step in range(walk_length):
                visit_counts[current_node] += 1
                nodes_visited.add(current_node)
                
                # Restart probability check
                if random.random() < restart_probability:
                    current_node = random.choice(valid_start_nodes)
                    continue
                
                # Get neighbors
                neighbors = graph.get_neighbors(current_node)
                if not neighbors:
                    break
                
                # Choose next node (weighted by edge weights)
                edges = graph.edges[current_node]
                if edges:
                    weights = [edge.weight * edge.confidence for edge in edges]
                    total_weight = sum(weights)
                    
                    if total_weight > 0:
                        # Weighted random selection
                        rand_val = random.random() * total_weight
                        cumulative = 0
                        
                        for edge, weight in zip(edges, weights):
                            cumulative += weight
                            if rand_val <= cumulative:
                                current_node = edge.target_id
                                break
                    else:
                        current_node = random.choice(neighbors)
                else:
                    current_node = random.choice(neighbors)
        
        # Convert visit counts to results
        results = []
        for node_id, count in visit_counts.items():
            if node_id not in valid_start_nodes:  # Exclude start nodes from results
                node = graph.nodes[node_id]
                
                # Score based on visit frequency and node importance
                visit_score = count / num_walks
                importance_score = node.importance
                combined_score = (visit_score + importance_score) / 2
                
                result = GraphSearchResult(
                    node_id=node_id,
                    content=node.content,
                    score=combined_score,
                    distance=0,  # Not applicable for random walk
                    metadata=node.metadata.copy(),
                    explanation={
                        'strategy': 'random_walk',
                        'visit_count': count,
                        'visit_frequency': visit_score,
                        'importance': importance_score
                    }
                )
                results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        final_results = results[:limit]
        
        search_time = time.time() - start_time
        
        return GraphSearchResults(
            query=query,
            results=final_results,
            start_nodes=valid_start_nodes,
            search_strategy="random_walk",
            search_time=search_time,
            nodes_visited=len(nodes_visited),
            max_distance=walk_length
        )


class GraphSearchEngine:
    """
    High-level graph search engine with multiple traversal strategies.
    
    Provides unified interface for graph-based content discovery with
    automatic relationship detection and various search algorithms.
    """
    
    def __init__(self):
        """Initialize graph search engine."""
        self.graph = KnowledgeGraph()
        self.strategies = {
            'bfs': BreadthFirstSearch(),
            'random_walk': RandomWalkSearch()
        }
        
        logger.info("Graph search engine initialized")
    
    async def build_graph_from_documents(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]],
        similarity_threshold: float = 0.8
    ):
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            similarity_threshold: Minimum similarity for creating edges
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents...")
        
        # Add nodes
        for doc_id, content, metadata in documents:
            node = GraphNode(
                node_id=doc_id,
                content=content,
                node_type="document",
                metadata=metadata or {},
                importance=0.0
            )
            self.graph.add_node(node)
        
        # Create similarity-based edges (simplified for now)
        # TODO: Use actual semantic similarity with embeddings
        nodes = list(self.graph.nodes.values())
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                # Simple word overlap similarity (placeholder)
                similarity = self._calculate_simple_similarity(node_a.content, node_b.content)
                
                if similarity >= similarity_threshold:
                    edge = GraphEdge(
                        source_id=node_a.node_id,
                        target_id=node_b.node_id,
                        relationship_type=RelationshipType.SIMILAR,
                        weight=similarity,
                        confidence=0.7  # Moderate confidence for simple similarity
                    )
                    self.graph.add_edge(edge)
                    
                    # Add reverse edge for bidirectional similarity
                    reverse_edge = GraphEdge(
                        source_id=node_b.node_id,
                        target_id=node_a.node_id,
                        relationship_type=RelationshipType.SIMILAR,
                        weight=similarity,
                        confidence=0.7
                    )
                    self.graph.add_edge(reverse_edge)
        
        # Calculate node importance
        importance_scores = self.graph.calculate_node_importance('degree')
        for node_id, importance in importance_scores.items():
            self.graph.nodes[node_id].importance = importance
        
        logger.info(f"Knowledge graph built: {self.graph.get_graph_stats()}")
    
    def _calculate_simple_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate simple word overlap similarity between two texts."""
        if not text_a or not text_b:
            return 0.0
        
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union) if union else 0.0
    
    async def search(
        self,
        query: str,
        start_nodes: List[str] = None,
        strategy: str = 'bfs',
        limit: int = 10,
        **strategy_params
    ) -> GraphSearchResults:
        """
        Search the knowledge graph using specified strategy.
        
        Args:
            query: Search query
            start_nodes: Starting node IDs (if None, use all nodes)
            strategy: Search strategy ('bfs', 'random_walk')
            limit: Maximum number of results
            **strategy_params: Additional parameters for search strategy
            
        Returns:
            GraphSearchResults: Search results
        """
        if start_nodes is None:
            start_nodes = list(self.graph.nodes.keys())
        
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy}, using BFS")
            strategy = 'bfs'
        
        search_strategy = self.strategies[strategy]
        
        logger.info(f"Performing graph search with {strategy} strategy: "
                   f"query='{query}', start_nodes={len(start_nodes)}")
        
        results = await search_strategy.search(
            graph=self.graph,
            start_nodes=start_nodes,
            query=query,
            limit=limit,
            **strategy_params
        )
        
        return results
    
    def find_related_content(
        self,
        node_id: str,
        relationship_types: List[RelationshipType] = None,
        max_distance: int = 2
    ) -> List[Tuple[str, float, int]]:
        """
        Find content related to a given node.
        
        Args:
            node_id: Source node ID
            relationship_types: Types of relationships to follow
            max_distance: Maximum hop distance
            
        Returns:
            List[Tuple[str, float, int]]: (node_id, score, distance) tuples
        """
        if node_id not in self.graph.nodes:
            return []
        
        related = []
        queue = deque([(node_id, 0, 1.0)])  # (node_id, distance, score)
        visited = set()
        
        while queue:
            current_id, distance, score = queue.popleft()
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            
            if distance > 0:  # Don't include the starting node
                related.append((current_id, score, distance))
            
            # Add neighbors
            neighbors = self.graph.get_neighbors(current_id, relationship_types)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    edge = self.graph.get_edge(current_id, neighbor_id)
                    neighbor_score = score * (edge.weight if edge else 0.5) * 0.8  # Decay with distance
                    queue.append((neighbor_id, distance + 1, neighbor_score))
        
        # Sort by score
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        graph_stats = self.graph.get_graph_stats()
        graph_stats['available_strategies'] = list(self.strategies.keys())
        graph_stats['relationship_types'] = [rt.value for rt in RelationshipType]
        
        return graph_stats