"""
BM25 Keyword Search Implementation for DBC Platform.

This module provides BM25-based keyword search functionality with multiple matching
strategies to complement vector similarity search in the hybrid retrieval pipeline.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import time
from difflib import SequenceMatcher

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    stopwords = None
    word_tokenize = None
    PorterStemmer = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BM25Document:
    """Document representation for BM25 indexing."""
    doc_id: str
    content: str
    tokens: List[str] = field(default_factory=list)
    token_counts: Dict[str, int] = field(default_factory=dict)
    length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25SearchResult:
    """Single search result from BM25 algorithm."""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    matched_terms: List[str]
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25SearchResults:
    """Complete search results from BM25 search."""
    query: str
    results: List[BM25SearchResult]
    total_docs_searched: int
    search_time: float
    strategy_used: str
    query_tokens: List[str] = field(default_factory=list)


class TextProcessor:
    """Text processing utilities for BM25 search."""
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        """
        Initialize text processor.
        
        Args:
            use_stemming: Whether to use stemming for text normalization
            remove_stopwords: Whether to remove common stopwords
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer() if use_stemming else None
            self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        else:
            self.stemmer = None
            # Basic English stopwords if NLTK not available
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'would', 'i', 'you', 'we', 'they',
                'this', 'these', 'those', 'have', 'had', 'been', 'being'
            } if remove_stopwords else set()
        
        logger.info(f"Text processor initialized: stemming={use_stemming}, "
                   f"stopwords={remove_stopwords}, nltk_available={NLTK_AVAILABLE}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = text.lower()
        
        # Use NLTK tokenizer if available, otherwise use simple regex
        if NLTK_AVAILABLE and word_tokenize:
            try:
                tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed, falling back to regex: {e}")
                tokens = re.findall(r'\b\w+\b', text)
        else:
            # Simple regex-based tokenization
            tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.use_stemming and self.stemmer:
            try:
                tokens = [self.stemmer.stem(token) for token in tokens]
            except Exception as e:
                logger.warning(f"Stemming failed: {e}")
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def process_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> BM25Document:
        """
        Process a document for BM25 indexing.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Additional document metadata
            
        Returns:
            BM25Document: Processed document ready for indexing
        """
        tokens = self.tokenize_text(content)
        token_counts = Counter(tokens)
        
        return BM25Document(
            doc_id=doc_id,
            content=content,
            tokens=tokens,
            token_counts=token_counts,
            length=len(tokens),
            metadata=metadata or {}
        )


class MatchingStrategy(ABC):
    """Abstract base class for different matching strategies."""
    
    @abstractmethod
    def match_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> Tuple[List[str], float]:
        """
        Match query terms against document tokens.
        
        Args:
            query_tokens: Tokenized query
            doc_tokens: Tokenized document
            
        Returns:
            Tuple[List[str], float]: Matched terms and matching confidence
        """
        pass


class ExactMatchStrategy(MatchingStrategy):
    """Exact term matching strategy."""
    
    def match_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> Tuple[List[str], float]:
        """Match exact terms between query and document."""
        doc_token_set = set(doc_tokens)
        matched_terms = [token for token in query_tokens if token in doc_token_set]
        confidence = len(matched_terms) / len(query_tokens) if query_tokens else 0.0
        return matched_terms, confidence


class FuzzyMatchStrategy(MatchingStrategy):
    """Fuzzy matching strategy for handling typos and variations."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize fuzzy matching.
        
        Args:
            similarity_threshold: Minimum similarity score for a match (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
    
    def match_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> Tuple[List[str], float]:
        """Match terms using fuzzy string similarity."""
        matched_terms = []
        doc_token_set = set(doc_tokens)
        
        for query_token in query_tokens:
            # First try exact match
            if query_token in doc_token_set:
                matched_terms.append(query_token)
                continue
            
            # Then try fuzzy matching
            best_match = None
            best_similarity = 0.0
            
            for doc_token in doc_token_set:
                similarity = SequenceMatcher(None, query_token, doc_token).ratio()
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = doc_token
            
            if best_match:
                matched_terms.append(query_token)  # Keep original query term
        
        confidence = len(matched_terms) / len(query_tokens) if query_tokens else 0.0
        return matched_terms, confidence


class NGramMatchStrategy(MatchingStrategy):
    """N-gram based matching for partial word matches."""
    
    def __init__(self, n: int = 3, min_overlap: float = 0.6):
        """
        Initialize n-gram matching.
        
        Args:
            n: N-gram size
            min_overlap: Minimum overlap ratio for a match
        """
        self.n = n
        self.min_overlap = min_overlap
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract n-grams from text."""
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def match_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> Tuple[List[str], float]:
        """Match terms using n-gram overlap."""
        matched_terms = []
        doc_token_set = set(doc_tokens)
        
        for query_token in query_tokens:
            # First try exact match
            if query_token in doc_token_set:
                matched_terms.append(query_token)
                continue
            
            # Try n-gram matching
            query_ngrams = self._get_ngrams(query_token, self.n)
            best_match = None
            best_overlap = 0.0
            
            for doc_token in doc_token_set:
                doc_ngrams = self._get_ngrams(doc_token, self.n)
                overlap = len(query_ngrams & doc_ngrams) / len(query_ngrams | doc_ngrams)
                
                if overlap > best_overlap and overlap >= self.min_overlap:
                    best_overlap = overlap
                    best_match = doc_token
            
            if best_match:
                matched_terms.append(query_token)
        
        confidence = len(matched_terms) / len(query_tokens) if query_tokens else 0.0
        return matched_terms, confidence


class SynonymExpansionStrategy(MatchingStrategy):
    """Synonym-based term expansion (placeholder for future implementation)."""
    
    def __init__(self):
        """Initialize synonym expansion."""
        # TODO: Integrate with WordNet or custom synonym database
        self.synonyms = {
            'book': ['novel', 'text', 'volume', 'publication'],
            'read': ['study', 'peruse', 'scan', 'review'],
            'write': ['author', 'compose', 'create', 'pen'],
            'learn': ['study', 'understand', 'master', 'comprehend']
        }
    
    def match_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> Tuple[List[str], float]:
        """Match terms including synonyms."""
        matched_terms = []
        doc_token_set = set(doc_tokens)
        
        for query_token in query_tokens:
            # Try exact match first
            if query_token in doc_token_set:
                matched_terms.append(query_token)
                continue
            
            # Try synonym matching
            synonyms = self.synonyms.get(query_token, [])
            for synonym in synonyms:
                if synonym in doc_token_set:
                    matched_terms.append(query_token)
                    break
        
        confidence = len(matched_terms) / len(query_tokens) if query_tokens else 0.0
        return matched_terms, confidence


class BM25Index:
    """
    BM25 search index with multiple matching strategies.
    
    Implements the BM25 ranking function with support for different term matching
    strategies including exact matching, fuzzy matching, n-gram matching, and
    synonym expansion.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = True,
        remove_stopwords: bool = True
    ):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (typical range: 1.2-2.0)
            b: Length normalization parameter (typical range: 0.0-1.0)
            use_stemming: Whether to use stemming for text normalization
            remove_stopwords: Whether to remove common stopwords
        """
        self.k1 = k1
        self.b = b
        
        # Text processing
        self.text_processor = TextProcessor(use_stemming, remove_stopwords)
        
        # Index storage
        self.documents: Dict[str, BM25Document] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.total_documents = 0
        self.average_doc_length = 0.0
        
        # Matching strategies
        self.matching_strategies = {
            'exact': ExactMatchStrategy(),
            'fuzzy': FuzzyMatchStrategy(similarity_threshold=0.8),
            'ngram': NGramMatchStrategy(n=3, min_overlap=0.6),
            'synonym': SynonymExpansionStrategy()
        }
        
        logger.info(f"BM25 index initialized: k1={k1}, b={b}")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the BM25 index.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to index
            metadata: Additional document metadata
        """
        if not content or not content.strip():
            logger.warning(f"Skipping empty document: {doc_id}")
            return
        
        # Process document
        document = self.text_processor.process_document(doc_id, content, metadata)
        
        # Remove existing document if present
        if doc_id in self.documents:
            self.remove_document(doc_id)
        
        # Add to index
        self.documents[doc_id] = document
        
        # Update term frequencies
        for term, count in document.token_counts.items():
            self.term_frequencies[term][doc_id] = count
            if count > 0:  # Document contains this term
                self.document_frequencies[term] += 1
        
        # Update statistics
        self.total_documents = len(self.documents)
        total_length = sum(doc.length for doc in self.documents.values())
        self.average_doc_length = total_length / self.total_documents if self.total_documents > 0 else 0.0
        
        logger.debug(f"Added document to BM25 index: {doc_id} ({document.length} tokens)")
    
    def remove_document(self, doc_id: str):
        """
        Remove a document from the BM25 index.
        
        Args:
            doc_id: Document identifier to remove
        """
        if doc_id not in self.documents:
            logger.warning(f"Document not found in index: {doc_id}")
            return
        
        document = self.documents[doc_id]
        
        # Update document frequencies
        for term in document.token_counts:
            if self.term_frequencies[term][doc_id] > 0:
                self.document_frequencies[term] -= 1
                if self.document_frequencies[term] <= 0:
                    del self.document_frequencies[term]
            del self.term_frequencies[term][doc_id]
            
            # Clean up empty term entries
            if not self.term_frequencies[term]:
                del self.term_frequencies[term]
        
        # Remove document
        del self.documents[doc_id]
        
        # Update statistics
        self.total_documents = len(self.documents)
        total_length = sum(doc.length for doc in self.documents.values())
        self.average_doc_length = total_length / self.total_documents if self.total_documents > 0 else 0.0
        
        logger.debug(f"Removed document from BM25 index: {doc_id}")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        strategy: str = 'exact',
        min_score: float = 0.0
    ) -> BM25SearchResults:
        """
        Search the BM25 index for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            strategy: Matching strategy to use ('exact', 'fuzzy', 'ngram', 'synonym')
            min_score: Minimum BM25 score threshold
            
        Returns:
            BM25SearchResults: Search results with scores and metadata
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return BM25SearchResults(
                query=query,
                results=[],
                total_docs_searched=0,
                search_time=time.time() - start_time,
                strategy_used=strategy
            )
        
        if self.total_documents == 0:
            logger.warning("No documents in BM25 index")
            return BM25SearchResults(
                query=query,
                results=[],
                total_docs_searched=0,
                search_time=time.time() - start_time,
                strategy_used=strategy
            )
        
        # Tokenize query
        query_tokens = self.text_processor.tokenize_text(query)
        if not query_tokens:
            return BM25SearchResults(
                query=query,
                results=[],
                total_docs_searched=0,
                search_time=time.time() - start_time,
                strategy_used=strategy,
                query_tokens=query_tokens
            )
        
        # Get matching strategy
        matching_strategy = self.matching_strategies.get(strategy, self.matching_strategies['exact'])
        
        # Calculate BM25 scores for all documents
        document_scores = []
        
        for doc_id, document in self.documents.items():
            score = self._calculate_bm25_score(query_tokens, document, matching_strategy)
            
            if score > min_score:
                # Get matched terms for explanation
                matched_terms, confidence = matching_strategy.match_terms(query_tokens, document.tokens)
                
                result = BM25SearchResult(
                    doc_id=doc_id,
                    score=score,
                    content=document.content,
                    metadata=document.metadata,
                    matched_terms=matched_terms,
                    explanation={
                        'bm25_score': score,
                        'matched_terms': matched_terms,
                        'matching_confidence': confidence,
                        'document_length': document.length,
                        'strategy_used': strategy
                    }
                )
                document_scores.append(result)
        
        # Sort by score (descending) and limit results
        document_scores.sort(key=lambda x: x.score, reverse=True)
        results = document_scores[:limit]
        
        search_time = time.time() - start_time
        
        logger.info(f"BM25 search completed: '{query}' -> {len(results)} results "
                   f"(strategy: {strategy}, time: {search_time:.3f}s)")
        
        return BM25SearchResults(
            query=query,
            results=results,
            total_docs_searched=self.total_documents,
            search_time=search_time,
            strategy_used=strategy,
            query_tokens=query_tokens
        )
    
    def _calculate_bm25_score(
        self,
        query_tokens: List[str],
        document: BM25Document,
        matching_strategy: MatchingStrategy
    ) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query_tokens: Tokenized query
            document: Document to score
            matching_strategy: Strategy for term matching
            
        Returns:
            float: BM25 score
        """
        if not query_tokens or document.length == 0:
            return 0.0
        
        score = 0.0
        
        # Get matched terms using the specified strategy
        matched_terms, _ = matching_strategy.match_terms(query_tokens, document.tokens)
        
        for term in matched_terms:
            # Get term frequency in document
            tf = document.token_counts.get(term, 0)
            
            if tf == 0:
                continue
            
            # Get document frequency for term
            df = self.document_frequencies.get(term, 0)
            
            if df == 0:
                continue
            
            # Calculate IDF (Inverse Document Frequency)
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # Calculate TF component with BM25 normalization
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (document.length / self.average_doc_length))
            )
            
            # Add to total score
            score += idf * tf_component
        
        return max(score, 0.0)  # Ensure non-negative score
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        total_terms = len(self.term_frequencies)
        total_tokens = sum(doc.length for doc in self.documents.values())
        
        return {
            'total_documents': self.total_documents,
            'total_unique_terms': total_terms,
            'total_tokens': total_tokens,
            'average_doc_length': self.average_doc_length,
            'k1_parameter': self.k1,
            'b_parameter': self.b,
            'text_processing': {
                'stemming_enabled': self.text_processor.use_stemming,
                'stopwords_enabled': self.text_processor.remove_stopwords,
                'nltk_available': NLTK_AVAILABLE
            },
            'matching_strategies': list(self.matching_strategies.keys())
        }
    
    async def build_index_async(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """
        Build index asynchronously from a list of documents.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
        """
        logger.info(f"Building BM25 index with {len(documents)} documents...")
        
        for doc_id, content, metadata in documents:
            self.add_document(doc_id, content, metadata)
            
            # Yield control periodically to avoid blocking
            if len(self.documents) % 100 == 0:
                await asyncio.sleep(0.001)
        
        logger.info(f"BM25 index built successfully: {self.total_documents} documents indexed")


class BM25SearchEngine:
    """
    High-level BM25 search engine with automatic index management.
    
    Provides a convenient interface for BM25 search with automatic indexing
    and support for multiple search strategies.
    """
    
    def __init__(self, **index_params):
        """
        Initialize BM25 search engine.
        
        Args:
            **index_params: Parameters to pass to BM25Index constructor
        """
        self.index = BM25Index(**index_params)
        self.indexed_documents: Set[str] = set()
        
        logger.info("BM25 search engine initialized")
    
    async def index_documents(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """
        Index a batch of documents.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
        """
        new_documents = []
        
        for doc_id, content, metadata in documents:
            if doc_id not in self.indexed_documents:
                new_documents.append((doc_id, content, metadata))
                self.indexed_documents.add(doc_id)
        
        if new_documents:
            await self.index.build_index_async(new_documents)
            logger.info(f"Indexed {len(new_documents)} new documents")
    
    async def search_with_fallback(
        self,
        query: str,
        limit: int = 10,
        strategies: List[str] = None,
        min_score: float = 0.0
    ) -> BM25SearchResults:
        """
        Search with fallback strategies if no results found.
        
        Args:
            query: Search query
            limit: Maximum number of results
            strategies: List of strategies to try in order
            min_score: Minimum score threshold
            
        Returns:
            BM25SearchResults: Best search results found
        """
        strategies = strategies or ['exact', 'fuzzy', 'ngram', 'synonym']
        
        for strategy in strategies:
            results = self.index.search(
                query=query,
                limit=limit,
                strategy=strategy,
                min_score=min_score
            )
            
            if results.results:
                logger.info(f"BM25 search successful with strategy: {strategy}")
                return results
            
            logger.debug(f"No results with strategy {strategy}, trying next...")
        
        # Return empty results with the last strategy tried
        return BM25SearchResults(
            query=query,
            results=[],
            total_docs_searched=self.index.total_documents,
            search_time=0.0,
            strategy_used=strategies[-1] if strategies else 'exact'
        )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        stats = self.index.get_index_stats()
        stats['indexed_document_ids'] = len(self.indexed_documents)
        return stats