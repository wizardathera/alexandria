"""
Enhanced text chunking utilities for Phase 1.1 of the DBC application.

This module implements advanced semantic chunking strategies with comprehensive
metadata extraction, including chapter-aware, heading-aware, and sentence-level
chunking with importance scoring and overlapping context windows.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    # Fallback for testing without full installation
    Document = dict
    RecursiveCharacterTextSplitter = object
    sent_tokenize = lambda x: x.split('.')
    word_tokenize = lambda x: x.split()

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)


class ChunkType(Enum):
    """Types of content chunks for semantic analysis."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    SUMMARY = "summary"
    INTRODUCTION = "introduction"
    CONCLUSION = "conclusion"


class ContentDifficulty(Enum):
    """Content difficulty levels for adaptive chunking."""
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SourceLocation:
    """Detailed source location information for chunks."""
    page_number: Optional[int] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    paragraph_number: Optional[int] = None
    line_number: Optional[int] = None
    character_start: Optional[int] = None
    character_end: Optional[int] = None
    xpath: Optional[str] = None  # For HTML/XML documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class NamedEntity:
    """Named entity with context and confidence."""
    text: str
    entity_type: str  # 'PERSON', 'ORG', 'GPE', 'DATE', etc.
    confidence: float
    context: str  # Surrounding text for disambiguation
    start_pos: int
    end_pos: int


@dataclass
class Concept:
    """Abstract concept with relationships."""
    name: str
    definition: Optional[str] = None
    confidence: float = 0.0
    related_concepts: List[str] = field(default_factory=list)
    concept_type: str = "general"  # 'technical', 'philosophical', 'scientific', etc.


@dataclass
class EnhancedChunkMetadata:
    """Comprehensive metadata for enhanced chunks."""
    
    # Basic identification
    chunk_id: str
    source_document_id: str
    chunk_index: int
    total_chunks: int
    
    # Content structure
    content_type: ChunkType = ChunkType.PARAGRAPH
    hierarchy_level: int = 0  # 0=document, 1=chapter, 2=section, 3=subsection
    parent_section_id: Optional[str] = None
    child_sections: List[str] = field(default_factory=list)
    
    # Location information
    source_location: SourceLocation = field(default_factory=SourceLocation)
    
    # Content analysis
    topic_tags: List[str] = field(default_factory=list)  # AI-extracted topics
    entities: List[NamedEntity] = field(default_factory=list)  # Named entities
    concepts: List[Concept] = field(default_factory=list)  # Abstract concepts
    reading_level: ContentDifficulty = ContentDifficulty.INTERMEDIATE
    
    # Importance and quality scores
    importance_score: float = 0.5  # 0.0-1.0, content importance
    coherence_score: float = 0.5  # 0.0-1.0, internal coherence
    completeness_score: float = 0.5  # 0.0-1.0, standalone completeness
    information_density: float = 0.5  # 0.0-1.0, information per token
    
    # Relationship information
    related_chunks: List[str] = field(default_factory=list)  # Semantically related chunks
    prerequisite_chunks: List[str] = field(default_factory=list)  # Required background
    follow_up_chunks: List[str] = field(default_factory=list)  # Natural follow-up content
    
    # Context windows
    preceding_context: str = ""  # Text before chunk for context
    following_context: str = ""  # Text after chunk for context
    context_window_size: int = 200  # Size of context windows
    
    # Language and style
    language: str = "en"
    writing_style: str = "formal"  # 'formal', 'casual', 'technical', 'creative'
    tone: str = "informative"  # 'informative', 'persuasive', 'narrative'
    
    # Processing metadata
    chunking_method: str = "enhanced_semantic"
    chunking_strategy: str = "chapter_aware"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Original metadata from LangChain Document
    original_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {}
        for key, value in asdict(self).items():
            if key == "source_location":
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                elif isinstance(value, dict):
                    result[key] = value
                else:
                    result[key] = asdict(value) if value else {}
            elif key in ["entities", "concepts"]:
                result[key] = [asdict(item) for item in value]
            elif key in ["content_type", "reading_level"]:
                result[key] = value.value if hasattr(value, 'value') else str(value)
            elif key in ["created_at", "last_updated"]:
                result[key] = value.isoformat() if isinstance(value, datetime) else value
            else:
                result[key] = value
        return result


@dataclass
class ChunkingConfig:
    """Enhanced configuration for text chunking parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    context_window_size: int = 200
    separator: str = "\n\n"
    keep_separator: bool = True
    is_separator_regex: bool = False
    length_function: str = "len"  # "len" or "tiktoken"
    
    # Enhanced chunking options
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    detect_headings: bool = True
    detect_lists: bool = True
    detect_code_blocks: bool = True
    detect_quotes: bool = True
    
    # Quality thresholds
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    min_coherence_score: float = 0.3
    enable_importance_scoring: bool = True
    enable_context_windows: bool = True


class DocumentStructureAnalyzer:
    """Analyzes document structure to identify headings, sections, and hierarchy."""
    
    def __init__(self):
        """Initialize the document structure analyzer."""
        # Patterns for detecting different content types
        self.heading_patterns = [
            re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),  # Markdown headings
            re.compile(r'^([A-Z][A-Z\s]{2,})\s*$', re.MULTILINE),  # ALL CAPS headings
            re.compile(r'^(\d+\.?\s+.+)$', re.MULTILINE),  # Numbered headings
            re.compile(r'^([IVXLCDM]+\.?\s+.+)$', re.MULTILINE),  # Roman numeral headings
            re.compile(r'^(Chapter\s+\d+.*)$', re.MULTILINE | re.IGNORECASE),  # Chapter headings
        ]
        
        self.list_patterns = [
            re.compile(r'^\s*[-*+]\s+(.+)$', re.MULTILINE),  # Bullet lists
            re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),  # Numbered lists
            re.compile(r'^\s*[a-zA-Z]\.\s+(.+)$', re.MULTILINE),  # Lettered lists
        ]
        
        self.code_patterns = [
            re.compile(r'```[\s\S]*?```', re.MULTILINE),  # Markdown code blocks
            re.compile(r'`[^`\n]+`'),  # Inline code
            re.compile(r'^\s{4,}.+$', re.MULTILINE),  # Indented code
        ]
        
        self.quote_patterns = [
            re.compile(r'^>\s+(.+)$', re.MULTILINE),  # Markdown quotes
            re.compile(r'"([^"]*)"'),  # Quoted text
            re.compile(r'[''"]([^''"""]*)[''"]'),  # Smart quotes
        ]
    
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze document structure and identify content elements.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Dict containing structure analysis results
        """
        structure = {
            "headings": self._detect_headings(text),
            "lists": self._detect_lists(text),
            "code_blocks": self._detect_code_blocks(text),
            "quotes": self._detect_quotes(text),
            "paragraphs": self._detect_paragraphs(text),
            "hierarchy": self._build_hierarchy(text)
        }
        
        logger.debug(f"Document structure analysis: {len(structure['headings'])} headings, "
                    f"{len(structure['paragraphs'])} paragraphs")
        
        return structure
    
    def _detect_headings(self, text: str) -> List[Dict[str, Any]]:
        """Detect headings in the text."""
        headings = []
        
        for pattern in self.heading_patterns:
            for match in pattern.finditer(text):
                heading = {
                    "text": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "level": self._determine_heading_level(match.group(0)),
                    "type": "heading"
                }
                headings.append(heading)
        
        # Sort by position and remove duplicates
        headings.sort(key=lambda x: x["start"])
        return self._remove_overlapping_matches(headings)
    
    def _detect_lists(self, text: str) -> List[Dict[str, Any]]:
        """Detect list items in the text."""
        lists = []
        
        for pattern in self.list_patterns:
            for match in pattern.finditer(text):
                list_item = {
                    "text": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "list_item"
                }
                lists.append(list_item)
        
        return self._remove_overlapping_matches(lists)
    
    def _detect_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Detect code blocks in the text."""
        code_blocks = []
        
        for pattern in self.code_patterns:
            for match in pattern.finditer(text):
                code_block = {
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "code_block"
                }
                code_blocks.append(code_block)
        
        return self._remove_overlapping_matches(code_blocks)
    
    def _detect_quotes(self, text: str) -> List[Dict[str, Any]]:
        """Detect quoted text in the document."""
        quotes = []
        
        for pattern in self.quote_patterns:
            for match in pattern.finditer(text):
                quote = {
                    "text": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "quote"
                }
                quotes.append(quote)
        
        return self._remove_overlapping_matches(quotes)
    
    def _detect_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Detect paragraph boundaries in the text."""
        paragraphs = []
        
        # Split by double newlines to find paragraphs
        parts = re.split(r'\n\s*\n', text)
        current_pos = 0
        
        for part in parts:
            part = part.strip()
            if part:
                start = text.find(part, current_pos)
                if start != -1:
                    paragraph = {
                        "text": part,
                        "start": start,
                        "end": start + len(part),
                        "type": "paragraph"
                    }
                    paragraphs.append(paragraph)
                    current_pos = start + len(part)
        
        return paragraphs
    
    def _build_hierarchy(self, text: str) -> Dict[str, Any]:
        """Build document hierarchy based on headings."""
        headings = self._detect_headings(text)
        
        hierarchy = {
            "chapters": [],
            "sections": [],
            "subsections": [],
            "max_level": 0
        }
        
        for heading in headings:
            level = heading["level"]
            hierarchy["max_level"] = max(hierarchy["max_level"], level)
            
            if level == 1:
                hierarchy["chapters"].append(heading)
            elif level == 2:
                hierarchy["sections"].append(heading)
            elif level >= 3:
                hierarchy["subsections"].append(heading)
        
        return hierarchy
    
    def _determine_heading_level(self, heading_text: str) -> int:
        """Determine the hierarchical level of a heading."""
        # Markdown-style headings
        if heading_text.startswith('#'):
            return len(heading_text) - len(heading_text.lstrip('#'))
        
        # All caps headings are usually top-level
        if heading_text.isupper():
            return 1
        
        # Chapter headings
        if re.match(r'chapter\s+\d+', heading_text.lower()):
            return 1
        
        # Numbered headings
        if re.match(r'\d+\.', heading_text):
            dots = heading_text.count('.')
            return min(dots + 1, 6)
        
        # Default to level 2
        return 2
    
    def _remove_overlapping_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping matches from a list."""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda x: x["start"])
        
        filtered = [matches[0]]
        
        for match in matches[1:]:
            last_match = filtered[-1]
            # If this match doesn't overlap with the last one, add it
            if match["start"] >= last_match["end"]:
                filtered.append(match)
        
        return filtered


class ContentAnalyzer:
    """Analyzes content for semantic properties and importance scoring."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.importance_keywords = {
            "high": ["important", "critical", "essential", "key", "main", "primary", 
                    "significant", "major", "crucial", "fundamental", "core"],
            "medium": ["relevant", "notable", "interesting", "useful", "consider",
                      "remember", "note", "observe", "example", "instance"],
            "low": ["however", "although", "perhaps", "possibly", "might",
                   "could", "sometimes", "occasionally", "briefly"]
        }
        
        self.difficulty_indicators = {
            "elementary": ["simple", "basic", "easy", "introduction", "beginner"],
            "intermediate": ["understand", "learn", "apply", "practice", "example"],
            "advanced": ["complex", "sophisticated", "detailed", "comprehensive"],
            "expert": ["theoretical", "abstract", "specialized", "technical", "advanced"]
        }
    
    def analyze_content(self, text: str, content_type: ChunkType) -> Dict[str, Any]:
        """
        Analyze content for semantic properties.
        
        Args:
            text: Text content to analyze
            content_type: Type of content chunk
            
        Returns:
            Dict containing analysis results
        """
        analysis = {
            "importance_score": self._calculate_importance_score(text),
            "coherence_score": self._calculate_coherence_score(text),
            "completeness_score": self._calculate_completeness_score(text, content_type),
            "information_density": self._calculate_information_density(text),
            "reading_level": self._determine_reading_level(text),
            "topic_tags": self._extract_topic_tags(text),
            "writing_style": self._analyze_writing_style(text),
            "tone": self._analyze_tone(text)
        }
        
        return analysis
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score based on keywords and structure."""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count importance indicators
        high_count = sum(1 for word in self.importance_keywords["high"] if word in text_lower)
        medium_count = sum(1 for word in self.importance_keywords["medium"] if word in text_lower)
        low_count = sum(1 for word in self.importance_keywords["low"] if word in text_lower)
        
        # Calculate weighted score
        importance_score = (high_count * 3 + medium_count * 2 - low_count) / total_words
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, importance_score * 10 + 0.3))
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate how coherent the text is."""
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.8  # Single sentence is considered coherent
        
        # Simple coherence metrics
        coherence_indicators = 0
        total_checks = 0
        
        for i in range(len(sentences) - 1):
            current = sentences[i].lower()
            next_sent = sentences[i + 1].lower()
            
            # Check for transition words
            transition_words = ["however", "therefore", "moreover", "furthermore", 
                              "additionally", "consequently", "thus", "hence"]
            if any(word in next_sent for word in transition_words):
                coherence_indicators += 1
            
            # Check for pronoun references
            pronouns = ["it", "this", "that", "they", "these", "those"]
            if any(pronoun in next_sent.split()[:5] for pronoun in pronouns):
                coherence_indicators += 1
            
            total_checks += 2
        
        if total_checks == 0:
            return 0.8
        
        base_score = coherence_indicators / total_checks
        return max(0.3, min(1.0, base_score + 0.4))
    
    def _calculate_completeness_score(self, text: str, content_type: ChunkType) -> float:
        """Calculate how complete/standalone the text chunk is."""
        # Different completeness criteria based on content type
        if content_type == ChunkType.HEADING:
            return 0.3  # Headings are usually incomplete without context
        elif content_type == ChunkType.CONCLUSION:
            return 0.9  # Conclusions are usually complete
        elif content_type == ChunkType.INTRODUCTION:
            return 0.7  # Introductions are moderately complete
        
        # For regular content, check for completeness indicators
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        completeness_score = 0.5  # Base score
        
        # Check if text ends with proper punctuation
        if text.strip().endswith(('.', '!', '?')):
            completeness_score += 0.2
        
        # Check for complete thoughts (simple heuristic)
        complete_sentences = sum(1 for s in sentences if s.strip().endswith(('.', '!', '?')))
        if len(sentences) > 0:
            completeness_score += 0.3 * (complete_sentences / len(sentences))
        
        return max(0.0, min(1.0, completeness_score))
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density (information per token)."""
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
        
        # Count unique words vs total words
        unique_words = len(set(words))
        total_words = len(words)
        
        # Count information-bearing words (nouns, verbs, adjectives)
        # This is a simplified approach without POS tagging
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", 
                    "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        
        information_words = sum(1 for word in words if word not in stopwords and len(word) > 2)
        
        # Calculate density
        uniqueness_ratio = unique_words / total_words
        information_ratio = information_words / total_words
        
        density = (uniqueness_ratio + information_ratio) / 2
        return max(0.0, min(1.0, density))
    
    def _determine_reading_level(self, text: str) -> ContentDifficulty:
        """Determine the reading difficulty level."""
        text_lower = text.lower()
        
        # Count difficulty indicators
        difficulty_scores = {}
        for level, keywords in self.difficulty_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            difficulty_scores[level] = score
        
        # Also consider sentence length and word complexity
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
        
        words = word_tokenize(text)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Adjust scores based on text complexity
        if avg_sentence_length > 20 or avg_word_length > 7:
            difficulty_scores["advanced"] += 2
            difficulty_scores["expert"] += 1
        elif avg_sentence_length < 10 and avg_word_length < 5:
            difficulty_scores["elementary"] += 2
            difficulty_scores["intermediate"] += 1
        
        # Return the level with highest score
        if not any(difficulty_scores.values()):
            return ContentDifficulty.INTERMEDIATE
        
        max_level = max(difficulty_scores.items(), key=lambda x: x[1])[0]
        return ContentDifficulty(max_level)
    
    def _extract_topic_tags(self, text: str) -> List[str]:
        """Extract topic tags from text content."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", 
                    "to", "for", "of", "with", "by", "is", "are", "was", "were",
                    "this", "that", "these", "those", "it", "he", "she", "they"}
        
        meaningful_words = [word for word in words 
                          if word not in stopwords and len(word) > 3 and word.isalpha()]
        
        # Count word frequency
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top frequent words as tags
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5] if freq > 1]
    
    def _analyze_writing_style(self, text: str) -> str:
        """Analyze writing style of the text."""
        # Simple heuristics for style detection
        if re.search(r'\b(shall|must|will|should)\b', text, re.IGNORECASE):
            return "formal"
        elif re.search(r'\b(gonna|wanna|can\'t|won\'t)\b', text, re.IGNORECASE):
            return "casual"
        elif re.search(r'\b(algorithm|function|variable|implementation)\b', text, re.IGNORECASE):
            return "technical"
        elif re.search(r'\b(imagine|suppose|perhaps|wonderful)\b', text, re.IGNORECASE):
            return "creative"
        else:
            return "formal"  # Default
    
    def _analyze_tone(self, text: str) -> str:
        """Analyze the tone of the text."""
        # Simple tone detection based on keywords
        if re.search(r'\b(should|must|recommend|advise|suggest)\b', text, re.IGNORECASE):
            return "prescriptive"
        elif re.search(r'\b(story|tale|once|happened|character)\b', text, re.IGNORECASE):
            return "narrative"
        elif re.search(r'\b(buy|purchase|best|amazing|incredible)\b', text, re.IGNORECASE):
            return "persuasive"
        else:
            return "informative"  # Default


class EnhancedSemanticChunker:
    """
    Enhanced semantic chunker with comprehensive structure analysis.
    
    This chunker implements Phase 1.1 requirements:
    - Chapter-aware chunking
    - Heading-aware chunking  
    - Sentence-level chunking
    - Enhanced metadata with importance scoring
    - Overlapping context windows
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the enhanced semantic chunker."""
        self.config = config or ChunkingConfig()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.content_analyzer = ContentAnalyzer()
        
    def chunk_documents(
        self,
        documents: List[Document],
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """
        Chunk documents using enhanced semantic analysis.
        
        Args:
            documents: List of documents to chunk
            config: Optional chunking configuration
            
        Returns:
            List[Document]: Enhanced semantically chunked documents
        """
        config = config or self.config
        logger.info(f"Enhanced semantic chunking {len(documents)} documents")
        
        enhanced_chunks = []
        
        for doc_idx, document in enumerate(documents):
            doc_chunks = self._chunk_single_document(document, config, doc_idx)
            enhanced_chunks.extend(doc_chunks)
        
        # Add global relationship analysis
        enhanced_chunks = self._analyze_chunk_relationships(enhanced_chunks)
        
        logger.info(f"Created {len(enhanced_chunks)} enhanced semantic chunks")
        return enhanced_chunks
    
    def _chunk_single_document(
        self, 
        document: Document, 
        config: ChunkingConfig,
        doc_idx: int
    ) -> List[Document]:
        """Chunk a single document with enhanced semantic analysis."""
        
        # Analyze document structure
        structure = self.structure_analyzer.analyze_structure(document.page_content)
        
        # Create chunks based on structure
        chunks = self._create_structure_aware_chunks(
            document.page_content, 
            structure, 
            config
        )
        
        # Convert to Document objects with enhanced metadata
        enhanced_docs = []
        for chunk_idx, chunk_data in enumerate(chunks):
            enhanced_metadata = self._create_enhanced_metadata(
                chunk_data, 
                document, 
                chunk_idx, 
                len(chunks),
                doc_idx
            )
            
            # Create document with enhanced metadata
            chunk_doc = Document(
                page_content=chunk_data["text"],
                metadata=enhanced_metadata.to_dict()
            )
            
            enhanced_docs.append(chunk_doc)
        
        return enhanced_docs
    
    def _create_structure_aware_chunks(
        self, 
        text: str, 
        structure: Dict[str, Any], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """Create chunks that respect document structure."""
        
        chunks = []
        
        # If document has clear chapter structure, chunk by chapters
        if structure["hierarchy"]["chapters"]:
            chunks = self._chunk_by_chapters(text, structure, config)
        # If document has section structure, chunk by sections
        elif structure["hierarchy"]["sections"]:
            chunks = self._chunk_by_sections(text, structure, config)
        # Otherwise, use paragraph-aware chunking
        else:
            chunks = self._chunk_by_paragraphs(text, structure, config)
        
        # Add overlapping context windows if enabled
        if config.enable_context_windows:
            chunks = self._add_context_windows(chunks, text, config)
        
        return chunks
    
    def _chunk_by_chapters(
        self, 
        text: str, 
        structure: Dict[str, Any], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """Chunk text by chapter boundaries."""
        
        chapters = structure["hierarchy"]["chapters"]
        chunks = []
        
        for i, chapter in enumerate(chapters):
            # Determine chapter boundaries
            start_pos = chapter["start"]
            end_pos = chapters[i + 1]["start"] if i + 1 < len(chapters) else len(text)
            
            chapter_text = text[start_pos:end_pos].strip()
            
            # If chapter is too long, split into smaller chunks
            if len(chapter_text) > config.max_chunk_size:
                sub_chunks = self._split_large_chunk(chapter_text, config, start_pos)
                for sub_chunk in sub_chunks:
                    sub_chunk.update({
                        "chapter_number": i + 1,
                        "chapter_title": chapter["text"],
                        "content_type": ChunkType.PARAGRAPH,
                        "hierarchy_level": 1
                    })
                chunks.extend(sub_chunks)
            else:
                chunk = {
                    "text": chapter_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "chapter_number": i + 1,
                    "chapter_title": chapter["text"],
                    "content_type": ChunkType.PARAGRAPH,
                    "hierarchy_level": 1
                }
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sections(
        self, 
        text: str, 
        structure: Dict[str, Any], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """Chunk text by section boundaries."""
        
        sections = structure["hierarchy"]["sections"]
        chunks = []
        
        for i, section in enumerate(sections):
            start_pos = section["start"]
            end_pos = sections[i + 1]["start"] if i + 1 < len(sections) else len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) > config.max_chunk_size:
                sub_chunks = self._split_large_chunk(section_text, config, start_pos)
                for sub_chunk in sub_chunks:
                    sub_chunk.update({
                        "section_title": section["text"],
                        "content_type": ChunkType.PARAGRAPH,
                        "hierarchy_level": 2
                    })
                chunks.extend(sub_chunks)
            else:
                chunk = {
                    "text": section_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "section_title": section["text"],
                    "content_type": ChunkType.PARAGRAPH,
                    "hierarchy_level": 2
                }
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(
        self, 
        text: str, 
        structure: Dict[str, Any], 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """Chunk text by paragraph boundaries with sentence-level refinement."""
        
        paragraphs = structure["paragraphs"]
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            para_text = paragraph["text"]
            
            # Check if adding this paragraph would exceed chunk size
            if (len(current_chunk) + len(para_text) > config.chunk_size and 
                len(current_chunk) > config.min_chunk_size):
                
                # Finalize current chunk
                if current_chunk.strip():
                    chunk = {
                        "text": current_chunk.strip(),
                        "start_pos": current_start,
                        "end_pos": current_start + len(current_chunk),
                        "content_type": ChunkType.PARAGRAPH,
                        "hierarchy_level": 3
                    }
                    chunks.append(chunk)
                
                # Start new chunk with overlap if configured
                if config.chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_sentence_overlap(current_chunk, config.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + para_text
                else:
                    current_chunk = para_text
                
                current_start = paragraph["start"]
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para_text
                else:
                    current_chunk = para_text
                    current_start = paragraph["start"]
        
        # Add final chunk
        if current_chunk.strip():
            chunk = {
                "text": current_chunk.strip(),
                "start_pos": current_start,
                "end_pos": current_start + len(current_chunk),
                "content_type": ChunkType.PARAGRAPH,
                "hierarchy_level": 3
            }
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_chunk(
        self, 
        text: str, 
        config: ChunkingConfig, 
        base_start_pos: int = 0
    ) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones respecting sentence boundaries."""
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_start = base_start_pos
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if (len(current_chunk) + len(sentence) > config.chunk_size and 
                len(current_chunk) > config.min_chunk_size):
                
                # Finalize current chunk
                if current_chunk.strip():
                    chunk = {
                        "text": current_chunk.strip(),
                        "start_pos": current_start,
                        "end_pos": current_start + len(current_chunk),
                        "content_type": ChunkType.PARAGRAPH,
                        "hierarchy_level": 4
                    }
                    chunks.append(chunk)
                
                # Start new chunk with sentence-level overlap
                if config.chunk_overlap > 0:
                    overlap_sentences = self._get_overlap_sentences(current_chunk, config.chunk_overlap)
                    current_chunk = overlap_sentences + " " + sentence
                else:
                    current_chunk = sentence
                
                current_start = text.find(sentence, current_start)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = text.find(sentence, base_start_pos)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = {
                "text": current_chunk.strip(),
                "start_pos": current_start,
                "end_pos": current_start + len(current_chunk),
                "content_type": ChunkType.PARAGRAPH,
                "hierarchy_level": 4
            }
            chunks.append(chunk)
        
        return chunks
    
    def _get_sentence_overlap(self, text: str, overlap_size: int) -> str:
        """Get overlap text based on sentence boundaries."""
        if len(text) <= overlap_size:
            return text
        
        sentences = sent_tokenize(text)
        if not sentences:
            return text[-overlap_size:]
        
        # Take the last few sentences that fit within overlap size
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_size:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def _get_overlap_sentences(self, text: str, overlap_size: int) -> str:
        """Get overlapping sentences from the end of text."""
        sentences = sent_tokenize(text)
        if not sentences:
            return ""
        
        # Take last sentences that fit in overlap size
        overlap_sentences = []
        current_length = 0
        
        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                current_length += len(sentence)
            else:
                break
        
        return " ".join(overlap_sentences)
    
    def _add_context_windows(
        self, 
        chunks: List[Dict[str, Any]], 
        full_text: str, 
        config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """Add context windows to chunks for improved continuity."""
        
        window_size = config.context_window_size
        
        for i, chunk in enumerate(chunks):
            start_pos = chunk["start_pos"]
            end_pos = chunk["end_pos"]
            
            # Calculate preceding context
            preceding_start = max(0, start_pos - window_size)
            preceding_text = full_text[preceding_start:start_pos].strip()
            
            # Calculate following context
            following_end = min(len(full_text), end_pos + window_size)
            following_text = full_text[end_pos:following_end].strip()
            
            # Add context to chunk
            chunk["preceding_context"] = preceding_text
            chunk["following_context"] = following_text
            chunk["context_window_size"] = window_size
        
        return chunks
    
    def _create_enhanced_metadata(
        self,
        chunk_data: Dict[str, Any],
        original_document: Document,
        chunk_index: int,
        total_chunks: int,
        doc_index: int
    ) -> EnhancedChunkMetadata:
        """Create enhanced metadata for a chunk."""
        
        # Generate unique chunk ID
        chunk_id = f"{original_document.metadata.get('book_id', 'unknown')}_{doc_index}_{chunk_index}"
        
        # Analyze content
        content_analysis = self.content_analyzer.analyze_content(
            chunk_data["text"], 
            chunk_data.get("content_type", ChunkType.PARAGRAPH)
        )
        
        # Create source location
        source_location = SourceLocation(
            page_number=chunk_data.get("page_number"),
            chapter_number=chunk_data.get("chapter_number"),
            chapter_title=chunk_data.get("chapter_title"),
            section_title=chunk_data.get("section_title"),
            character_start=chunk_data.get("start_pos"),
            character_end=chunk_data.get("end_pos")
        )
        
        # Create enhanced metadata
        metadata = EnhancedChunkMetadata(
            chunk_id=chunk_id,
            source_document_id=original_document.metadata.get('book_id', 'unknown'),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content_type=chunk_data.get("content_type", ChunkType.PARAGRAPH),
            hierarchy_level=chunk_data.get("hierarchy_level", 3),
            source_location=source_location,
            
            # Content analysis results
            topic_tags=content_analysis["topic_tags"],
            reading_level=content_analysis["reading_level"],
            importance_score=content_analysis["importance_score"],
            coherence_score=content_analysis["coherence_score"],
            completeness_score=content_analysis["completeness_score"],
            information_density=content_analysis["information_density"],
            
            # Context windows
            preceding_context=chunk_data.get("preceding_context", ""),
            following_context=chunk_data.get("following_context", ""),
            context_window_size=chunk_data.get("context_window_size", 200),
            
            # Style analysis
            writing_style=content_analysis["writing_style"],
            tone=content_analysis["tone"],
            
            # Processing metadata
            chunking_method="enhanced_semantic",
            chunking_strategy="structure_aware",
            original_metadata=original_document.metadata
        )
        
        return metadata
    
    def _analyze_chunk_relationships(self, chunks: List[Document]) -> List[Document]:
        """Analyze relationships between chunks."""
        
        # Simple relationship analysis based on topic overlap
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk.metadata
            chunk_tags = chunk_metadata.get("topic_tags", [])
            
            related_chunks = []
            
            # Find chunks with overlapping topics
            for j, other_chunk in enumerate(chunks):
                if i == j:
                    continue
                
                other_tags = other_chunk.metadata.get("topic_tags", [])
                
                # Calculate topic overlap
                if chunk_tags and other_tags:
                    overlap = len(set(chunk_tags) & set(other_tags))
                    if overlap > 0:
                        related_chunks.append(other_chunk.metadata["chunk_id"])
            
            # Update metadata with relationships
            chunk.metadata["related_chunks"] = related_chunks[:5]  # Limit to top 5
            
            # Add prerequisite relationships for sequential chunks
            if i > 0:
                prev_chunk = chunks[i - 1]
                if (chunk.metadata.get("hierarchy_level", 3) >= 
                    prev_chunk.metadata.get("hierarchy_level", 3)):
                    chunk.metadata["prerequisite_chunks"] = [prev_chunk.metadata["chunk_id"]]
            
            # Add follow-up relationships
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if (chunk.metadata.get("hierarchy_level", 3) >= 
                    next_chunk.metadata.get("hierarchy_level", 3)):
                    chunk.metadata["follow_up_chunks"] = [next_chunk.metadata["chunk_id"]]
        
        return chunks