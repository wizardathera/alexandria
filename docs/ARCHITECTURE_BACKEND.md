**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of 2025-07-05.**

# 🔧 Backend Architecture

## Overview

This document details the backend infrastructure for the Alexandria platform, covering FastAPI routing, RAG pipeline, module architecture, and system integrations.

For additional architecture details, see:
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Frontend Architecture](ARCHITECTURE_FRONTEND.md)
- [Data Model & Storage](ARCHITECTURE_DATA_MODEL.md)
- [AI Services & RAG](ARCHITECTURE_AI_SERVICES.md)

## 🏗️ Module Architecture

### Smart Library Module

**Core Components:**
- **Book Processor**: Handles multiple file formats (PDF, EPUB, DOC, etc.)
- **RAG Engine**: Vector search + LLM generation for Q&A
- **Progress Tracker**: Reading analytics and milestone tracking
- **Note Manager**: User annotations and reflections
- **Discovery Engine**: AI-powered book recommendations
- **Theme Manager**: Personalized reading environment

**API Endpoints:**
```
POST /library/books/upload     # Upload and process new book
GET  /library/books           # List user's books
POST /library/books/{id}/query # Ask questions about specific book
GET  /library/progress        # Get reading progress analytics
POST /library/notes           # Create/update notes
GET  /library/catalog         # Browse public domain catalog
POST /library/purchase        # Purchase premium books
GET  /library/recommendations # Get personalized recommendations
```

### Learning Suite Module

**Core Components:**
- **Course Builder**: AI-assisted course creation from book content
- **Assessment Engine**: Quiz generation and grading
- **Learning Path AI**: Personalized learning recommendations
- **Progress Analytics**: Student and educator dashboards

**API Endpoints:**
```
POST /lms/courses             # Create new course
GET  /lms/courses/{id}        # Get course details
POST /lms/lessons             # Add lesson to course
POST /lms/assessments/generate # AI-generate quiz from content
GET  /lms/analytics/student   # Student progress dashboard
GET  /lms/analytics/educator  # Educator analytics dashboard
```

### Marketplace Module

**Core Components:**
- **Content Monetization**: Pricing, payments, revenue sharing
- **Discovery Engine**: Search, recommendations, curation
- **Community Features**: Reviews, ratings, discussions
- **Creator Tools**: Analytics, marketing, content management

**API Endpoints:**
```
POST /marketplace/items       # List content for sale
GET  /marketplace/search      # Search marketplace content
POST /marketplace/purchase    # Purchase content
GET  /marketplace/recommendations # Personalized recommendations
POST /marketplace/reviews     # Add review/rating
```

## 🧠 Enhanced RAG Service Architecture

The Enhanced RAG (Retrieval-Augmented Generation) Service is the core intelligence layer that powers contextual interactions across all platform modules.

### Core RAG Components

#### 1. Multi-Modal Content Processing Pipeline

**Document Ingestion & Processing:**
```python
# Multi-format document processing with module-aware chunking
class ContentProcessor:
    def __init__(self):
        self.format_handlers = {
            'pdf': PDFHandler(),
            'epub': EPUBHandler(), 
            'docx': DOCXHandler(),
            'txt': TextHandler(),
            'html': HTMLHandler(),
            'markdown': MarkdownHandler(),
            'video': VideoTranscriptHandler(),  # Future: video transcription
            'audio': AudioTranscriptHandler(),  # Future: audio transcription
            'image': ImageOCRHandler()          # Future: image text extraction
        }
    
    async def process_content(
        self, 
        content: bytes, 
        content_type: str,
        module: str,  # 'library', 'lms', 'marketplace'
        metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """Process content with module-specific chunking strategies."""
        pass
```

#### 2. Advanced Vector Storage & Retrieval

**Enhanced Embedding Strategy:**
```python
class EnhancedEmbeddingService:
    def __init__(self):
        self.embedding_models = {
            'text': 'text-embedding-ada-002',      # General text
            'code': 'text-embedding-ada-002',      # Code snippets
            'academic': 'text-embedding-ada-002',   # Academic content
            'conversational': 'text-embedding-ada-002'  # Q&A pairs
        }
        
        # Multi-dimensional embeddings for different use cases
        self.embedding_dimensions = {
            'semantic': 1536,      # Meaning-based similarity
            'syntactic': 768,      # Structure-based similarity
            'domain_specific': 512  # Domain-specific embeddings
        }
    
    async def generate_multi_embeddings(
        self,
        content: str,
        content_type: str,
        module: str
    ) -> Dict[str, List[float]]:
        """Generate multiple embedding types for comprehensive search."""
        pass
```

**Hybrid Search Implementation:**
```python
class HybridSearchEngine:
    def __init__(self):
        self.vector_search = VectorSearchEngine()
        self.text_search = FullTextSearchEngine()
        self.semantic_search = SemanticSearchEngine()
        self.graph_search = GraphSearchEngine()  # For relationship-based search
    
    async def hybrid_search(
        self,
        query: str,
        search_params: SearchParameters
    ) -> RankedResults:
        """
        Perform hybrid search combining multiple strategies:
        1. Vector similarity search (semantic)
        2. Full-text search (keyword matching)
        3. Graph-based search (relationship traversal)
        4. Contextual search (user history, preferences)
        """
        
        # Parallel search execution
        vector_results = await self.vector_search.search(query, search_params)
        text_results = await self.text_search.search(query, search_params)
        semantic_results = await self.semantic_search.search(query, search_params)
        graph_results = await self.graph_search.search(query, search_params)
        
        # Intelligent result fusion with relevance scoring
        fused_results = self.fuse_results([
            vector_results, text_results, semantic_results, graph_results
        ])
        
        return fused_results
```

#### 3. Context-Aware Query Processing

**Multi-Stage Query Understanding:**
```python
class QueryProcessor:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.context_manager = ContextManager()
        self.query_expander = QueryExpander()
    
    async def process_query(
        self,
        raw_query: str,
        user_context: UserContext,
        conversation_history: List[Message]
    ) -> ProcessedQuery:
        """
        Multi-stage query processing:
        1. Intent classification (question, search, comparison, etc.)
        2. Entity extraction (books, concepts, people, etc.)
        3. Context integration (previous queries, user preferences)
        4. Query expansion (synonyms, related concepts)
        5. Search strategy selection
        """
        
        # Stage 1: Understand user intent
        intent = await self.intent_classifier.classify(raw_query)
        
        # Stage 2: Extract entities and concepts
        entities = await self.entity_extractor.extract(raw_query)
        
        # Stage 3: Build contextual understanding
        context = await self.context_manager.build_context(
            user_context, conversation_history
        )
        
        # Stage 4: Expand query with related concepts
        expanded_query = await self.query_expander.expand(
            raw_query, entities, context
        )
        
        return ProcessedQuery(
            original=raw_query,
            expanded=expanded_query,
            intent=intent,
            entities=entities,
            context=context
        )
```

#### 4. Intelligent Response Generation

**Multi-Strategy Response Generation:**
```python
class ResponseGenerator:
    def __init__(self):
        self.llm_providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),  # Future
            'local': LocalModelProvider()      # Future
        }
        
        self.response_strategies = {
            'direct_answer': DirectAnswerStrategy(),
            'comparative': ComparativeAnalysisStrategy(),
            'explanatory': ExplanatoryStrategy(),
            'creative': CreativeStrategy(),
            'analytical': AnalyticalStrategy(),
            'summarization': SummarizationStrategy()
        }
    
    async def generate_response(
        self,
        query: ProcessedQuery,
        retrieved_context: RetrievedContext,
        response_requirements: ResponseRequirements
    ) -> EnhancedResponse:
        """
        Generate contextually appropriate response:
        1. Select appropriate response strategy
        2. Choose optimal LLM provider and model
        3. Construct context-aware prompt
        4. Generate response with source citations
        5. Validate response quality and relevance
        """
        pass
```

### RAG Service Capabilities by Module

#### Library Module RAG Features

**Book-Centric Intelligence:**
- **Multi-Book Analysis**: Compare themes, concepts, and arguments across books
- **Reading Comprehension**: Generate discussion questions and analysis prompts
- **Knowledge Synthesis**: Combine insights from multiple sources
- **Progress-Aware Responses**: Tailor responses to user's reading progress

```python
class LibraryRAGService:
    async def analyze_reading_patterns(self, user_id: str) -> ReadingInsights:
        """Analyze user's reading patterns and generate insights."""
        pass
    
    async def generate_book_connections(self, book_ids: List[str]) -> BookConnections:
        """Find thematic and conceptual connections between books."""
        pass
    
    async def create_reading_discussion(self, book_id: str, chapter: str) -> Discussion:
        """Generate discussion questions for book club or self-reflection."""
        pass
```

#### LMS Module RAG Features

**Educational Intelligence:**
- **Adaptive Learning**: Adjust explanations based on student progress and understanding
- **Assessment Generation**: Create quizzes, exercises, and evaluation materials
- **Learning Path Optimization**: Recommend optimal learning sequences
- **Difficulty Adaptation**: Scale content complexity to student level

```python
class LMSRAGService:
    async def generate_adaptive_explanation(
        self,
        concept: str,
        student_profile: StudentProfile
    ) -> AdaptiveExplanation:
        """Generate explanations adapted to student's level and learning style."""
        pass
    
    async def create_assessment_questions(
        self,
        content: str,
        difficulty_level: str,
        question_types: List[str]
    ) -> AssessmentQuestions:
        """Generate various types of assessment questions from content."""
        pass
    
    async def recommend_learning_path(
        self,
        student_goals: List[str],
        current_knowledge: KnowledgeProfile
    ) -> LearningPath:
        """Recommend personalized learning paths based on goals and current knowledge."""
        pass
```

#### Marketplace Module RAG Features

**Commerce Intelligence:**
- **Content Discovery**: Intelligent product recommendations based on user interests
- **Quality Assessment**: Automated content quality evaluation and scoring
- **Market Analysis**: Trend identification and market insight generation
- **Personalized Curation**: Custom content collections based on user preferences

```python
class MarketplaceRAGService:
    async def generate_content_recommendations(
        self,
        user_profile: UserProfile,
        interaction_history: List[Interaction]
    ) -> ContentRecommendations:
        """Generate personalized content recommendations."""
        pass
    
    async def analyze_market_trends(
        self,
        time_period: DateRange,
        categories: List[str]
    ) -> MarketTrendAnalysis:
        """Analyze market trends and opportunities."""
        pass
    
    async def evaluate_content_quality(
        self,
        content_id: str,
        evaluation_criteria: List[str]
    ) -> QualityAssessment:
        """Automated content quality evaluation."""
        pass
```

## 🤖 Hypatia Assistant Architecture

The Hypatia Conversational Assistant is the branded AI companion that provides personalized, context-aware interactions across the platform.

### Core Architecture Components

#### Chat Interface System

**Backend Chat Service:**
```python
class HypatiaService:
    def __init__(self):
        self.prompt_router = PromptRouter()
        self.personality_engine = PersonalityEngine()
        self.memory_service = MemoryService()
        self.analytics_service = AnalyticsService()
    
    async def process_message(
        self,
        message: str,
        user_context: UserContext,
        conversation_history: List[Message]
    ) -> HypatiaResponse:
        """
        Process user message through Hypatia's intelligence pipeline:
        1. Intent classification and context understanding
        2. Personality-aware response generation
        3. Memory integration and session management
        4. Analytics and feedback collection
        """
        pass
```

#### Prompt Routing System

**Intelligent Context Classification:**
```python
class PromptRouter:
    def __init__(self):
        self.context_types = {
            'onboarding': OnboardingPrompts(),
            'feature_help': FeatureHelpPrompts(),
            'book_discovery': BookDiscoveryPrompts(),
            'book_qa': BookQAPrompts(),
            'general_chat': GeneralChatPrompts()
        }
        
        self.intent_classifier = IntentClassifier()
        self.context_switcher = ContextSwitcher()
    
    async def route_conversation(
        self,
        message: str,
        current_context: ConversationContext,
        user_profile: UserProfile
    ) -> RoutedPrompt:
        """
        Route conversation to appropriate prompt strategy:
        1. Classify user intent and determine conversation type
        2. Consider current context and conversation history
        3. Select optimal prompt template and response strategy
        4. Handle context switching with smooth transitions
        """
        pass
```

#### Personality Engine

**Multi-Personality System:**
```python
class PersonalityEngine:
    def __init__(self):
        self.personalities = {
            'friendly': FriendlyPersonality(),
            'scholarly': ScholarlyPersonality(),
            'witty': WittyPersonality(),
            'empathetic': EmpatheticPersonality(),
            'pragmatic': PragmaticPersonality(),
            'philosophical': PhilosophicalPersonality()
        }
        
        self.personality_adapter = PersonalityAdapter()
        self.context_analyzer = ContextAnalyzer()
    
    async def generate_response(
        self,
        message: str,
        personality_type: PersonalityType,
        conversation_context: ConversationContext
    ) -> PersonalizedResponse:
        """
        Generate personality-aware responses:
        1. Analyze conversation context and user emotional state
        2. Apply personality-specific response patterns
        3. Maintain consistency with chosen personality traits
        4. Adapt tone and style based on conversation type
        """
        pass
```

## 💳 E-commerce & Payment Architecture

### Stripe Integration Architecture

**Payment Flow Design:**
```python
interface PaymentSystem {
  checkout: StripeCheckoutSession
  webhooks: PaymentWebhookHandler
  subscription: SubscriptionManager
  billing: BillingEngine
}

// Secure payment processing
class PaymentProcessor {
  async createCheckoutSession(items: PurchaseItem[]): Promise<CheckoutSession>
  async processPaymentWebhook(event: StripeWebhookEvent): Promise<void>
  async handleRefund(paymentId: string, amount?: number): Promise<RefundResult>
  async calculateTax(address: BillingAddress, items: PurchaseItem[]): Promise<TaxCalculation>
}
```

**Digital Rights Management:**
```python
class DRMSystem:
    def __init__(self):
        self.licensing = ContentLicenseManager()
        self.access = AccessControlManager()
        self.encryption = ContentEncryption()
        self.delivery = SecureContentDelivery()

class ContentDeliveryManager:
    async def authorizeAccess(self, userId: str, contentId: str) -> AccessToken:
        """Authorize user access to purchased content."""
        pass
    
    async def trackUsage(self, userId: str, contentId: str, usage: UsageMetrics) -> None:
        """Track content usage for analytics and compliance."""
        pass
    
    async def revokeAccess(self, userId: str, contentId: str) -> None:
        """Revoke access to content when necessary."""
        pass
    
    async def generateDownloadLink(self, contentId: str, accessToken: str) -> SecureDownloadLink:
        """Generate secure download links for purchased content."""
        pass
```

## 🔄 Migration Strategies

### Database Migration (Chroma → Supabase) ✅ **COMPLETED**

**Migration Architecture Implementation Status:**
- ✅ **Supabase pgvector schema migration scripts** - Complete production-ready schema
- ✅ **Dual-write capability** - Seamless write operations to both Chroma + Supabase
- ✅ **VectorDatabaseInterface abstraction** - Clean migration path with zero API changes
- ✅ **Data validation and consistency tools** - Comprehensive validation framework
- ✅ **Rollback strategy and zero-data-loss validation** - Complete disaster recovery
- ✅ **Performance testing and optimization** - Validated equivalent performance

**Migration Tools (Implemented):**
```python
# Production-ready migration system
class SupabaseMigrationService:
    """Complete migration service with validation and rollback capabilities."""
    
    def __init__(self, chroma_client, supabase_client):
        self.chroma = chroma_client
        self.supabase = supabase_client
        self.validator = MigrationValidator()
    
    async def create_schema(self):
        """Create production pgvector schema with indexes"""
        # ✅ Implemented - Creates all required tables and indexes
    
    async def migrate_content_items(self, batch_size=1000):
        """Migrate content items with progress tracking"""
        # ✅ Implemented - Batch migration with validation
    
    async def migrate_embeddings(self, batch_size=500):
        """Migrate vector embeddings with metadata"""
        # ✅ Implemented - Full embedding migration with enhanced metadata
    
    async def migrate_relationships(self):
        """Migrate content relationships and permissions"""
        # ✅ Implemented - Complete relationship mapping
    
    async def validate_migration(self, sample_size=100):
        """Comprehensive migration validation"""
        # ✅ Implemented - Multi-dimensional validation
    
    async def rollback_migration(self):
        """Safe rollback to Chroma if needed"""
        # ✅ Implemented - Complete rollback strategy

# Dual-write capability for seamless transition
class DualWriteVectorDB(VectorDatabaseInterface):
    """Write to both Chroma and Supabase during migration"""
    # ✅ Implemented - Transparent dual-write operations
```

### Frontend Migration (Streamlit → Next.js)

**Component Mapping:**
- Streamlit file uploader → Next.js file upload component
- Streamlit chat interface → Custom React chat component
- Streamlit charts → Chart.js or D3.js visualizations
- Streamlit forms → React Hook Form implementations

**Migration Strategy:**
1. **Component Inventory**: Map all Streamlit components
2. **API First**: Ensure all functionality accessible via API
3. **Progressive Migration**: Replace components incrementally
4. **Feature Parity**: Maintain all existing functionality

## 📊 Monitoring & Observability

### Application Monitoring

**Health Checks:**
- API endpoint health monitoring
- Database connection health
- Vector database query performance
- External service availability (OpenAI, Stripe)

**Performance Metrics:**
- Response time percentiles (p50, p95, p99)
- Throughput (requests per second)
- Error rates by endpoint
- Database query performance

**Business Metrics:**
- User registration and retention
- Content upload and query volumes
- Course creation and completion rates
- Revenue and transaction metrics

### Alerting Strategy

**Critical Alerts (Immediate):**
- API downtime or high error rates
- Database connectivity issues
- Payment processing failures
- Security breach indicators

**Warning Alerts (Within 1 hour):**
- Performance degradation
- High resource usage
- Failed background jobs
- User experience issues

**Informational Alerts (Daily summary):**
- Usage trends and anomalies
- Business metric changes
- System capacity planning needs

## 🚀 Deployment Architecture

### Containerization Strategy

```dockerfile
# Multi-stage Docker build
FROM python:3.11-slim as base
# Base dependencies and security updates

FROM base as deps
# Install Python dependencies

FROM deps as app
# Copy application code and configuration

FROM app as production
# Production optimizations and health checks
```

### Infrastructure as Code

**Development Environment:**
```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment: [DB_URL, OPENAI_API_KEY]
  
  db:
    image: postgres:15
    environment: [POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

**Production Environment:**
- Kubernetes manifests for container orchestration
- Terraform for cloud infrastructure provisioning
- Helm charts for application deployment
- GitOps with ArgoCD for deployment automation

### Scalability Considerations

**Horizontal Scaling:**
- Load balancers for API servers
- Database read replicas
- Caching layers (Redis)
- CDN for static assets

**Performance Optimization:**
- Database query optimization
- API response caching
- Background job processing
- Resource monitoring and alerting

## 🔗 Related Documentation

- [Frontend Architecture →](ARCHITECTURE_FRONTEND.md)
- [Data Model & Storage →](ARCHITECTURE_DATA_MODEL.md)
- [AI Services & RAG →](ARCHITECTURE_AI_SERVICES.md)
- [Architecture Overview →](ARCHITECTURE_OVERVIEW.md)