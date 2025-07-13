# 📋 Alexandria Platform - Active Development Tasks

**Last Updated**: 2025-07-12  
**Current Status**: Phase 2.0 - Next.js Migration & Learning Suite Foundation 🚀 **ACTIVE**

## 🎯 Phase 2.1 Catalog and Library Features ✅ *COMPLETED*

### Recently Completed ✅
- **2.10** Implement Public Domain Catalog Page ✅
- **2.11** Implement "Add to My Library" Functionality ✅  
- **2.12** Update Personal Library to Show User Collection ✅
- **2.13** Add Catalog Navigation and UI Integration ✅

### Task Details 📋

#### 2.10-2.13 Catalog and Library Implementation ✅ *COMPLETED*
**Epic**: Frontend User Experience  
**Story**: Users can browse public domain books and add them to personal library  
**Labels**: frontend, Next.js, React, Supabase, user-library  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-07-12

**Requirements**:
- ✅ Public domain catalog page with book browsing
- ✅ "Add to My Library" functionality for individual users  
- ✅ Personal library showing user's added books
- ✅ Secure user authentication and data isolation
- ✅ Integration with existing Supabase schema (user_progress table)
- ✅ Navigation updates to include catalog access

**Implementation Summary**:
- **Frontend Pages**: 
  - `/catalog` - Browse public domain books with search and filtering
  - `/library` - User's personal book collection with progress tracking
- **Database Integration**: 
  - Uses existing `user_progress` table for user library management
  - Integrates with `books` table for public domain catalog
- **New React Hooks**: 
  - `useUserLibrary()` - Fetches user's personal book collection
  - `useRemoveFromLibrary()` - Removes books from user's library
  - `useUpdateReadingProgress()` - Updates reading progress tracking
- **Security**: 
  - Row Level Security (RLS) policies for user data isolation
  - Authentication-based access control for personal library actions
- **UI/UX Features**:
  - Grid/List view toggle for book browsing
  - Search and filtering for catalog discovery
  - Book cards with cover images, descriptions, and metadata
  - Loading states and error handling
  - Progress tracking for user library books

**Acceptance Criteria**:
- ✅ Users can browse public domain catalog with 5+ seeded books
- ✅ Users can add books to personal library with one-click action
- ✅ Personal library shows only user's books with progress tracking
- ✅ Proper authentication and data security implemented
- ✅ Navigation includes catalog link in sidebar
- ✅ TypeScript compilation passes without errors
- ✅ Responsive design works on desktop and mobile

## 🎯 Phase 1.6 Critical Fixes ✅ *COMPLETED*

### All Tasks Completed ✅
- **1.61** OpenAI Library Installation and ImportError Resolution ✅
- **1.62** Enhanced Embedding Service Implementation ✅  
- **1.63** Fix /api/enhanced/content 404 Endpoint ✅
- **1.64** Fix Book Upload 500 Server Error ✅
- **1.65** Fix Search Endpoint Functionality ✅
- **1.66** Fix Q&A Chat Empty State ✅
- **1.67** Fix Content Relationships Explorer Backend ✅
- **1.68** Fix Reading Analytics Dashboard Backend ✅
- **1.69** Fix Theme Modes Visual Consistency ✅

### Task Details 📋

#### 1.65 Fix Search Endpoint Functionality ✅ *COMPLETED*
**Epic**: Backend Stability  
**Story**: Search endpoints return results correctly  
**Labels**: backend, search, rag  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-07-06

**Requirements**:
- Confirm embeddings exist in vector database after successful ingestion
- Validate search results return with proper relevance scoring and metadata
- Test both simple (/api/chat) and enhanced (/api/enhanced/search) search endpoints
- Verify search performance meets <3 second requirement
- Ensure hybrid search (vector + BM25) functionality works correctly

**Subtasks**:
- Verify embedding generation and storage in vector database
- Test vector similarity search functionality
- Validate BM25 keyword search integration
- Test hybrid search result fusion (RRF algorithm)
- Verify search result metadata and confidence scoring
- Performance testing for search response times

**Acceptance Criteria**:
- ✅ Search endpoints return relevant results for ingested content
- ✅ Search results include proper metadata, confidence scores, and source citations
- ✅ Search response times consistently under 3 seconds (average 0.21s)
- ✅ Vector similarity search functionality working correctly
- ✅ Hybrid search infrastructure implemented (BM25 requires separate indexing)
- ✅ Search quality demonstrates improved relevance (0.09-0.22 similarity scores)

**Implementation Summary**:
- Fixed EmbeddingService.default_model attribute issue
- Rebuilt alexandria_books collection with enhanced metadata format (522 documents migrated)
- Enhanced database now loads existing collections automatically
- Fixed hardcoded collection name references in hybrid search system
- All search endpoints now return relevant results with proper metadata
- Performance excellent: 0.21s average response time vs 3s target
- RAG service confidence improved to 0.398 with relevant psychological content

#### 1.66 Fix Q&A Chat Empty State ✅ *COMPLETED*
**Epic**: Frontend User Experience  
**Story**: Q&A chat interface handles empty states gracefully  
**Labels**: frontend, chat, ux  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-01-06  
**Estimated Effort**: 4 hours

**Requirements**:
- Ensure frontend displays chat input interface correctly in all scenarios
- Add appropriate fallback messaging when no content exists or is uploaded
- Test chat functionality with and without uploaded books
- Verify chat history and conversation flow works properly
- Add loading states and error handling for API calls

**Subtasks**:
- Implement empty state UI for when no books are uploaded
- Add helpful messaging with clear next steps for new users
- Test chat interface across different content availability states
- Implement loading states for all chat-related API calls
- Add error handling with user-friendly messages and retry mechanisms

**Acceptance Criteria**:
- ✅ Chat interface displays correctly in all scenarios (empty library, loading, error states)
- ✅ Helpful messaging when no books are uploaded with clear next steps
- ✅ Q&A functionality works seamlessly with uploaded content
- ✅ Chat history and export features functional
- ✅ Error states provide actionable feedback to users

**Implementation Summary**:
- Implemented comprehensive empty state UI with clear call-to-action buttons
- Added "Getting Started" guide with supported file types and feature preview
- Enhanced error handling with specific error messages and retry mechanisms
- Added loading indicators for all API calls with status messages
- Implemented backend connectivity checks and user-friendly retry options
- Verified chat history persistence and export functionality (Text, JSON, Markdown)
- All states tested: empty library, loading, various error scenarios, normal operation

#### 1.67 Fix Content Relationships Explorer Backend ✅ *COMPLETED*
**Epic**: Backend Analytics  
**Story**: Backend returns data for relationship visualization  
**Labels**: backend, analytics, relationships  
**Priority**: Medium  
**Status**: ✅ Complete  
**Completed**: 2025-01-06  
**Estimated Effort**: 5 hours

**Requirements**:
- Confirm backend content relationship APIs return valid graph data structure
- Validate relationship discovery algorithms work with ingested content
- Test relationship mapping with multiple books and content types
- Ensure performance acceptable with larger content libraries (100+ items)
- Add proper error handling for relationship API endpoints

**Subtasks**:
- Test relationship discovery API endpoints (/api/enhanced/relationships)
- Validate graph data structure format for frontend visualization
- Test relationship algorithm performance with multiple content items
- Implement proper error handling for no relationships scenarios
- Verify relationship strength scoring and confidence metrics

**Acceptance Criteria**:
- ✅ Relationship APIs return valid graph data structure
- ✅ Relationship discovery identifies meaningful connections between content
- ✅ API performance acceptable with realistic data volumes (<2 seconds response)
- ✅ Proper error handling when no relationships exist
- ✅ Graph data structure compatible with frontend visualization requirements

**Implementation Summary**:
- Implemented three new relationship API endpoints: content relationships, graph data, and AI-powered discovery
- Built comprehensive graph retrieval engine with BFS and Random Walk algorithms
- Added frontend-compatible response models with proper type validation
- Implemented comprehensive error handling for all edge cases
- Performance tested with 100+ content items (all endpoints <2s response time)
- Created extensive test suite with 12 unit tests and integration validation
- Validated frontend compatibility with all expected data structures

#### 1.68 Fix Reading Analytics Dashboard Backend ✅ *COMPLETED*
**Epic**: Backend Analytics  
**Story**: Analytics data is returned and displayed correctly  
**Labels**: backend, analytics, dashboard  
**Priority**: Medium  
**Status**: ✅ Complete  
**Completed**: 2025-07-07
**Estimated Effort**: 4 hours

**Requirements**:
- Confirm backend returns reading history and metrics data via APIs
- Validate analytics calculation accuracy for reading progress and patterns
- Test analytics with various reading patterns and content types
- Ensure charts data structure is compatible with frontend visualization
- Add proper aggregation and time-based analytics functionality

**Subtasks**:
- Test analytics API endpoints (/api/analytics/*)
- Validate reading progress calculation accuracy
- Test analytics data aggregation and time-based queries
- Verify data structure compatibility with frontend charting libraries
- Implement proper error handling for analytics operations

**Acceptance Criteria**:
- ✅ Analytics APIs return accurate reading data (15 new endpoints implemented)
- ✅ Reading progress calculations are mathematically correct (0-100% with completion detection)
- ✅ Analytics provide meaningful insights (reading time, sessions, completion rates, streaks)
- ✅ Data structure compatible with frontend charting libraries (Plotly/Chart.js compatible)
- ✅ Performance acceptable for analytics queries (average 32ms response time vs 1s target)

**Implementation Summary**:
- Created comprehensive analytics API with 15 endpoints at `/api/analytics/*`
- Implemented complete analytics data models (ReadingSession, ReadingProgress, UsageMetrics)
- Built interactive Plotly dashboard with time-series charts, pie charts, and progress visualization
- Added session tracking, progress monitoring, and aggregated analytics summaries
- Created comprehensive test suite with 47 tests achieving 97.9% success rate
- All endpoints meet performance requirements with average 32ms response time
- Enhanced frontend dashboard with real-time data loading and interactive visualizations

#### 1.69 Fix Theme Modes Visual Consistency ✅ *COMPLETED*
**Epic**: Frontend User Experience  
**Story**: Theme system visual consistency improvements  
**Labels**: frontend, themes, ux  
**Priority**: Low  
**Status**: ✅ Complete  
**Completed**: 2025-07-07
**Estimated Effort**: 2 hours

**Requirements**:
- Validate theme preference APIs work correctly
- Ensure theme settings are properly stored and retrieved
- Test theme mode switching across different user sessions
- Validate theme persistence and synchronization
- Add proper error handling for theme-related operations

**Subtasks**:
- Test theme preference storage and retrieval APIs
- Verify theme persistence across user sessions
- Implement consistent theme application across all UI components
- Test theme switching performance and visual transitions
- Add error handling for theme operations

**Acceptance Criteria**:
- ✅ Theme preference APIs store and retrieve settings correctly
- ✅ Theme synchronization works across user sessions
- ✅ Proper default theme handling for new users
- ✅ Error handling provides clear feedback for theme operations
- ✅ Theme settings integrate properly with user preferences system

**Implementation Summary**:
- Added theme selection UI to sidebar under "🎨 Theme Settings"
- Enhanced error handling with success/error messages and graceful fallback
- Updated Alexandria Classic theme with correct color palette (Marble White, Gold, Indigo, Lapis Blue, Charcoal)
- Added visual enhancements (gradients, shadows, themed emojis) for Alexandria Classic
- Verified theme persistence across sessions in `~/.alexandria/themes/user_preferences.json`
- Performance tested: <5s for 30 theme switches, <2s for 30 CSS generations
- All 6 comprehensive test suites passing

---

## 🚀 Phase 2.0: Next.js Frontend Migration & Learning Suite Foundation ⭐ *ACTIVE*

**Timeline**: 12-16 weeks (2025-07-08 to 2025-10-31)  
**Sprint Structure**: 2-week sprints with thematic focus  
**Primary Goals**: Multi-user platform + Learning Suite MVP + Hypatia Assistant 1.0

### 2.1 Next.js Migration & Multi-User Setup 🔄

#### 2.11 Migrate Frontend to Next.js 14 + TypeScript 📋
**Epic**: Frontend Migration  
**Story**: Complete migration from Streamlit to production-ready Next.js application  
**Labels**: frontend, migration, nextjs, typescript  
**Priority**: High  
**Status**: ✅ **COMPLETE**  
**Completed**: 2025-07-09  
**Estimated Effort**: 20 hours  
**Dependencies**: None (foundational task)

**Requirements**:
- Set up Next.js 14+ project with TypeScript and strict type checking
- Configure Tailwind CSS and Shadcn/ui component library
- Implement state management with Zustand and React Query
- Set up routing with Next.js App Router and protected routes
- Create responsive layout components and navigation system

**Subtasks**:
- Initialize Next.js project with TypeScript configuration
- Configure Tailwind CSS, Shadcn/ui, and design tokens
- Set up Zustand for global state management
- Implement React Query for API state management
- Create responsive layout components and navigation
- Migrate core Streamlit pages to React components

**Acceptance Criteria**:
- ✅ Next.js application runs locally with TypeScript strict mode
- ✅ All major Streamlit pages have React equivalents
- ✅ Responsive design works on mobile and desktop
- ✅ State management handles complex user interactions
- ✅ Performance meets Lighthouse scores >90

**Implementation Summary**:
- Successfully migrated to Next.js 14.2.30 with TypeScript strict mode enabled
- Implemented complete Alexandria Classic theme system with Lora fonts and brand colors
- Created all major application pages: dashboard, library, chat, search, analytics, upload, auth
- Configured Zustand store for global state management and React Query for server state
- Built responsive layout system with Tailwind CSS and Shadcn/ui components
- Implemented authentication context and protected route middleware
- TypeScript compilation passes without errors, ready for production deployment

#### 2.114 Build Unified Design System with Tokens 📋
**Epic**: Frontend Migration  
**Story**: Convert current visual system into centralized design tokens + Figma components  
**Labels**: design, frontend  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 8 hours  
**Dependencies**: Brand team finalization of palette and typography

**Requirements**:
- Centralized design token system for consistent styling
- Figma component library synchronized with code components
- Color palette, typography, and spacing token definitions
- Component documentation and usage guidelines
- Integration with Next.js theme system

**Subtasks**:
- Create design token definitions for colors, typography, and spacing
- Build Figma component library with design tokens
- Implement design token integration in Next.js components
- Create component documentation and usage guidelines
- Set up synchronization between Figma and code components

**Acceptance Criteria**:
- Design tokens are consistently applied across all components
- Figma library matches coded component styling exactly
- Documentation enables designers and developers to use system effectively
- Token changes propagate automatically through both design and code
- Brand consistency is maintained across all user interfaces

#### 2.121 Implement Supabase Auth (Reader, Author, Educator, Admin) ✅ **COMPLETED**
**Epic**: Multi-User Platform  
**Story**: Implement role-based authentication with Supabase  
**Labels**: backend, frontend, auth, supabase  
**Priority**: High  
**Status**: ✅ **Complete**  
**Completed**: 2025-01-09  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.11 (Next.js foundation)

**Requirements**:
- ✅ User registration and login flows with email verification
- ✅ Role-based access control (Reader, Author, Educator, Admin)
- ✅ Session management with JWT tokens and refresh logic
- ✅ Account dashboard and profile management
- ✅ Password reset and security features

**Subtasks**:
- ✅ Set up Supabase project and authentication configuration
- ✅ Implement user registration with email verification
- ✅ Create login/logout flows with session management
- ✅ Build role assignment and permission checking system
- ✅ Create account dashboard and profile management UI
- ✅ Implement password reset and security features

**Acceptance Criteria**:
- ✅ Users can register, verify email, and login successfully
- ✅ Role-based permissions restrict access appropriately
- ✅ Session management handles token refresh automatically
- ✅ Account dashboard allows profile updates and security settings
- ✅ All authentication flows handle errors gracefully

**Implementation Summary**:
- Created comprehensive Supabase authentication system with TypeScript support
- Implemented role-based access control with Reader, Author, Educator, and Admin roles
- Built complete auth UI components: login, register, password reset, email verification
- Created middleware for route protection and role-based access control
- Implemented profile management with settings dashboard and profile editor
- Added session management with JWT tokens and automatic refresh
- Created comprehensive test suite with 100% success rate on integration tests
- Built developer role-switching functionality for testing (dev mode only)
- Implemented secure password reset flow with email verification
- Added email verification with resend functionality
- Created extensive permission system with granular resource/action controls

#### 2.122 Add Role-Based Access Control Layer ✅ *COMPLETED*
**Epic**: Authentication & Permissions  
**Story**: Implement scoped permissions per user role (Reader, Author, Educator, Admin)  
**Labels**: backend, auth, rbac  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-07-09
**Estimated Effort**: 8 hours  
**Dependencies**: 2.121 (authentication foundation)

**Requirements**:
- ✅ Define permission matrix for each user role
- ✅ Implement middleware for endpoint access control
- ✅ Create role-specific UI component visibility
- ✅ Add audit logging for permission-sensitive operations
- ✅ Test role isolation and access restrictions

**Subtasks**:
- ✅ Design role-based permission matrix and data models
- ✅ Implement backend middleware for API endpoint protection
- ✅ Create frontend permission checking utilities and guards
- ✅ Build audit logging for security-sensitive operations
- ✅ Test permission enforcement across all user roles

**Acceptance Criteria**:
- ✅ Each role has appropriate access to features and data
- ✅ Unauthorized access attempts are blocked and logged
- ✅ Frontend components respect user permissions dynamically
- ✅ Permission changes take effect immediately
- ✅ Audit logs capture all permission-sensitive operations

**Implementation Details**:
- Created comprehensive permission matrix with 4 user roles (Reader, Author, Educator, Admin)
- Implemented extensible permission system with resource-action model
- Built FastAPI middleware for API endpoint protection with audit logging
- Created Streamlit-compatible frontend permission utilities with caching
- Established comprehensive audit logging system for all security events
- Delivered 500+ lines of test coverage across all RBAC components

#### 2.13 Create Public Domain Catalog (via Gutenberg ingestion) ✅ **COMPLETED**
**Epic**: Content Discovery  
**Story**: Build searchable catalog of public domain books
**Labels**: backend, content, gutenberg, catalog  
**Priority**: High  
**Status**: ✅ **Complete**  
**Completed**: 2025-07-12  
**Estimated Effort**: 25 hours  
**Dependencies**: 2.121 (authentication for user libraries)

**Requirements**:
- ✅ Public domain catalog with searchable database schema and RLS policies
- ✅ Search and filtering interface with language and text-based filters  
- ✅ Admin tools for catalog curation and metadata management
- ✅ "Add to My Library" functionality for authenticated users (existing from 2.11-2.12)
- ✅ Soft delete functionality with is_active field for content moderation

**Subtasks**:
- ✅ **2.131** Build search and filtering interface (Public catalog page)
- ✅ **2.132** Create admin interface for catalog management
- **2.133** Build Project Gutenberg ingestion pipeline (deferred to Phase 2.2)

**Implementation Summary**:
- **Public Catalog**: `/catalog-public` page with search by title/author and language filtering
- **Admin Interface**: `/admin/catalog` with role-based authentication, inline editing, and soft delete
- **Database Schema**: `public_books` table with comprehensive RLS policies for public read/admin write
- **Search Performance**: Client-side filtering with Supabase .ilike queries for responsive UX
- **Moderation Tools**: Soft delete via `is_active` field, admin-only visibility for inactive books

**Acceptance Criteria**:
- ✅ Catalog contains at least 20 unique public domain books with complete metadata
- ✅ Search and filtering provide relevant results quickly (<2s client-side filtering)
- ✅ Admin can curate catalog and edit metadata efficiently via inline editing
- ✅ Role-based access control restricts admin operations to admin users only
- ✅ Soft delete system allows content moderation without data loss

**Future Enhancements**:
- **Task 2.133**: Full Project Gutenberg ingestion pipeline (Phase 2.2)
- **Milestone 3.1.x/3.2.x**: Expand catalog to 500+ books for client beta testing

**Note**: Reader functionality and full-text ingestion have been scoped under Phase 2.2 (Tasks 2.23, 2.231, 2.232) to ensure clean separation of concerns.

#### 2.134 Add Ingestion QA Layer and Book Summary Generator ✅ **COMPLETED**
**Epic**: Public Domain Catalog  
**Story**: Auto-check and summarize new public domain books before publishing  
**Labels**: backend, ai, ingestion  
**Priority**: High  
**Status**: ✅ **Complete**  
**Completed**: 2025-07-12  
**Estimated Effort**: 10 hours  
**Dependencies**: 2.13 (catalog infrastructure)

**Requirements**:
- ✅ Automated content quality assessment for ingested books
- ✅ AI-powered book summary generation with quality validation
- ✅ Metadata extraction and enhancement pipeline
- ✅ Content filtering for inappropriate or duplicate material
- ✅ Quality scoring system for catalog curation

**Subtasks**:
- ✅ Implement content quality assessment algorithms
- ✅ Build AI-powered book summary generation system
- ✅ Create metadata extraction and enhancement pipeline
- ✅ Add content filtering and duplicate detection
- ✅ Build quality scoring and approval workflow

**Implementation Summary**:
- **QA Module**: Complete TypeScript CLI script at `scripts/ingestion-qa.ts`
- **Quality Scoring**: Multi-criteria assessment (length, readability, structure, content)
- **AI Summaries**: Claude API integration with fallback to simple summaries
- **Metadata Extraction**: Word count, chapter detection, language identification
- **Database Integration**: Added QA fields to `public_books` table with migration script
- **CLI Interface**: Supports batch processing and individual book QA
- **Testing**: Comprehensive test suite with sample books validation

**Acceptance Criteria**:
- ✅ All ingested books pass automated quality checks (scoring 0-100)
- ✅ AI-generated summaries are accurate and engaging (5-sentence format)
- ✅ Metadata extraction captures comprehensive book information (word/chapter count, language)
- ✅ Content filtering prevents inappropriate material from being published
- ✅ Quality scores enable efficient catalog curation (60+ threshold for approval)

#### 2.135 Implement Dataset QA Dashboard ✅ **COMPLETED**
**Epic**: Analytics & Admin Tools  
**Story**: Show ingestion stats, chunk quality, confidence scores per book  
**Labels**: backend, qa, analytics  
**Priority**: Medium  
**Status**: ✅ **Complete**  
**Completed**: 2025-07-12  
**Estimated Effort**: 8 hours  
**Dependencies**: 2.134 (QA layer foundation)

**Requirements**:
- ✅ Comprehensive ingestion statistics and metrics dashboard
- ✅ Book-level quality analytics and confidence scoring
- ✅ Chunk-level analysis and quality indicators
- ✅ Performance monitoring for ingestion pipeline
- ✅ Admin tools for content quality management

**Subtasks**:
- ✅ Build ingestion statistics collection and aggregation
- ✅ Create book-level quality analytics dashboard
- ✅ Implement chunk-level analysis and visualization
- ✅ Add performance monitoring for ingestion pipeline
- ✅ Build admin tools for quality management and oversight

**Implementation Summary**:
- **Dashboard Route**: `/admin/qa-dashboard` with role-based access control
- **Database Schema**: Added `book_chunks` and `book_content` tables with QA metrics
- **Ingestion Overview**: Stats cards, trend charts, quality distribution, and issues tracking
- **Book Quality Panel**: Sortable table with search, filtering, and pagination
- **Book Detail View**: Individual book analysis with chunk-level breakdown
- **Performance Monitoring**: Processing time analysis and optimization insights
- **Admin Tools**: Re-run QA, toggle approval, export problematic chunks
- **Interactive Charts**: Recharts integration for data visualization
- **Export Functionality**: CSV/JSON export of problematic chunks for manual review

**Acceptance Criteria**:
- ✅ Dashboard provides comprehensive view of content quality metrics (Overview panel with stats and charts)
- ✅ Book-level analytics help identify issues and improvements (Quality panel with filtering and sorting)
- ✅ Chunk-level analysis enables fine-grained quality control (Detail view with chunk table)
- ✅ Performance monitoring prevents ingestion bottlenecks (Performance panel with timing analysis)
- ✅ Admin tools enable efficient quality management workflow (Re-run QA, approval toggle, export)

#### 2.14 Migrate Content Database to Supabase (pgvector) ✅ **COMPLETED**
**Epic**: Infrastructure Migration  
**Story**: Migrate from Chroma to Supabase with pgvector for production scalability  
**Labels**: backend, database, migration, supabase, pgvector  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-07-12  
**Estimated Effort**: 22 hours  
**Dependencies**: 2.121 (multi-user data model)

**Requirements**:
- Design multi-tenant data schema with user isolation
- Migrate existing content and embeddings to Supabase
- Implement pgvector for semantic search capabilities
- Maintain search performance and relevance quality
- Add backup and disaster recovery procedures

**Subtasks**:
- Design multi-tenant database schema with user_id isolation
- Set up Supabase project with pgvector extension
- Create migration scripts for existing content and embeddings
- Implement search APIs using pgvector similarity search
- Migrate RAG service to use Supabase as vector database
- Set up backup and monitoring for production database

**Acceptance Criteria**:
- All existing content migrated successfully with data integrity
- Multi-tenant isolation prevents data leakage between users
- Search performance matches or exceeds Chroma benchmarks
- Vector similarity search maintains relevance quality
- Database includes proper indexes, backups, and monitoring

#### 2.15 Design & Implement Role-Based Permissions ✅ **COMPLETED**
**Epic**: Access Control  
**Story**: Implement granular permissions system for different user roles  
**Labels**: backend, permissions, rbac, security  
**Priority**: High  
**Status**: ✅ Complete  
**Completed**: 2025-07-12  
**Estimated Effort**: 15 hours  
**Dependencies**: 2.121 (authentication foundation)

**Requirements**:
- Define permission matrix for Reader/Author/Educator/Admin roles
- Implement middleware for route and resource protection
- Create organization support for educators and institutions
- Build permission checking utilities for frontend components
- Add audit logging for security-sensitive operations

**Subtasks**:
- Design role-based permission matrix and data models
- Implement authentication middleware with role checking
- Create organization membership and role assignment
- Build frontend permission checking hooks and components
- Implement audit logging for admin and security events
- Create role management interface for administrators

**Acceptance Criteria**:
- Each role has appropriate access to features and data
- Organization membership isolates data between institutions
- Frontend components respect user permissions dynamically
- Admin can manage roles and permissions through interface
- Security audit log captures all permission-sensitive operations

### 2.2 Client-Facing Site + Brand System 🎨
*Note: These tasks have marketing/design dependencies - see Marketing Dependencies section*

#### 2.21 Build Public-Facing Homepage (branding dependency) 📋
**Epic**: Marketing Site  
**Story**: Create compelling homepage that communicates Alexandria's value proposition  
**Labels**: frontend, marketing, branding, homepage  
**Priority**: Medium  
**Status**: Blocked (awaiting brand materials)  
**Estimated Effort**: 15 hours  
**Dependencies**: Brand strategy, visual identity, copy/messaging

**Requirements**:
- Hero section with clear value proposition and call-to-action
- Feature showcase with benefits for each user type
- Social proof section with testimonials and usage statistics
- Pricing tier comparison and signup flow integration
- Mobile-responsive design with fast loading times

**Subtasks**:
- Implement hero section with animated value proposition
- Build feature showcase with role-based messaging
- Create testimonial and social proof sections
- Integrate pricing comparison and signup flows
- Optimize for performance and SEO

**Acceptance Criteria**:
- Homepage clearly communicates value to each user segment
- Call-to-action buttons drive signups and trial conversions
- Mobile experience is optimized and loads quickly
- SEO meta tags and structured data are properly implemented
- Analytics tracking is set up for conversion optimization

#### 2.22 Add Marketing Landing Pages and Pricing Tiers 📋
**Epic**: Marketing Site  
**Story**: Create landing pages for different user segments and pricing  
**Labels**: frontend, marketing, pricing, conversion  
**Priority**: Medium  
**Status**: Blocked (awaiting pricing strategy)  
**Estimated Effort**: 12 hours  
**Dependencies**: Pricing strategy, user segment messaging

**Requirements**:
- Dedicated landing pages for Readers, Educators, and Organizations
- Pricing page with tier comparison and feature breakdown
- Free trial and signup flow optimization
- A/B testing framework for conversion optimization
- Integration with payment processing and subscription management

**Subtasks**:
- Build segment-specific landing pages with targeted messaging
- Create pricing page with interactive tier comparison
- Implement free trial signup and onboarding flows
- Set up A/B testing framework for page optimization
- Integrate Stripe for subscription and payment processing

**Acceptance Criteria**:
- Landing pages convert visitors to trial users effectively
- Pricing page clearly communicates value at each tier
- Free trial signup process has minimal friction
- A/B testing framework enables rapid iteration
- Payment integration works seamlessly for all subscription tiers

#### 2.23 Implement Onboarding/Signup UI 📋
**Epic**: User Experience  
**Story**: Create smooth onboarding experience for new users  
**Labels**: frontend, onboarding, ux, conversion  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.121 (authentication), role-specific messaging

**Requirements**:
- Multi-step onboarding flow based on user role selection
- Interactive tutorial highlighting key features
- Sample content and guided first actions
- Progress indicators and completion incentives
- Integration with Hypatia for personalized guidance

**Subtasks**:
- Design role-based onboarding flows and wireframes
- Implement multi-step signup with role selection
- Create interactive feature tutorials and tooltips
- Build sample content library for new user exploration
- Integrate Hypatia assistant for onboarding guidance

**Acceptance Criteria**:
- New users complete onboarding and take first meaningful action
- Onboarding is personalized based on selected user role
- Interactive tutorials effectively demonstrate key features
- Sample content helps users understand value immediately
- Progress tracking shows completion rates >80%

#### 2.24 Implement Theme Selector and Finalize Alexandria Classic Visuals 📋
**Epic**: Visual Design  
**Story**: Complete theme system with polished Alexandria Classic brand theme  
**Labels**: frontend, themes, branding, visual-design  
**Priority**: Low  
**Status**: To Do  
**Estimated Effort**: 8 hours  
**Dependencies**: Visual brand guide, color palette finalization

**Requirements**:
- Migrate theme system from Streamlit to Next.js components
- Polish Alexandria Classic theme with final brand colors and typography
- Add theme preview and customization options
- Implement smooth theme transitions and persistence
- Create theme documentation for brand consistency

**Subtasks**:
- Migrate theme system to Next.js with CSS-in-JS or CSS modules
- Finalize Alexandria Classic colors, typography, and visual elements
- Implement theme preview and live switching functionality
- Add custom theme creation tools for organizations
- Create theme documentation and usage guidelines

**Acceptance Criteria**:
- Theme switching works seamlessly in Next.js application
- Alexandria Classic theme reflects final brand identity accurately
- Theme persistence works across devices and sessions
- Theme preview helps users understand visual differences
- Documentation enables consistent theme usage across platform

#### 2.25 Build Annotation Engine with Export Functionality 📋
**Epic**: Reader Tools  
**Story**: Add notes, highlights, and export features per user and book  
**Labels**: frontend, backend, annotations  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 15 hours  
**Dependencies**: 2.121 (authentication), 2.14 (content database)

**Requirements**:
- User-specific annotation and highlighting system
- Rich text note-taking with formatting options
- Export annotations to multiple formats (PDF, Markdown, JSON)
- Synchronization across devices and sessions
- Integration with reading interface and search

**Subtasks**:
- Build user-specific annotation storage and retrieval system
- Implement rich text note-taking with formatting capabilities
- Create export functionality for multiple formats
- Add annotation synchronization across devices
- Integrate annotations with reading interface and search functionality

**Acceptance Criteria**:
- Users can create, edit, and delete annotations per book
- Annotations persist across sessions and devices
- Export functionality works reliably for all supported formats
- Rich text formatting is preserved in notes and exports
- Annotations integrate seamlessly with reading and search experience

#### 2.26 Research & Prototype Academic Integrations 📋
**Epic**: Learning Integrations  
**Story**: Research support for Zotero, ORCID, and citation management APIs  
**Labels**: integrations, research  
**Priority**: Low  
**Status**: To Do  
**Estimated Effort**: 6 hours  
**Dependencies**: None (research task)

**Requirements**:
- Research integration possibilities with Zotero citation management
- Investigate ORCID authentication and profile integration
- Evaluate academic citation format support (APA, MLA, Chicago)
- Prototype basic integration with at least one academic tool
- Document integration architecture and implementation plan

**Subtasks**:
- Research Zotero API capabilities and integration requirements
- Investigate ORCID authentication flow and profile data access
- Evaluate citation format libraries and academic standards
- Build prototype integration with one academic tool
- Document findings and create implementation roadmap

**Acceptance Criteria**:
- Research documentation covers all major academic integration options
- Prototype demonstrates feasibility of at least one integration
- Implementation plan provides clear path for academic features
- Technical requirements and constraints are well documented
- Integration architecture supports future academic tool additions

### 2.3 Ingestion & Book Storage 📚

#### 2.31 Full Text Ingestion & Preprocessing 📋
**Epic**: Content Processing  
**Story**: Ingest and prepare public domain book content for reading and AI  
**Labels**: backend, ingestion, content-processing  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.13 (public domain catalog), Project Gutenberg API access

**Objective**: Ingest and prepare public domain book content for reading and AI.

**Subtasks**:
- [ ] Pull raw text from Gutenberg/EPUB sources
- [ ] Clean and normalize the text (remove headers/footers)
- [ ] Store content in `public_books.book_text` or a linked table
- [ ] Chunk text into chapters or semantic units
- [ ] Add metadata fields: e.g., `word_count`, `chapter_count`

**Acceptance Criteria**:
- Raw text is successfully extracted from Gutenberg and EPUB sources
- Text is cleaned and normalized for consistent reading experience
- Content is properly stored with efficient retrieval capabilities
- Text is semantically chunked for optimal AI processing
- Metadata enriches book information with structural details

#### 2.32 Create Ingestion Tracker Table and Logging System 📋
**Epic**: Content Ingestion  
**Story**: Record metadata about all book ingestion events and pipelines  
**Labels**: backend, ingestion, logging  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 5 hours  
**Dependencies**: 2.21

**Requirements**:
- Table to track ingestion attempts, source, method, user, and timestamps
- Log failures, success, retry history
- Track who/what ingested the book (manual, script, pipeline)
- Store ingestion metadata (source URL, file type, text word count)

**Acceptance Criteria**:
- Ingestion pipeline logs every action to database
- Admin can review what books were ingested, how, and by whom
- Failed ingestions are clearly labeled for retry/debugging

#### 2.33 Add Licensing and Attribution Fields to Book Metadata 📋
**Epic**: Metadata Infrastructure  
**Story**: Track copyright status and license of all content  
**Labels**: backend, metadata, ingestion  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 3 hours  
**Dependencies**: 2.22

**Requirements**:
- Add fields to `public_books`: `license`, `attribution`, `source_url`
- Parse and extract from ingestion sources (Gutenberg, Archive, etc.)
- Display basic license info in frontend (catalog/book page)

**Acceptance Criteria**:
- Every book has a license (e.g. Public Domain, CC-BY)
- Attribution is stored or displayed where needed
- Prevents accidental ingestion of non-permissible works

#### 2.34 Define Book ID Strategy and Normalization Rules 📋
**Epic**: Ingestion Scaling  
**Story**: Prevent ID collisions across ingestion sources  
**Labels**: backend, architecture  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 4 hours  
**Dependencies**: 2.21

**Requirements**:
- Normalize `book_id` format for all sources
- Prevent duplicate UUIDs across manual vs. auto ingestion
- Add `source_id` + `origin_platform` fields to track external sources

**Acceptance Criteria**:
- Every book has a globally unique ID
- External sources can be deduplicated
- Future ingestion at scale is safe and traceable

#### 2.35 Define and Implement Chunking Strategy for Book Content 📋
**Epic**: AI & Embeddings  
**Story**: Improve AI quality and retrieval with consistent chunking  
**Labels**: backend, ai, RAG  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 5 hours  
**Dependencies**: 2.25 (annotation system)

**Requirements**:
- Define chunking rules: paragraph-based, semantic, or fixed-length
- Normalize paragraph breaks and punctuation
- Ensure overlap or context padding between chunks
- Annotate chunks with position metadata

**Acceptance Criteria**:
- Every chunk follows a consistent format
- Hypatia returns more accurate and complete answers
- Chunk overlaps improve context retention

#### 2.36 Handle Re-ingestion and Versioning of Book Content 📋
**Epic**: Data Integrity  
**Story**: Prevent overwrites and track updates to books  
**Labels**: ingestion, updates, backend  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 6 hours  
**Dependencies**: 2.32 (ingestion logging), 2.25 (annotation system)

**Requirements**:
- Add `version` field to `public_books` or create `book_versions` table
- Detect re-ingestion of the same book
- Append new versions instead of overwrite
- Compare metadata to flag updates
- Update QA and vector re-embedding pipeline to support changes

**Acceptance Criteria**:
- Updated books preserve history or version metadata
- No data loss during re-ingestion
- Hypatia always references latest version or user-defined version

#### 2.37 Build In-App Reader View 📋
**Epic**: Reading Experience  
**Story**: Allow users to read full books directly in the browser  
**Labels**: frontend, reader, book-content  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 20 hours  
**Dependencies**: 2.14 (content database migration), authentication system

**Objective**: Allow users to read full books directly in the browser.

**Subtasks**:
- [ ] Add `book_text` or separate `book_content` table for raw text storage
- [ ] Create `/read/[bookId]` page with responsive reader UI
- [ ] Implement features: font size toggle, light/dark mode, progress tracker
- [ ] Persist reading state (last read position) in `user_books`
- [ ] Ensure access control to only serve full text to authenticated users

**Acceptance Criteria**:
- Users can read full book content in a responsive web interface
- Reading preferences (font size, theme) are saved per user
- Reading progress is tracked and persisted across sessions
- Access control ensures only authenticated users can read full text
- Reader interface works smoothly on desktop and mobile devices

#### 🆕 2.252 Full Text Ingestion & Preprocessing 📋
**Epic**: Content Processing  
**Story**: Ingest and prepare public domain book content for reading and AI  
**Labels**: backend, ingestion, content-processing  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.13 (public domain catalog), Project Gutenberg API access

**Objective**: Ingest and prepare public domain book content for reading and AI.

**Subtasks**:
- [ ] Pull raw text from Gutenberg/EPUB sources
- [ ] Clean and normalize the text (remove headers/footers)
- [ ] Store content in `public_books.book_text` or a linked table
- [ ] Chunk text into chapters or semantic units
- [ ] Add metadata fields: e.g., `word_count`, `chapter_count`

**Acceptance Criteria**:
- Raw text is successfully extracted from Gutenberg and EPUB sources
- Text is cleaned and normalized for consistent reading experience
- Content is properly stored with efficient retrieval capabilities
- Text is semantically chunked for optimal AI processing
- Metadata enriches book information with structural details

#### 🆕 2.253 Embedding Pipeline for Hypatia (RAG Prep) 📋
**Epic**: AI Infrastructure  
**Story**: Enable AI retrieval for Hypatia by embedding book content  
**Labels**: backend, ai, embeddings, rag  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 22 hours  
**Dependencies**: 2.252 (text preprocessing), 2.14 (content database), vector storage system

**Objective**: Enable AI retrieval for Hypatia by embedding book content.

**Subtasks**:
- [ ] Chunk content into indexable blocks
- [ ] Generate embeddings and store them (pgvector or vector DB)
- [ ] Build API endpoints to query book content via embeddings
- [ ] Add volume-level and chapter-level memory integration for Hypatia

**Acceptance Criteria**:
- Content is chunked into optimal blocks for embedding generation
- Embeddings are generated and stored efficiently for fast retrieval
- API endpoints enable semantic search across book content
- Hypatia can access and reference book content in conversations
- Memory integration preserves context across reading and chat sessions

*Tasks 2.31–2.38 reordered to ensure proper ingestion workflow: content processing → logging → metadata → chunking → embedding → reader interface.*

### 2.4 Document Reader & UX System 📖

#### 2.41 Build Advanced Digital Book Reader (annotations, highlights) 📋
**Epic**: Reading Experience  
**Story**: Create immersive digital reading experience with annotation tools  
**Labels**: frontend, reader, annotations, ux  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 28 hours  
**Dependencies**: 2.14 (content database migration)

**Requirements**:
- Full-screen reading interface with customizable typography
- Highlighting and annotation tools with personal note-taking
- Progress tracking and reading statistics integration
- Keyboard shortcuts and accessibility features
- Synchronized reading position across devices

**Subtasks**:
- Build responsive reading interface with typography controls
- Implement text selection, highlighting, and annotation tools
- Create personal note-taking and organization system
- Add progress tracking and reading analytics integration
- Implement keyboard shortcuts and accessibility features
- Build reading position synchronization across devices

**Acceptance Criteria**:
- Reading interface provides comfortable, distraction-free experience
- Annotation tools are intuitive and support rich text formatting
- Reading progress syncs automatically across all user devices
- Keyboard navigation and screen readers work properly
- Performance remains smooth with large documents and many annotations

#### 2.42 Add Semantic Page Navigation and Chapter Preview 📋
**Epic**: Reading Experience  
**Story**: Intelligent navigation system based on document structure  
**Labels**: frontend, navigation, ai, content-analysis  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.41 (reader foundation), enhanced content analysis

**Requirements**:
- AI-powered chapter and section detection
- Interactive table of contents with progress indicators
- Chapter preview with summaries and key concepts
- Semantic search within documents
- Smart bookmarking with context preservation

**Subtasks**:
- Implement AI-powered document structure analysis
- Build interactive table of contents with progress visualization
- Create chapter preview with AI-generated summaries
- Add semantic search functionality within documents
- Implement smart bookmarking with context and notes

**Acceptance Criteria**:
- Document structure is automatically detected and navigable
- Chapter previews help users understand content organization
- Semantic search finds relevant passages effectively
- Bookmarks preserve context and support personal organization
- Navigation performance remains fast with large documents

#### 2.43 Support Notes Export and Printable Format 📋
**Epic**: Content Management  
**Story**: Enable users to export and share their reading notes and highlights  
**Labels**: frontend, export, pdf, sharing  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 12 hours  
**Dependencies**: 2.25 (annotation system)

**Requirements**:
- Export annotations and highlights to multiple formats (PDF, Markdown, DOCX)
- Printable summary reports with reading statistics
- Shareable note collections with privacy controls
- Integration with external note-taking apps (Notion, Obsidian)
- Batch export and organization tools

**Subtasks**:
- Implement multi-format export for annotations and highlights
- Create printable reading reports with statistics and insights
- Build sharing functionality with privacy and permission controls
- Add integration APIs for external note-taking applications
- Create batch export and organization tools for power users

**Acceptance Criteria**:
- Exports preserve formatting and include relevant context
- Printable reports provide valuable reading insights and statistics
- Sharing controls allow selective privacy and collaboration
- External integrations work seamlessly with popular tools
- Batch operations handle large annotation collections efficiently

#### 2.44 Test Reader on Mobile 📋
**Epic**: Mobile Experience  
**Story**: Ensure reading experience is optimized for mobile devices  
**Labels**: frontend, mobile, testing, responsive  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 10 hours  
**Dependencies**: 2.41 (reader foundation)

**Requirements**:
- Touch-friendly interface with gesture navigation
- Mobile-optimized typography and layout
- Performance optimization for slower devices
- Offline reading capabilities with sync
- Mobile-specific annotation and highlighting tools

**Subtasks**:
- Optimize reading interface for touch interactions and gestures
- Test and tune typography and layout for mobile screens
- Implement performance optimizations for mobile devices
- Add offline reading with background synchronization
- Create mobile-optimized annotation and highlighting tools

**Acceptance Criteria**:
- Reading experience feels native and responsive on mobile
- Touch gestures for navigation and interaction work intuitively
- Performance is smooth on mid-range mobile devices
- Offline reading works reliably with automatic sync
- Mobile annotation tools are easy to use with touch input

### 2.4 Admin Dashboard & Public Library Control ⚙️

#### 2.41 Admin Dashboard to View and Delete All Content 📋
**Epic**: Content Management  
**Story**: Administrative interface for platform content oversight  
**Labels**: backend, admin, content-management, dashboard  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.15 (role-based permissions)

**Requirements**:
- Comprehensive content dashboard with search and filtering
- Bulk operations for content management and moderation
- User content overview and privacy compliance tools
- Analytics dashboard for platform usage and health
- Security monitoring and audit log visualization

**Subtasks**:
- Build admin-only content dashboard with advanced search
- Implement bulk operations for content deletion and moderation
- Create user data management tools for privacy compliance
- Add platform analytics and usage monitoring dashboard
- Build security monitoring and audit log interface

**Acceptance Criteria**:
- Admins can efficiently search, view, and manage all platform content
- Bulk operations enable efficient content moderation and cleanup
- Privacy compliance tools support user data requests
- Analytics provide insights into platform usage and health
- Security monitoring helps identify and respond to threats

#### 2.42 Deduplication Tooling for Public Domain Books 📋
**Epic**: Content Quality  
**Story**: Automated tools to identify and manage duplicate content  
**Labels**: backend, deduplication, content-quality, automation  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 14 hours  
**Dependencies**: 2.13 (public domain catalog)

**Requirements**:
- Automated duplicate detection using content similarity
- Manual review interface for edge cases and decisions
- Merge functionality to consolidate metadata and user data
- Quality scoring system for different versions of same content
- Integration with content ingestion pipeline

**Subtasks**:
- Implement content similarity detection using embeddings
- Build manual review interface for duplicate candidates
- Create merge functionality to consolidate duplicate entries
- Add quality scoring to identify best version of duplicated content
- Integrate deduplication checks into content ingestion workflow

**Acceptance Criteria**:
- Duplicate detection identifies similar content with high accuracy
- Manual review interface enables efficient duplicate management
- Merge functionality preserves user data and maintains references
- Quality scoring helps select best version of duplicate content
- Automated pipeline prevents most duplicates from entering catalog

#### 2.43 Metadata Editor with AI Suggestions 📋
**Epic**: Content Enhancement  
**Story**: Tools for improving content metadata with AI assistance  
**Labels**: backend, metadata, ai, content-enhancement  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.13 (catalog system), AI content analysis

**Requirements**:
- AI-powered metadata generation and enhancement
- Manual editing interface with validation and suggestions
- Batch operations for metadata updates across collections
- Version history and approval workflow for changes
- Integration with external metadata sources (Library of Congress, WorldCat)

**Subtasks**:
- Implement AI-powered metadata extraction and enhancement
- Build manual metadata editing interface with validation
- Create batch update tools for metadata management
- Add version history and approval workflow for metadata changes
- Integrate external metadata sources for validation and enrichment

**Acceptance Criteria**:
- AI suggestions improve metadata quality and completeness
- Manual editing interface is efficient and prevents errors
- Batch operations enable large-scale metadata improvements
- Version history and approval workflow maintain quality control
- External integrations provide authoritative metadata validation

#### 2.44 User Management Interface 📋
**Epic**: User Administration  
**Story**: Administrative tools for user account and role management  
**Labels**: backend, frontend, user-management, admin  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.15 (role-based permissions)

**Requirements**:
- User search and filtering with advanced criteria
- Role assignment and permission management tools
- Account status management (active, suspended, deleted)
- Usage analytics and activity monitoring per user
- Bulk operations for user management tasks

**Subtasks**:
- Build user search and filtering interface with advanced criteria
- Implement role assignment and permission management tools
- Create account status management with reason tracking
- Add per-user usage analytics and activity monitoring
- Build bulk operations for common user management tasks

**Acceptance Criteria**:
- Admin can efficiently find and manage user accounts
- Role and permission changes are applied immediately and correctly
- Account status changes are logged with reasons and history
- User analytics provide insights into engagement and behavior
- Bulk operations enable efficient management of large user bases

#### 2.45 Build Admin Moderation Panel 📋
**Epic**: Trust Layer  
**Story**: Admin UI to approve, remove, or flag problematic content or users  
**Labels**: admin, trust, dashboard  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 6 hours  
**Dependencies**: 2.122 (role-based access control), 2.44 (user management)

**Requirements**:
- Content moderation interface with approval workflows
- User moderation tools for account management
- Flagging system for problematic content and behavior
- Automated content filtering with manual review options
- Audit trail for all moderation actions

**Subtasks**:
- Build content moderation interface with bulk operations
- Create user moderation tools with status management
- Implement flagging and reporting system
- Add automated content filtering with manual overrides
- Build comprehensive audit trail for moderation decisions

**Acceptance Criteria**:
- Admins can efficiently review and moderate content
- User account management tools enable appropriate interventions
- Flagging system captures and prioritizes problematic behavior
- Automated filtering reduces manual moderation workload
- All moderation actions are logged with detailed attribution

### 2.5 Hypatia Assistant 1.0 🤖

#### 2.51 Hypatia Chatbot Frontend UI 📋
**Epic**: Conversational AI  
**Story**: Engaging chat interface for Hypatia assistant  
**Labels**: frontend, ai, chat, hypatia  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.11 (Next.js foundation)

**Requirements**:
- Modern chat interface with typing indicators and message history
- Context-aware conversation threading and topic management
- Quick action buttons for common tasks and queries
- Integration with reading interface for contextual assistance
- Accessibility features for screen readers and keyboard navigation

**Subtasks**:
- Design and implement modern chat interface with smooth animations
- Build conversation threading and topic management system
- Create quick action buttons for common user tasks
- Integrate chat interface with reading and library interfaces
- Implement accessibility features and keyboard navigation

**Acceptance Criteria**:
- Chat interface feels modern, responsive, and engaging
- Conversation history is preserved and easily navigable
- Quick actions reduce friction for common tasks
- Integration with other interfaces feels seamless and contextual
- Accessibility standards are met for all users

#### 2.52 Basic Persona Memory and Helpful Tone 📋
**Epic**: AI Personality  
**Story**: Establish Hypatia's personality and conversation memory  
**Labels**: backend, ai, personality, memory  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 12 hours  
**Dependencies**: Personality guide and voice documentation

**Requirements**:
- Consistent personality traits across all conversations
- Memory of user preferences, reading history, and past interactions
- Adaptive tone based on user expertise and context
- Helpful and encouraging responses that motivate learning
- Cultural sensitivity and inclusive language patterns

**Subtasks**:
- Implement personality configuration and consistency checking
- Build conversation memory and user preference tracking
- Create adaptive tone and response generation based on context
- Design helpful and encouraging response patterns
- Implement cultural sensitivity and inclusive language filters

**Acceptance Criteria**:
- Hypatia maintains consistent personality across all interactions
- Conversation memory enhances user experience with personalization
- Tone adapts appropriately to user expertise and emotional state
- Responses are consistently helpful and encourage continued learning
- Language patterns are inclusive and culturally sensitive

#### 2.53 Intelligent Prompt Routing to Q&A / Search 📋
**Epic**: Conversational AI  
**Story**: Smart routing of user queries to appropriate backend services  
**Labels**: backend, ai, routing, intent-classification  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 15 hours  
**Dependencies**: 2.51 (chat interface), enhanced RAG system

**Requirements**:
- Intent classification for different query types (Q&A, search, help, navigation)
- Context-aware routing based on current user activity
- Seamless handoff between conversation and search results
- Fallback handling for ambiguous or unclear requests
- Integration with existing RAG and search infrastructure

**Subtasks**:
- Implement intent classification system for query routing
- Build context-aware routing logic based on user state
- Create seamless handoff between chat and search interfaces
- Add fallback handling for unclear or complex requests
- Integrate routing system with existing RAG and search APIs

**Acceptance Criteria**:
- Intent classification accurately routes queries to appropriate services
- Context awareness improves routing decisions based on user activity
- Handoffs between conversation and search feel natural and helpful
- Fallback handling gracefully manages ambiguous requests
- Integration maintains performance and response quality

#### 2.54 Log Queries and Responses for QA Refinement 📋
**Epic**: AI Quality Assurance  
**Story**: Data collection and analysis for improving Hypatia's responses  
**Labels**: backend, logging, analytics, qa  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 10 hours  
**Dependencies**: 2.53 (query routing), privacy compliance framework

**Requirements**:
- Comprehensive logging of queries, responses, and user feedback
- Analytics dashboard for identifying improvement opportunities
- Privacy-compliant data collection with user consent
- A/B testing framework for response optimization
- Integration with feedback collection and rating systems

**Subtasks**:
- Implement comprehensive query and response logging system
- Build analytics dashboard for conversation quality analysis
- Create privacy-compliant data collection with user consent
- Set up A/B testing framework for response optimization
- Integrate feedback collection and rating systems

**Acceptance Criteria**:
- All conversations are logged with appropriate privacy protections
- Analytics provide actionable insights for improving response quality
- Data collection complies with privacy regulations and user preferences
- A/B testing enables systematic improvement of responses
- Feedback systems provide user input for quality improvements

#### 2.55 Add AI Attribution & Provenance Metadata Layer 📋
**Epic**: Hypatia & Trust Layer  
**Story**: Every AI-generated answer must include metadata tags (model, source, timestamp)  
**Labels**: ai, backend, trust  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 5 hours  
**Dependencies**: 2.53 (query routing), enhanced RAG system

**Requirements**:
- Metadata tagging for all AI-generated responses
- Model attribution (GPT version, provider, parameters)
- Source citation and confidence scoring
- Timestamp and provenance tracking
- User-facing transparency indicators

**Subtasks**:
- Implement metadata collection system for AI responses
- Add model attribution and parameter tracking
- Build source citation and confidence scoring system
- Create timestamp and provenance tracking infrastructure
- Design user-facing transparency and attribution display

**Acceptance Criteria**:
- All AI responses include complete metadata attribution
- Model information is accurately captured and displayed
- Source citations are comprehensive and traceable
- Confidence scores reflect actual response reliability
- Users can easily access and understand AI provenance information

### 2.6 RAG Upgrade & Semantic Infrastructure 🧠

#### 2.61 Upgrade Vector Metadata with Semantic Tags 📋
**Epic**: Content Intelligence  
**Story**: Enhance content discoverability with AI-generated semantic metadata  
**Labels**: backend, ai, metadata, semantic-analysis  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 20 hours  
**Dependencies**: 2.14 (Supabase migration)

**Requirements**:
- AI-powered semantic tag extraction from content
- Hierarchical tag taxonomy for improved organization
- Topic modeling and concept clustering
- Automated tag validation and quality control
- Integration with search and recommendation systems

**Subtasks**:
- Implement AI-powered semantic tag extraction pipeline
- Design hierarchical tag taxonomy and validation rules
- Build topic modeling and concept clustering algorithms
- Create automated tag quality control and validation system
- Integrate semantic tags with search and recommendation engines

**Acceptance Criteria**:
- Semantic tags improve content discoverability and search relevance
- Tag taxonomy provides logical organization and navigation
- Topic clustering reveals meaningful content relationships
- Quality control ensures tag accuracy and consistency
- Integration enhances search and recommendation performance

#### 2.62 Implement Hybrid Re-ranking (keyword + vector) 📋
**Epic**: Search Quality  
**Story**: Advanced search combining multiple ranking signals for optimal results  
**Labels**: backend, search, ranking, hybrid  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.61 (semantic metadata), BM25 indexing

**Requirements**:
- Fusion of vector similarity and keyword relevance scores
- Learning-to-rank algorithms for personalized results
- Query expansion and synonym handling
- Real-time ranking with sub-second response times
- A/B testing framework for ranking optimization

**Subtasks**:
- Implement score fusion algorithms for vector and keyword signals
- Build learning-to-rank system with user behavior data
- Add query expansion and intelligent synonym handling
- Optimize ranking pipeline for real-time performance
- Create A/B testing framework for ranking experiments

**Acceptance Criteria**:
- Hybrid ranking provides better results than individual methods
- Personalization improves search relevance for individual users
- Query expansion handles synonyms and related terms effectively
- Search performance meets sub-second response time requirements
- A/B testing enables continuous ranking improvement

#### 2.63 Validate Confidence Scores and Hallucination Boundaries 📋
**Epic**: AI Safety  
**Story**: Robust confidence measurement and hallucination detection  
**Labels**: backend, ai, safety, confidence  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: Enhanced RAG system, evaluation framework

**Requirements**:
- Calibrated confidence scores that reflect actual accuracy
- Hallucination detection and prevention mechanisms
- Uncertainty quantification for AI-generated responses
- User-friendly confidence indicators and warnings
- Continuous monitoring and improvement of AI safety

**Subtasks**:
- Implement calibrated confidence scoring for AI responses
- Build hallucination detection using multiple validation methods
- Create uncertainty quantification and user communication
- Design user-friendly confidence indicators and safety warnings
- Set up continuous monitoring and safety improvement pipeline

**Acceptance Criteria**:
- Confidence scores accurately predict response reliability
- Hallucination detection prevents misleading or incorrect responses
- Uncertainty quantification helps users make informed decisions
- Safety indicators are clear and actionable for users
- Monitoring system identifies and addresses safety issues proactively

#### 2.64 Explore Document-Level Memory Bundling for Topic Coherence 📋
**Epic**: Content Intelligence  
**Story**: Improved context preservation through intelligent content grouping  
**Labels**: backend, ai, memory, context  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 14 hours  
**Dependencies**: 2.61 (semantic infrastructure)

**Requirements**:
- Document-level context preservation across conversations
- Topic coherence maintenance in multi-turn interactions
- Intelligent context window management for long documents
- Cross-document relationship preservation
- Memory efficiency optimization for large content libraries

**Subtasks**:
- Implement document-level context bundling and preservation
- Build topic coherence tracking across conversation turns
- Create intelligent context window management for long content
- Add cross-document relationship tracking and recall
- Optimize memory usage for large-scale content processing

**Acceptance Criteria**:
- Document context is preserved throughout extended conversations
- Topic coherence is maintained across multiple interaction turns
- Context management handles long documents efficiently
- Cross-document relationships enhance answer quality
- Memory optimization supports large content libraries without degradation

### 2.7 Learning Suite Foundation 🎓

#### 2.71 AI-Powered Course Builder Backend 📋
**Epic**: Course Creation  
**Story**: Automated course generation from existing content with AI assistance  
**Labels**: backend, ai, course-creation, lms  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 25 hours  
**Dependencies**: 2.14 (content database), 2.61 (semantic infrastructure)

**Requirements**:
- Automated course outline generation from book content
- AI-powered lesson planning with learning objectives
- Quiz and assessment creation using content analysis
- Multimedia content integration and management
- Course template system with customization options

**Subtasks**:
- Build AI-powered course outline generation from content analysis
- Implement lesson planning with automatic learning objective extraction
- Create quiz and assessment generation using content comprehension
- Add multimedia content integration and management system
- Build course template system with educator customization options

**Acceptance Criteria**:
- Course outlines are pedagogically sound and comprehensive
- Learning objectives align with content and educational standards
- Generated quizzes effectively test content comprehension
- Multimedia integration enhances learning experience
- Template system enables rapid course creation with customization

#### 2.72 Student Learning Journey Tracking 📋
**Epic**: Learning Analytics  
**Story**: Comprehensive tracking of student progress and learning outcomes  
**Labels**: backend, analytics, learning-tracking, progress  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 22 hours  
**Dependencies**: 2.71 (course structure), 2.15 (permissions)

**Requirements**:
- Detailed progress tracking across courses and content
- Learning outcome measurement and competency mapping
- Adaptive pathways based on individual student performance
- Real-time progress visualization and reporting
- Integration with external LMS and gradebook systems

**Subtasks**:
- Implement comprehensive progress tracking data models
- Build learning outcome measurement and competency mapping
- Create adaptive pathway algorithms based on performance data
- Add real-time progress visualization and reporting dashboards
- Build integration APIs for external LMS and gradebook systems

**Acceptance Criteria**:
- Progress tracking provides detailed insights into student learning
- Competency mapping aligns with educational standards and objectives
- Adaptive pathways personalize learning experience effectively
- Visualizations help students and educators understand progress
- External integrations work seamlessly with popular LMS platforms

#### 2.73 Smart Flashcards and Spaced Repetition Prototype 📋
**Epic**: Learning Tools  
**Story**: AI-powered study tools with spaced repetition algorithms  
**Labels**: frontend, backend, learning-tools, spaced-repetition  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.71 (course content), learning science research

**Requirements**:
- AI-generated flashcards from course content and highlights
- Spaced repetition algorithms optimized for long-term retention
- Progress tracking and performance analytics for study sessions
- Mobile-optimized interface for on-the-go studying
- Integration with reading annotations and course materials

**Subtasks**:
- Implement AI-powered flashcard generation from content analysis
- Build spaced repetition algorithms with optimal scheduling
- Create progress tracking and analytics for study session performance
- Design mobile-optimized study interface with smooth interactions
- Integrate flashcards with reading annotations and course content

**Acceptance Criteria**:
- Flashcards effectively reinforce key concepts from content
- Spaced repetition improves long-term retention measurably
- Progress tracking motivates continued study and improvement
- Mobile interface enables effective studying in short sessions
- Integration creates seamless flow from reading to studying

### 2.8 Cross-Cutting Upgrades 🔄

#### 2.81 Content Versioning and Changelog System 📋
**Epic**: Content Management  
**Story**: Track content changes and maintain version history  
**Labels**: backend, versioning, content-management, audit  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.14 (content database), 2.15 (permissions)

**Requirements**:
- Version control for all content changes and updates
- Change tracking with author attribution and timestamps
- Rollback capabilities for content and metadata
- Visual diff interface for comparing versions
- Integration with content approval and publishing workflows

**Subtasks**:
- Implement version control system for content and metadata
- Build change tracking with attribution and detailed logging
- Create rollback functionality with validation and safety checks
- Add visual diff interface for version comparison
- Integrate versioning with approval and publishing workflows

**Acceptance Criteria**:
- All content changes are tracked with complete version history
- Change attribution provides accountability and audit trail
- Rollback functionality enables safe content management
- Visual diffs make version comparison clear and actionable
- Workflow integration maintains content quality and control

#### 2.82 Role-Based Access Control Implementation 📋
**Epic**: Security  
**Story**: Comprehensive RBAC system across all platform features  
**Labels**: backend, security, rbac, permissions  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 20 hours  
**Dependencies**: 2.15 (permissions foundation), all feature modules

**Requirements**:
- Granular permissions for all platform features and resources
- Dynamic role assignment with organization context
- Permission inheritance and delegation mechanisms
- Audit logging for all permission-sensitive operations
- Performance optimization for permission checking at scale

**Subtasks**:
- Implement granular permission system for all features
- Build dynamic role assignment with organizational context
- Create permission inheritance and delegation mechanisms
- Add comprehensive audit logging for security operations
- Optimize permission checking performance for large user bases

**Acceptance Criteria**:
- Permissions are enforced consistently across all platform features
- Role assignment adapts to organizational needs and hierarchies
- Permission inheritance reduces management overhead while maintaining security
- Audit logs provide complete trail of security-sensitive operations
- Performance remains fast even with complex permission structures

#### 2.83 Organization Support for Educators 📋
**Epic**: Multi-Tenancy  
**Story**: Organization management for educational institutions  
**Labels**: backend, frontend, organizations, multi-tenancy  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 22 hours  
**Dependencies**: 2.15 (permissions), 2.82 (RBAC)

**Requirements**:
- Organization creation and management interface
- Multi-tenant data isolation and security
- Role hierarchy within organizations (admin, teacher, student)
- Billing and subscription management at organization level
- Usage analytics and reporting for organizational administrators

**Subtasks**:
- Build organization creation and management interface
- Implement multi-tenant data isolation and security controls
- Create organizational role hierarchy and permission mapping
- Add organization-level billing and subscription management
- Build usage analytics and reporting for organizational insights

**Acceptance Criteria**:
- Organizations can be created and managed independently
- Data isolation prevents access across organizational boundaries
- Role hierarchies enable appropriate delegation and management
- Billing and subscriptions work seamlessly at organizational level
- Analytics provide valuable insights for organizational decision-making

#### 2.84 Reader Activity Log for Analytics 📋
**Epic**: Analytics  
**Story**: Detailed tracking of reading behavior and engagement  
**Labels**: backend, analytics, reading-behavior, tracking  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 12 hours  
**Dependencies**: 2.41 (reader interface), privacy compliance

**Requirements**:
- Comprehensive tracking of reading behavior and engagement
- Privacy-compliant data collection with user consent
- Real-time analytics dashboard for reading insights
- Integration with learning analytics and course progress
- Anonymized data export for research and platform improvement

**Subtasks**:
- Implement comprehensive reading behavior tracking system
- Build privacy-compliant data collection with granular user consent
- Create real-time analytics dashboard for reading insights
- Integrate reading analytics with learning progress tracking
- Add anonymized data export capabilities for platform improvement

**Acceptance Criteria**:
- Reading behavior is tracked comprehensively while respecting privacy
- Analytics provide actionable insights for users and educators
- Data collection complies with privacy regulations and user preferences
- Integration enhances learning analytics and progress tracking
- Anonymized data supports platform improvement and research

### 🔐 2.9 Platform Security Foundations (Improved)

**Epic**: AI Safety, Privacy, and Abuse Prevention  
**Status**: 🔒 Critical Pre-Launch Milestone  
**Priority**: High  
**Total Estimated Effort**: 95 hours across 9 security epics

#### 2.91 AI Upload Safety Pipeline 📋
**Epic**: Content Security  
**Story**: Sanitize uploaded content to prevent prompt injection and LLM exploits  
**Labels**: backend, security, ai-safety, content-filtering  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 18 hours  
**Dependencies**: 2.121 (authentication), content ingestion pipeline

**Subtasks**:
- **2.91.1** Sanitize PDFs, DOCX (strip layers, chunk, flatten) - 8 hours
- **2.91.2** Detect adversarial prompt structures using pattern-matching + ML - 6 hours  
- **2.91.3** Rate-limit uploads and validate MIME/file-type boundaries - 4 hours

**Acceptance Criteria**:
- All uploaded documents undergo multi-stage sanitization
- Adversarial prompt patterns are detected and blocked
- Upload rate limits prevent abuse and system overload
- MIME type validation prevents malicious file uploads

#### 2.92 Annotation & Forum Moderation Filters 📋
**Epic**: Content Moderation  
**Story**: Content filtering for user-generated annotations and community posts  
**Labels**: backend, moderation, content-filtering, community-safety  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 15 hours  
**Dependencies**: 2.231 (annotation system), community features

**Subtasks**:
- **2.92.1** Moderate prompt injection, profane, or manipulative language - 6 hours
- **2.92.2** Shadow-ban filter + manual flag review workflow - 5 hours
- **2.92.3** Community reporting + appeals system - 4 hours

**Acceptance Criteria**:
- User-generated content is automatically screened for harmful material
- Shadow-ban system prevents spam without alerting bad actors
- Manual review workflow enables human oversight of flagged content
- Community reporting provides user-driven content moderation

#### 2.93 Agent Tool Invocation Guardrails 📋
**Epic**: AI Tool Security  
**Story**: Secure AI agent tool invocation with validation and sandboxing  
**Labels**: backend, ai-safety, tool-security, agent-protection  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 12 hours  
**Dependencies**: 2.53 (query routing), agent tool system

**Subtasks**:
- **2.93.1** Validate tool name + parameter scope against whitelist - 4 hours
- **2.93.2** Add guard layer for prompt-suggested functions - 4 hours
- **2.93.3** Sanitize arguments before passing to tool execution layer - 4 hours

**Acceptance Criteria**:
- All tool invocations are validated against approved whitelist
- Prompt-suggested tool calls are blocked or require explicit approval
- Tool arguments are sanitized to prevent injection attacks
- Tool execution is properly sandboxed and logged

#### 2.94 Prompt Wrapping & Role Separation 📋
**Epic**: Prompt Security  
**Story**: Implement prompt wrapping and role separation for LLM interactions  
**Labels**: backend, ai-safety, prompt-engineering, role-isolation  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 14 hours  
**Dependencies**: 2.53 (query routing), enhanced RAG system

**Subtasks**:
- **2.94.1** Mark user input, memory, documents with origin tags (`[UNTRUSTED_INPUT]`) - 5 hours
- **2.94.2** Wrap prompts with system guardrails and content boundaries - 5 hours
- **2.94.3** Add AI output wrapping with attribution metadata - 4 hours

**Acceptance Criteria**:
- All user content is tagged with appropriate trust levels
- System prompts maintain clear separation between trusted and untrusted content
- AI outputs include complete attribution and source metadata
- Content boundaries prevent prompt injection and role confusion

#### 2.95 Memory Trust Isolation & Exportability 📋
**Epic**: Memory Security  
**Story**: Comprehensive memory isolation with trust levels and user controls  
**Labels**: backend, frontend, memory-security, privacy-control  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 16 hours  
**Dependencies**: 2.52 (Hypatia memory), 2.14 (database migration)

**Subtasks**:
- **2.95.1** Add `trust_level` metadata to all embedded content (private, public, internal) - 6 hours
- **2.95.2** Enforce query scope boundaries on vector store access - 6 hours
- **2.95.3** Build user-facing memory privacy dashboard - 4 hours

**Acceptance Criteria**:
- All embedded content includes trust level metadata
- Vector queries are properly scoped by trust level and user permissions
- Users have complete control over memory visibility and sharing
- Privacy dashboard provides transparent memory management

#### 2.96 Output Sanitization & Policy Leakage Defense 📋
**Epic**: Output Security  
**Story**: Prevent system prompt exfiltration and policy leakage  
**Labels**: backend, ai-safety, output-filtering, policy-protection  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 10 hours  
**Dependencies**: 2.94 (prompt wrapping), AI output processing

**Subtasks**:
- **2.96.1** Prevent system prompt exfiltration via LLM response audits - 4 hours
- **2.96.2** Strip "Act as..." or system-imitating content from AI output - 3 hours
- **2.96.3** Mark hallucinated policies with warning tags - 3 hours

**Acceptance Criteria**:
- System prompts cannot be extracted through AI responses
- Role-playing attempts in output are detected and filtered
- Hallucinated policies are clearly marked with warnings
- Output filtering maintains response quality while ensuring security

#### 2.97 Token Obfuscation Defense Layer 📋
**Epic**: Input Security  
**Story**: Defend against encoding tricks and obfuscated prompt injection  
**Labels**: backend, input-filtering, encoding-security, pattern-detection  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 8 hours  
**Dependencies**: 2.91 (upload pipeline), input processing

**Subtasks**:
- **2.97.1** Normalize homoglyphs, leetspeak, encoding edge cases - 3 hours
- **2.97.2** Zero-width and invisible character filters - 3 hours
- **2.97.3** TokenBreak pattern blocklist + logging - 2 hours

**Acceptance Criteria**:
- Homoglyph and leetspeak obfuscation is normalized
- Invisible characters and zero-width attacks are filtered
- TokenBreak patterns are detected and blocked
- All filtering attempts are logged for analysis

#### 2.98 Red Teaming & Threat Simulation Suite 📋
**Epic**: Security Testing  
**Story**: Automated adversarial testing and vulnerability discovery  
**Labels**: testing, security-validation, red-team, continuous-security  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 12 hours  
**Dependencies**: All security infrastructure tasks

**Subtasks**:
- **2.98.1** Maintain corpus of adversarial prompts (prompt injection, policy hijack, etc.) - 4 hours
- **2.98.2** Automate simulation tests across upload/chat pipelines - 4 hours
- **2.98.3** Log safety failures and regressions in CI/CD - 4 hours

**Acceptance Criteria**:
- Comprehensive adversarial prompt corpus is maintained and updated
- Automated tests validate security across all input vectors
- Security regressions are detected in CI/CD pipeline
- Test results inform security improvements and patches

#### 2.999 Final LLM Safety Hardening 📋
**Epic**: Security Validation  
**Story**: Comprehensive security review and cross-phase validation  
**Labels**: security-review, validation, hardening, compliance  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 8 hours  
**Dependencies**: All Phase 2.9 security tasks

**Subtasks**:
- **2.999.1** Cross-phase review of all injection surfaces and memory leaks - 3 hours
- **2.999.2** Validate Hypatia-specific safeguards across all roles - 3 hours
- **2.999.3** Harden all third-party plugin/tool integrations - 2 hours

**Acceptance Criteria**:
- All potential injection surfaces have been identified and secured
- Hypatia assistant maintains safe behavior across all user roles
- Third-party integrations include appropriate security controls
- Security validation is complete and documented

---

## 🎯 Phase 4.0 – User Experience & Growth 🚀 *FUTURE*

### 4.1 Visual Design & Growth Optimization 🎨

#### 4.11 Finalize Visual Identity & Homepage Styling 🎨  
**Epic**: UX & Branding  
**Story**: Apply final design system to homepage for a polished public experience  
**Labels**: design, branding, polish, frontend  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 10 hours  
**Dependencies**:  
- ✅ 2.21 (homepage scaffold complete with modular components)  
- 🎯 Finalized brand identity (logo, typography, color system)  
- 📣 Approved messaging and homepage copy (see `MESSAGING.md`)  

**Requirements**:
- Replace all placeholder logos, color accents, and visual elements  
- Apply finalized brand design system site-wide  
- Tighten layout, spacing, and typography for visual harmony  
- Ensure full responsiveness and visual integrity across screen sizes  
- Confirm that homepage aligns with growth, trust, and positioning goals

**Acceptance Criteria**:
- Homepage reflects official brand identity (logo, color, type)  
- Design feels cohesive, responsive, and launch-ready  
- All placeholder assets removed or replaced  
- Ready for external marketing, investor traffic, and public visibility

📝 Note: This task is **not required for app MVP**, but is **critical for launch readiness and external credibility**. Development will continue using placeholder elements until design is finalized in Phase 4.

---

## 📋 Phase 3.0: Marketplace & Community Platform 🔮 *FUTURE*

### 3.1 Marketplace Infrastructure Epic 📋

#### 3.11 Content Monetization Backend 📋
**Epic**: Marketplace Platform  
**Story**: Creator monetization with revenue sharing  
**Labels**: backend, marketplace, payments  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 30 hours

**Requirements**:
- Creator content listing and pricing APIs
- Revenue sharing calculation and distribution
- Payment processing integration (Stripe Connect)
- Creator earnings dashboard and analytics
- Content licensing and rights management

#### 3.12 Community Features Implementation 📋
**Epic**: Social Platform  
**Story**: Book clubs and discussion forums  
**Labels**: frontend, community, social  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 25 hours

**Requirements**:
- Discussion forums and book club management
- Social sharing with privacy controls
- Content moderation and community guidelines
- User reputation and achievement systems

### 3.3 Hypatia Advanced Features Epic 📋

#### 3.31 Voice Interaction Backend 📋
**Epic**: Advanced AI  
**Story**: Voice input and output capabilities  
**Labels**: backend, ai, voice  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 20 hours

**Requirements**:
- Voice-to-text processing backend with high accuracy
- Text-to-speech backend with Hypatia's personality
- Voice interaction APIs and real-time processing
- Audio streaming and buffering optimization

#### 3.32 Multilingual Backend Support 📋
**Epic**: Global Platform  
**Story**: Multiple language support  
**Labels**: backend, ai, i18n  
**Priority**: Medium  
**Status**: To Do  
**Estimated Effort**: 18 hours

**Requirements**:
- Multilingual text processing for Spanish, French, and German
- Automatic language detection APIs
- Personality adaptation backend for different cultural contexts
- Multilingual content recommendation engine

### 3.13 In-Memory and Sorted Data Infrastructure 🔄

#### 3.131 Evaluate suitable sorted data structures for dynamic user interactions 📋
**Epic**: Data Infrastructure  
**Story**: Analyze and select appropriate sorted data structures for real-time user interactions  
**Labels**: backend, performance, data-structures  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 8 hours

**Requirements**:
- Evaluate performance characteristics of different sorted data structures
- Analyze use cases for trending content, user rankings, and dynamic feeds
- Compare memory usage and insertion/deletion performance
- Document decision rationale for structure selection

#### 3.132 Implement `heapq` + dict pattern for real-time trending content cache 📋
**Epic**: Data Infrastructure  
**Story**: Implement efficient caching for trending content using Python native structures  
**Labels**: backend, caching, performance  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 12 hours

**Requirements**:
- Implement trending content cache using heapq for ranking and dict for fast lookups
- Support real-time updates to trending scores and content positioning
- Handle cache eviction and size management efficiently
- Integrate with existing content recommendation systems

#### 3.133 Integrate SortedContainers (or equivalent) for dynamic user note indexing 📋
**Epic**: Data Infrastructure  
**Story**: Implement efficient indexing for user notes and annotations  
**Labels**: backend, indexing, user-content  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 15 hours

**Requirements**:
- Implement SortedContainers library for maintaining sorted user note indexes
- Support multiple sort criteria (timestamp, relevance, custom user ordering)
- Optimize for frequent insertions and range queries
- Integrate with existing note management and search systems

#### 3.134 Rely on Postgres B-tree indexes for persistent sorted metadata 📋
**Epic**: Data Infrastructure  
**Story**: Optimize database queries using PostgreSQL B-tree indexes for sorted data  
**Labels**: backend, database, indexing  
**Priority**: High  
**Status**: To Do  
**Estimated Effort**: 10 hours

**Requirements**:
- Design and implement optimized B-tree indexes for content metadata
- Create composite indexes for multi-column sorting requirements
- Optimize query patterns for sorted data retrieval
- Monitor and tune index performance for production workloads

#### 3.135 Defer skip list implementation unless performance testing indicates need 📋
**Epic**: Data Infrastructure  
**Story**: Placeholder for skip list implementation based on performance requirements  
**Labels**: backend, performance, conditional  
**Priority**: Low  
**Status**: To Do  
**Estimated Effort**: 20 hours

**Requirements**:
- Monitor performance of existing sorted data structures
- Identify specific use cases where skip lists would provide benefits
- Implement skip list structure only if performance testing indicates significant gains
- Document performance benchmarks and decision criteria

#### 3.136 (Optional/Future) Investigate Redis Sorted Sets for session-based ranking memory 📋
**Epic**: Data Infrastructure  
**Story**: Evaluate Redis Sorted Sets for session-based user ranking and personalization  
**Labels**: backend, redis, session-management, optional  
**Priority**: Low  
**Status**: To Do  
**Estimated Effort**: 16 hours

**Requirements**:
- Investigate Redis Sorted Sets for session-based user activity ranking
- Evaluate integration with existing session management systems
- Assess performance benefits for real-time user personalization
- Design implementation strategy for session-based ranking memory

---

## 📝 Task Management Notes

### Sprint Planning Principles
- **Sprint Length**: 2-week sprints focusing on complete epic delivery
- **Definition of Done**: All acceptance criteria met, tests passing, documentation updated
- **Dependency Management**: Clear blocking/blocked relationships tracked
- **Risk Assessment**: High-risk tasks identified and mitigation planned

### Priority Classification
- **High**: Blocking other work or critical for user experience
- **Medium**: Important for feature completeness but not blocking
- **Low**: Nice-to-have improvements or optimizations

### Estimation Guidelines
- **Small** (1-4 hours): Single component changes, bug fixes
- **Medium** (5-12 hours): Feature implementation, integration work
- **Large** (15-30 hours): Epic-level deliverables, major system changes

### Phase 1.6 Sprint Completion
- **Phase 1.6**: 9 of 9 critical tasks completed ✅
- **Completion Rate**: 100% of Phase 1.6 sprint completed
- **Status**: Phase 2.0 active development

### Phase 2.0 Progress Tracking
- **Total Tasks**: 31 tasks across 8 epic areas
- **Estimated Effort**: 485 hours (12-16 weeks)
- **Completion Rate**: 0% (Phase 2 launching)
- **Critical Path**: 2.11 → 2.12 → 2.14 → 2.15 (foundation tasks)
- **Current Focus**: Next.js migration and multi-user authentication

---

## 📣 MARKETING & BRANDING DEPENDENCIES (NON-CODING)

The following marketing and branding materials are required before certain Phase 2 tasks can be completed:

### Brand Strategy & Identity 🎨
- **Brand manifesto and core messaging** (mission, vision, values)
- **Target audience personas** for Readers, Educators, and Organizations
- **Value proposition refinement** and competitive positioning
- **Hypatia personality guide** with voice, tone, and conversation patterns
- **Visual brand guide** with final color palette, typography, and design tokens

### Homepage & Marketing Content 📝
- **Homepage copy and messaging** aligned with brand strategy
- **Feature descriptions** with benefits for each user segment
- **User testimonials and case studies** for social proof
- **Pricing strategy and tier definitions** with feature breakdowns
- **Free plan messaging** and premium upgrade flow copy

### Visual Assets & Design 🖼️
- **Homepage hero illustrations** and feature showcase graphics
- **Alexandria Classic theme finalization** with brand-accurate colors and typography
- **Hypatia avatar and visual identity** for assistant branding
- **Marketing page layouts** and wireframes for landing pages
- **App walkthrough videos or interactive GIFs** for onboarding

### User Experience Copy ✍️
- **Onboarding flow messaging** tailored to each user role
- **Feature tutorial copy** and tooltip text for guided tours
- **Empty state messaging** with encouraging and helpful guidance
- **Error message copy** that maintains brand voice while being helpful
- **Email templates** for verification, welcome, and engagement sequences

### Blocked Development Tasks 🚫
These tasks cannot proceed without marketing dependencies:
- **2.21** Build Public-Facing Homepage → needs brand strategy, copy, visual assets
- **2.22** Marketing Landing Pages → needs pricing strategy, segment messaging
- **2.24** Alexandria Classic Theme Finalization → needs final brand colors/typography
- **2.52** Hypatia Personality → needs personality guide and voice documentation

### Recommended Delivery Timeline 📅
- **Week 1-2**: Brand strategy, messaging, and Hypatia personality guide
- **Week 3-4**: Visual brand guide and homepage copy
- **Week 5-6**: Pricing strategy and marketing page content
- **Week 7-8**: Visual assets, illustrations, and app walkthrough materials

*These materials should be developed in parallel with technical implementation to avoid blocking critical user-facing features.*

---

*This task file contains only active development work. Completed tasks are archived in Git history.*