**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of 2025-07-05.**

# 🎨 Frontend Architecture

## Overview

This document details the frontend architecture for the Alexandria platform, covering the evolution from Streamlit MVP to Next.js production application, and future Electron desktop application.

For additional architecture details, see:
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Backend Architecture](ARCHITECTURE_BACKEND.md)
- [Data Model & Storage](ARCHITECTURE_DATA_MODEL.md)
- [AI Services & RAG](ARCHITECTURE_AI_SERVICES.md)

## 🎯 Phase 1: Streamlit MVP Architecture

### Component Structure

```
streamlit_app/
├── main.py                 # Main application entry
├── pages/
│   ├── 📚_upload.py       # Book upload interface
│   ├── 💬_chat.py         # Q&A interface
│   ├── 📊_dashboard.py    # Progress dashboard
│   └── ⚙️_settings.py     # Settings page
├── components/
│   ├── book_uploader.py   # Drag & drop component
│   ├── chat_interface.py  # Chat UI components
│   ├── progress_bar.py    # Progress indicators
│   └── theme_selector.py  # Basic theme switching
└── utils/
    ├── session_state.py   # State management
    ├── file_handlers.py   # File processing
    └── api_client.py      # Backend API calls
```

### Success Criteria Met
- ✅ Drag & drop book upload with validation
- ✅ Q&A interface with source citations
- ✅ Basic chat history (in-session)
- ✅ Reading progress dashboard
- ✅ Settings for API keys and model selection

## 🚀 Phase 2: Next.js Production Architecture

### Technology Stack
- **Framework**: Next.js 14+ with TypeScript
- **Styling**: Tailwind CSS + Shadcn/ui components
- **State Management**: Zustand for global state + React Query for server state
- **Authentication**: NextAuth.js or Supabase Auth
- **PWA**: Next-PWA for offline capabilities

### Component Architecture

```
frontend/
├── app/                    # Next.js App Router
│   ├── (auth)/            # Authentication routes
│   ├── library/           # Main library interface
│   ├── books/[id]/        # Individual book pages
│   ├── themes/            # Theme management
│   └── settings/          # User preferences
├── components/
│   ├── ui/                # Shadcn/ui base components
│   ├── library/           # Library-specific components
│   ├── reading/           # Reading interface components
│   ├── chat/              # Enhanced chat components
│   ├── themes/            # Theme system components
│   └── purchasing/        # E-commerce components
├── hooks/                 # Custom React hooks
├── lib/                   # Utilities and configurations
├── stores/                # Zustand state stores
└── types/                 # TypeScript type definitions
```

### Key Features Implementation

#### Authentication System
```typescript
// Authentication architecture
interface AuthStore {
  user: User | null
  session: Session | null
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => Promise<void>
  register: (userData: RegisterData) => Promise<void>
}

// User profile with preferences
interface UserProfile {
  id: string
  preferences: {
    themes: ThemePreferences
    reading: ReadingPreferences
    notifications: NotificationSettings
  }
  library: PersonalLibrary
  progress: ReadingProgress
}
```

#### Main Library Architecture
```typescript
// Public domain catalog system
interface LibraryCatalog {
  books: PublicDomainBook[]
  categories: BookCategory[]
  featured: FeaturedContent[]
  search: SearchInterface
  filters: FilterSystem
}

// Purchasing system integration
interface PurchasingSystem {
  cart: ShoppingCart
  payment: StripePaymentProcessor
  drm: DigitalRightsManager
  history: PurchaseHistory
}
```

## 🎨 Theme System Architecture

### Reading Environment Themes

**Core Theme System:**
```typescript
interface ThemeSystem {
  themes: ReadingTheme[]
  customizer: ThemeCustomizer
  presets: ThemePreset[]
  persistence: ThemeStorage
  engine: ThemeRenderingEngine
}

// Core reading environment themes
enum ReadingEnvironment {
  SPACE = 'space',           // Dark with cosmic elements and star fields
  ZEN_GARDEN = 'zen',        // Minimalist with nature elements and zen aesthetics  
  FOREST = 'forest',         // Green nature imagery with wood textures
  LOG_CABIN = 'cabin',       // Warm wood tones and cozy fireplace elements
  CLASSIC_STUDY = 'study',   // Traditional library with leather and mahogany
  CYBERPUNK = 'cyberpunk',   // Neon colors and tech aesthetics (Phase 3)
  ART_NOUVEAU = 'nouveau',   // Elegant curves and artistic patterns (Phase 3)
  MINIMALIST = 'minimal',    // Ultra-clean and distraction-free (Phase 3)
  VINTAGE = 'vintage'        // Antique book and retro library aesthetics (Phase 3)
}

interface ThemeConfiguration {
  id: string
  name: string
  environment: ReadingEnvironment
  colorPalette: ColorPalette
  typography: TypographySettings
  layout: LayoutConfiguration
  assets: ThemeAssets
  animations: AnimationSettings
  accessibility: AccessibilitySettings
}
```

**Theme Customization Engine:**
```typescript
class ThemeCustomizer {
  // Color scheme customization
  async customizeColors(
    themeId: string,
    colorOverrides: ColorOverrides
  ): Promise<CustomTheme>
  
  // Typography customization
  async customizeTypography(
    themeId: string,
    typographySettings: TypographySettings
  ): Promise<CustomTheme>
  
  // Layout and spacing customization
  async customizeLayout(
    themeId: string,
    layoutSettings: LayoutSettings
  ): Promise<CustomTheme>
  
  // Create custom theme from scratch
  async createCustomTheme(
    baseTheme: ReadingEnvironment,
    customizations: ThemeCustomizations
  ): Promise<CustomTheme>
}

interface ThemeAssets {
  backgroundImages: BackgroundImage[]
  textureOverlays: TextureOverlay[]
  iconSet: IconSet
  soundEffects?: SoundEffect[] // Optional ambient sounds
  animations: AnimationAsset[]
}

interface ThemePreferences {
  primaryTheme: string
  darkModePreference: 'auto' | 'light' | 'dark'
  deviceSpecificThemes: Record<DeviceType, string>
  contextualThemes: ContextualThemeMapping
  customizations: ThemeCustomizations
}
```

**Theme Rendering Performance:**
```typescript
class ThemeRenderingEngine {
  // Optimized theme switching without page reload
  async switchTheme(
    fromTheme: ThemeConfiguration,
    toTheme: ThemeConfiguration,
    transitionStyle: ThemeTransition
  ): Promise<void>
  
  // Preload themes for instant switching
  async preloadThemes(themeIds: string[]): Promise<void>
  
  // CSS-in-JS theme generation
  async generateThemeCSS(theme: ThemeConfiguration): Promise<CSSStyleSheet>
  
  // Theme performance optimization
  async optimizeThemeAssets(theme: ThemeConfiguration): Promise<OptimizedTheme>
}

interface ThemeTransition {
  type: 'fade' | 'slide' | 'instant'
  duration: number // milliseconds
  easing: 'ease' | 'ease-in' | 'ease-out' | 'ease-in-out'
}

// Performance requirements
interface ThemePerformanceTargets {
  themeSwitch: 100, // ms - must complete within 100ms
  assetLoad: 2000, // ms - theme assets should load within 2s
  memoryFootprint: 50, // MB - maximum memory usage per theme
  cssGeneration: 50 // ms - CSS generation should complete within 50ms
}
```

## 🌟 Phase 3: Advanced Community Architecture

### Enhanced Component Structure

```
frontend/
├── app/
│   ├── community/         # Social features
│   ├── clubs/             # Book clubs
│   ├── authors/           # Author following
│   ├── themes/            # Extended theme system
│   └── analytics/         # Progress dashboards
├── components/
│   ├── social/            # Social sharing components
│   ├── annotations/       # Note-taking system
│   ├── analytics/         # Progress visualization
│   ├── clubs/             # Book club management
│   └── themes/            # Advanced theme builder
└── features/
    ├── multi-book-chat/   # Cross-book analysis
    ├── recommendations/   # AI-driven suggestions
    ├── community/         # Social interactions
    └── marketplace/       # Creator monetization
```

### Advanced Features

#### Social Architecture
```typescript
interface SocialSystem {
  sharing: ContentSharing
  following: AuthorFollowing
  discussions: CommunityDiscussions
  moderation: ContentModeration
}

interface AnnotationSystem {
  highlights: HighlightManager
  notes: NoteManager
  sharing: AnnotationSharing
  collaboration: CollaborativeAnnotations
}
```

#### Multi-Book Analysis
```typescript
interface MultiBookAnalysis {
  comparison: CrossBookComparison
  synthesis: InformationSynthesis
  discovery: IntelligentDiscovery
  recommendations: PersonalizedSuggestions
}
```

#### Advanced Theme Features
```typescript
interface AdvancedThemeFeatures {
  // Custom theme builder
  themeBuilder: CustomThemeBuilder
  
  // Community sharing
  communityThemes: CommunityThemeStore
  
  // Adaptive themes
  adaptiveThemes: AdaptiveThemeEngine
  
  // Theme marketplace
  themeMarketplace: ThemeMarketplace
}

class AdaptiveThemeEngine {
  // Time-based theme adaptation
  async adaptForTimeOfDay(
    baseTheme: ThemeConfiguration,
    currentTime: Date
  ): Promise<AdaptedTheme>
  
  // Reading context adaptation
  async adaptForReadingContext(
    baseTheme: ThemeConfiguration,
    context: ReadingContext
  ): Promise<AdaptedTheme>
  
  // Genre-specific adaptations
  async adaptForGenre(
    baseTheme: ThemeConfiguration,
    bookGenre: BookGenre
  ): Promise<AdaptedTheme>
}
```

## 🖥️ Phase 4: Electron Desktop Application Architecture

### Desktop Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Electron Desktop App                        │
├─────────────────┬───────────────────┬───────────────────────────┤
│   Main Process  │  Renderer Process │    Enhanced Features      │
│   (Node.js)     │   (Next.js Web)   │   (Desktop-Specific)      │
│                 │                   │                           │
│ • Window Mgmt   │ • Same Next.js    │ • Offline Storage         │
│ • File System   │   Frontend        │ • Multi-Window Support    │
│ • OS Integration│ • Same Components │ • Plugin Architecture     │
│ • IPC Bridge    │ • Enhanced for    │ • Native Integrations     │
│ • Auto-Updates  │   Desktop UX      │ • Performance Opts        │
└─────────────────┴───────────────────┴───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    IPC Bridge       │
                    │ (Secure Communication)
                    │ • API Calls         │
                    │ • File Operations   │
                    │ • System Events     │
                    │ • Window Control    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Local Storage     │
                    │ • IndexedDB (Books) │
                    │ • SQLite (Progress) │
                    │ • File System Cache │
                    │ • Plugin Storage    │
                    └─────────────────────┘
```

### Core Technical Architecture

#### Electron Main Process (Node.js)
```typescript
// Main process architecture
interface ElectronMainProcess {
  windowManager: WindowManager
  fileSystemHandler: FileSystemHandler
  osIntegration: OperatingSystemIntegration
  autoUpdater: AutoUpdateManager
  ipcHandler: IPCCommunicationHandler
}

// Window management for multi-window support
interface WindowManager {
  createMainWindow(): BrowserWindow
  createReadingWindow(bookId: string): BrowserWindow
  manageWindowSessions(): void
  handleMultiMonitor(): void
}

// File system operations
interface FileSystemHandler {
  importBooks(filePaths: string[]): Promise<BookImportResult[]>
  exportNotes(format: ExportFormat): Promise<string>
  manageLocalStorage(): Promise<StorageStats>
  handleDragDrop(files: FileList): Promise<void>
}
```

#### Renderer Process (Next.js + Desktop Enhancements)
```typescript
// Enhanced Next.js app for desktop
interface DesktopRenderer {
  webApp: NextJSApplication
  desktopEnhancements: DesktopFeatures
  offlineCapabilities: OfflineManager
  performanceOptimizations: PerformanceManager
}

// Desktop-specific features
interface DesktopFeatures {
  multiWindow: MultiWindowCoordinator
  nativeMenus: NativeMenuHandler
  keyboardShortcuts: ShortcutManager
  systemTray: TrayIntegration
  notifications: NativeNotifications
}
```

### Desktop-Specific Features

#### Multi-Window Support
```typescript
interface MultiWindowSystem {
  mainWindow: MainApplicationWindow
  readingWindows: ReadingWindow[]
  comparisonWindows: BookComparisonWindow[]
  settingsWindow: SettingsWindow
}

// Cross-window communication
interface WindowCommunication {
  sharedState: CrossWindowStateManager
  windowBridge: WindowBridgeAPI
  sessionPersistence: WindowSessionManager
}
```

#### Plugin Architecture
```typescript
// Plugin system design
interface PluginArchitecture {
  pluginManager: PluginManager
  pluginAPI: PluginAPI
  securitySandbox: PluginSandbox
  pluginDiscovery: PluginDiscoveryService
}

// Plugin API for developers
interface PluginAPI {
  reading: ReadingPluginAPI
  notes: NotesPluginAPI
  themes: ThemePluginAPI
  export: ExportPluginAPI
  ui: UIExtensionAPI
}
```

### Security Architecture

#### Desktop Security Model
```typescript
interface DesktopSecurity {
  processIsolation: ProcessIsolationManager
  ipcSecurity: SecureIPCCommunication
  fileSystemSecurity: FileSystemSecurityManager
  pluginSecurity: PluginSecuritySandbox
}

// Secure IPC implementation
interface SecureIPC {
  contextIsolation: boolean
  nodeIntegration: false
  webSecurity: true
  allowRunningInsecureContent: false
  enableRemoteModule: false
}
```

### Cross-Platform Compatibility

#### Platform-Specific Adaptations
```typescript
// Platform detection and adaptation
interface PlatformAdapter {
  windows: WindowsPlatformAdapter
  macOS: macOSPlatformAdapter
  linux: LinuxPlatformAdapter
}

// Windows-specific features
interface WindowsPlatformAdapter {
  taskbarIntegration: WindowsTaskbar
  jumpList: WindowsJumpList
  notifications: WindowsNotifications
  fileAssociations: WindowsFileAssociations
}

// macOS-specific features
interface macOSPlatformAdapter {
  dockIntegration: macOSDock
  menuBarIntegration: macOSMenuBar
  touchBarSupport: macOSTouchBar
  darkModeIntegration: macOSDarkMode
}
```

### Performance Benchmarks

#### Desktop Performance Targets
```yaml
Performance Requirements:
  Startup Time: <3 seconds from click to usable
  Memory Usage: <500MB baseline, <1GB with large libraries
  CPU Usage: <5% idle, <25% during intensive operations
  Storage Efficiency: 70% compression for offline content
  
Multi-Window Performance:
  Window Creation: <500ms per additional window
  Cross-Window Sync: <100ms state synchronization
  Memory Per Window: <200MB additional per reading window
  
Offline Capabilities:
  Book Import Speed: >10MB/s for local files
  Search Performance: <200ms for offline search
  Sync Speed: >5MB/s for background synchronization
```

## 🔄 Frontend Migration Strategy

### Streamlit to Next.js Migration
1. **Component Mapping**: Document Streamlit components → Next.js equivalents
2. **State Management**: Migrate session state to Zustand stores
3. **API Integration**: Maintain same FastAPI endpoints
4. **User Experience**: Enhance UX with professional UI components

### Next.js to Electron Migration
1. **Frontend Reuse**: Same Next.js app runs in Electron renderer
2. **Enhanced Features**: Add desktop-specific capabilities
3. **Performance**: Optimize for desktop environment
4. **Distribution**: Package as native desktop applications

## 📊 Frontend Performance Targets

### Web Application Performance
- **Initial Load**: <3 seconds to interactive
- **Route Changes**: <500ms between pages
- **Theme Switching**: <100ms transition
- **Search Results**: <1 second response time

### Desktop Application Performance
- **Application Startup**: <3 seconds
- **Window Management**: <500ms per window
- **File Operations**: >10MB/s for imports
- **Offline Search**: <200ms response time

## 🔗 Related Documentation

- [Backend Architecture →](ARCHITECTURE_BACKEND.md)
- [Data Model & Storage →](ARCHITECTURE_DATA_MODEL.md)
- [AI Services & RAG →](ARCHITECTURE_AI_SERVICES.md)
- [Architecture Overview →](ARCHITECTURE_OVERVIEW.md)