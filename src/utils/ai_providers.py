"""
Multi-provider AI interface for Alexandria platform.

This module provides abstract interfaces and implementations for different
AI providers (OpenAI, Anthropic, local models) to enable easy switching
and provider-agnostic AI operations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AIProviderType(str, Enum):
    """Supported AI provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class AIUsageMetrics:
    """Metrics for tracking AI provider usage."""
    provider_type: AIProviderType
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    start_time: datetime = None
    end_time: Optional[datetime] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.errors is None:
            self.errors = []
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage to metrics."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.request_count += 1
    
    def add_error(self, error: str):
        """Add error to metrics."""
        self.errors.append(error)
    
    def finish(self):
        """Mark metrics as complete."""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_cost_estimate(self) -> float:
        """Estimate cost based on provider and model."""
        # Basic cost estimation - would need provider-specific pricing
        if self.provider_type == AIProviderType.OPENAI:
            if "gpt-4" in self.model_name.lower():
                return (self.prompt_tokens * 0.00003 + self.completion_tokens * 0.00006)
            elif "gpt-3.5" in self.model_name.lower():
                return (self.prompt_tokens * 0.0000015 + self.completion_tokens * 0.000002)
        elif self.provider_type == AIProviderType.ANTHROPIC:
            if "claude-3" in self.model_name.lower():
                return (self.prompt_tokens * 0.000015 + self.completion_tokens * 0.000075)
        
        return 0.0  # Unknown pricing


@dataclass
class AIResponse:
    """Response from AI provider."""
    content: str
    model: str
    provider_type: AIProviderType
    usage_metrics: AIUsageMetrics
    finish_reason: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AIProviderInterface(ABC):
    """
    Abstract interface for AI providers.
    
    This interface defines the common methods that all AI providers
    must implement to be compatible with the Alexandria platform.
    """
    
    @abstractmethod
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AIResponse:
        """
        Generate text completion.
        
        Args:
            messages: List of messages in chat format
            model: Model name (provider-specific)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for context
        
        Returns:
            AIResponse: Generated response with metadata
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], AIUsageMetrics]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
        
        Returns:
            Tuple of embeddings and usage metrics
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> AIProviderType:
        """Get the provider type."""
        pass
    
    @abstractmethod
    def get_available_models(self, model_type: str = "text") -> List[str]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on provider."""
        pass


class OpenAIProvider(AIProviderInterface):
    """
    OpenAI AI provider implementation.
    
    Provides integration with OpenAI's GPT and embedding models.
    """
    
    def __init__(self):
        """Initialize OpenAI provider."""
        self.settings = get_settings()
        self.client = None
        self.provider_type = AIProviderType.OPENAI
        
        # Available models
        self.text_models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k", 
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-32k"
        ]
        
        self.embedding_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            # Import OpenAI library
            from openai import AsyncOpenAI
            
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            logger.info("OpenAI provider initialized")
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AIResponse:
        """Generate text using OpenAI."""
        model = model or self.settings.llm_model
        metrics = AIUsageMetrics(
            provider_type=self.provider_type,
            model_name=model
        )
        
        try:
            # Prepare messages
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.extend(messages)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            # Update metrics
            if response.usage:
                metrics.add_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
            
            metrics.finish()
            
            return AIResponse(
                content=content,
                model=model,
                provider_type=self.provider_type,
                usage_metrics=metrics,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            metrics.add_error(str(e))
            metrics.finish()
            logger.error(f"OpenAI text generation failed: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], AIUsageMetrics]:
        """Generate embeddings using OpenAI."""
        model = model or self.settings.embedding_model
        metrics = AIUsageMetrics(
            provider_type=self.provider_type,
            model_name=model
        )
        
        try:
            # Make API call
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Update metrics
            if response.usage:
                metrics.add_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=0  # Embeddings don't have completion tokens
                )
            
            metrics.finish()
            
            return embeddings, metrics
            
        except Exception as e:
            metrics.add_error(str(e))
            metrics.finish()
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def get_provider_type(self) -> AIProviderType:
        """Get provider type."""
        return self.provider_type
    
    def get_available_models(self, model_type: str = "text") -> List[str]:
        """Get available models."""
        if model_type == "text":
            return self.text_models
        elif model_type == "embedding":
            return self.embedding_models
        else:
            return self.text_models + self.embedding_models
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Simple API call to check connectivity
            response = await self.client.models.list()
            
            return {
                "provider": "openai",
                "status": "healthy",
                "api_accessible": True,
                "models_available": len(response.data) if response.data else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "provider": "openai",
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class AnthropicProvider(AIProviderInterface):
    """
    Anthropic AI provider implementation.
    
    Provides integration with Anthropic's Claude models.
    """
    
    def __init__(self):
        """Initialize Anthropic provider."""
        self.settings = get_settings()
        self.client = None
        self.provider_type = AIProviderType.ANTHROPIC
        
        # Available models
        self.text_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022"
        ]
        
        # Anthropic doesn't have embedding models, would need to use OpenAI or others
        self.embedding_models = []
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            # Import Anthropic library
            import anthropic
            
            api_key = getattr(self.settings, 'anthropic_api_key', None)
            if not api_key:
                logger.warning("Anthropic API key not configured - provider will be unavailable")
                return
            
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info("Anthropic provider initialized")
            
        except ImportError:
            logger.warning("Anthropic library not installed. Run: pip install anthropic")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AIResponse:
        """Generate text using Anthropic Claude."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        model = model or "claude-3-sonnet-20240229"
        max_tokens = max_tokens or 1000
        
        metrics = AIUsageMetrics(
            provider_type=self.provider_type,
            model_name=model
        )
        
        try:
            # Prepare messages for Claude format
            claude_messages = []
            for msg in messages:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make API call
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=claude_messages
            )
            
            # Extract response
            content = response.content[0].text if response.content else ""
            
            # Update metrics (Anthropic response format)
            if hasattr(response, 'usage'):
                metrics.add_usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens
                )
            
            metrics.finish()
            
            return AIResponse(
                content=content,
                model=model,
                provider_type=self.provider_type,
                usage_metrics=metrics,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else None
            )
            
        except Exception as e:
            metrics.add_error(str(e))
            metrics.finish()
            logger.error(f"Anthropic text generation failed: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Tuple[List[List[float]], AIUsageMetrics]:
        """Generate embeddings - not supported by Anthropic."""
        raise NotImplementedError(
            "Anthropic does not provide embedding models. "
            "Use OpenAI or another provider for embeddings."
        )
    
    def get_provider_type(self) -> AIProviderType:
        """Get provider type."""
        return self.provider_type
    
    def get_available_models(self, model_type: str = "text") -> List[str]:
        """Get available models."""
        if model_type == "text":
            return self.text_models
        elif model_type == "embedding":
            return self.embedding_models
        else:
            return self.text_models
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        if not self.client:
            return {
                "provider": "anthropic",
                "status": "unavailable",
                "error": "Client not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Simple API call to check connectivity
            # Note: Anthropic doesn't have a models.list endpoint like OpenAI
            # We'll make a minimal message call instead
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",  # Use smallest model for health check
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            return {
                "provider": "anthropic", 
                "status": "healthy",
                "api_accessible": True,
                "test_successful": len(response.content) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "provider": "anthropic",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class AIProviderManager:
    """
    Manager for multiple AI providers.
    
    Handles provider selection, fallback, and unified access to AI capabilities.
    """
    
    def __init__(self):
        """Initialize provider manager."""
        self.settings = get_settings()
        self.providers: Dict[AIProviderType, AIProviderInterface] = {}
        self.primary_provider = AIProviderType.OPENAI
        self.embedding_provider = AIProviderType.OPENAI  # Only OpenAI supports embeddings
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers."""
        # Initialize OpenAI (always available if configured)
        try:
            self.providers[AIProviderType.OPENAI] = OpenAIProvider()
            logger.info("OpenAI provider registered")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Anthropic (optional)
        try:
            anthropic_provider = AnthropicProvider()
            if anthropic_provider.client:  # Only register if properly initialized
                self.providers[AIProviderType.ANTHROPIC] = anthropic_provider
                logger.info("Anthropic provider registered")
            else:
                logger.info("Anthropic provider not configured - skipping")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
    
    def get_provider(self, provider_type: AIProviderType) -> Optional[AIProviderInterface]:
        """Get specific provider."""
        return self.providers.get(provider_type)
    
    def get_primary_provider(self) -> AIProviderInterface:
        """Get primary text generation provider."""
        provider = self.providers.get(self.primary_provider)
        if not provider:
            # Fallback to any available provider
            if self.providers:
                provider = next(iter(self.providers.values()))
            else:
                raise RuntimeError("No AI providers available")
        return provider
    
    def get_embedding_provider(self) -> AIProviderInterface:
        """Get embedding generation provider."""
        provider = self.providers.get(self.embedding_provider)
        if not provider:
            raise RuntimeError("No embedding provider available")
        return provider
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        provider_type: Optional[AIProviderType] = None,
        **kwargs
    ) -> AIResponse:
        """Generate text using specified or primary provider."""
        if provider_type:
            provider = self.get_provider(provider_type)
            if not provider:
                raise ValueError(f"Provider {provider_type} not available")
        else:
            provider = self.get_primary_provider()
        
        return await provider.generate_text(messages, **kwargs)
    
    async def generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> Tuple[List[List[float]], AIUsageMetrics]:
        """Generate embeddings using embedding provider."""
        provider = self.get_embedding_provider()
        return await provider.generate_embeddings(texts, **kwargs)
    
    def set_primary_provider(self, provider_type: AIProviderType):
        """Set primary text generation provider."""
        if provider_type in self.providers:
            self.primary_provider = provider_type
            logger.info(f"Primary provider set to: {provider_type}")
        else:
            raise ValueError(f"Provider {provider_type} not available")
    
    def get_available_providers(self) -> List[AIProviderType]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all providers."""
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                results[provider_type.value] = await provider.health_check()
            except Exception as e:
                results[provider_type.value] = {
                    "provider": provider_type.value,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return results


# Global provider manager instance
_provider_manager: Optional[AIProviderManager] = None


async def get_ai_provider_manager() -> AIProviderManager:
    """
    Get or create the AI provider manager instance.
    
    Returns:
        AIProviderManager: The provider manager instance
    """
    global _provider_manager
    
    if _provider_manager is None:
        _provider_manager = AIProviderManager()
        logger.info("AI Provider Manager singleton created")
    
    return _provider_manager


# Convenience functions
async def generate_text(
    messages: List[Dict[str, str]],
    provider_type: Optional[AIProviderType] = None,
    **kwargs
) -> AIResponse:
    """Generate text using AI provider."""
    manager = await get_ai_provider_manager()
    return await manager.generate_text(messages, provider_type, **kwargs)


async def generate_embeddings(
    texts: List[str],
    **kwargs
) -> Tuple[List[List[float]], AIUsageMetrics]:
    """Generate embeddings using AI provider."""
    manager = await get_ai_provider_manager()
    return await manager.generate_embeddings(texts, **kwargs)