"""
Adapter to integrate multi-provider AI system with RAG service.

This module provides an adapter that bridges the new AI provider system
with the existing RAG service LLM provider interface.
"""

from typing import Dict, Optional, Tuple, Any
from src.rag.rag_service import LLMProviderInterface
from src.utils.ai_providers import get_ai_provider_manager, AIProviderType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiProviderLLMAdapter(LLMProviderInterface):
    """
    Adapter that implements RAG service LLM interface using the new AI provider system.
    
    This allows the RAG service to use multiple AI providers (OpenAI, Anthropic, etc.)
    through a unified interface.
    """
    
    def __init__(self, preferred_provider: Optional[AIProviderType] = None):
        """
        Initialize the multi-provider adapter.
        
        Args:
            preferred_provider: Preferred AI provider type
        """
        self.preferred_provider = preferred_provider
        self._manager = None
        logger.info(f"Multi-provider LLM adapter initialized with preferred provider: {preferred_provider}")
    
    async def _get_manager(self):
        """Get or initialize the AI provider manager."""
        if self._manager is None:
            self._manager = await get_ai_provider_manager()
        return self._manager
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate response using AI provider system.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Specific model to use
        
        Returns:
            Tuple of (response_text, token_usage)
        """
        try:
            manager = await self._get_manager()
            
            # Format prompt as messages for chat completion
            messages = [{"role": "user", "content": prompt}]
            
            # Generate response using preferred provider or primary
            response = await manager.generate_text(
                messages=messages,
                provider_type=self.preferred_provider,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert usage metrics to expected format
            token_usage = {
                "prompt_tokens": response.usage_metrics.prompt_tokens,
                "completion_tokens": response.usage_metrics.completion_tokens,
                "total_tokens": response.usage_metrics.total_tokens
            }
            
            logger.info(f"Generated response using {response.provider_type.value} provider: "
                       f"{token_usage['total_tokens']} tokens")
            
            return response.content, token_usage
            
        except Exception as e:
            logger.error(f"Multi-provider response generation failed: {e}")
            raise
    
    def get_context_limit(self, model: Optional[str] = None) -> int:
        """
        Get maximum context length for the model.
        
        Args:
            model: Model name
        
        Returns:
            int: Maximum context length
        """
        # Default context limits by provider and model
        context_limits = {
            # OpenAI models
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-32k": 32768,
            
            # Anthropic models
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20241022": 200000,
        }
        
        if model and model in context_limits:
            return context_limits[model]
        
        # Default based on preferred provider
        if self.preferred_provider == AIProviderType.ANTHROPIC:
            return 200000  # Claude default
        else:
            return 16385   # GPT-3.5-turbo default
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the adapter and providers.
        
        Returns:
            Dict with health status information
        """
        try:
            manager = await self._get_manager()
            
            # Get health status from all providers
            provider_health = await manager.health_check_all()
            
            # Check if preferred provider is healthy
            preferred_healthy = True
            if self.preferred_provider:
                provider_status = provider_health.get(self.preferred_provider.value, {})
                preferred_healthy = provider_status.get("status") == "healthy"
            
            return {
                "adapter": "multi_provider_llm",
                "status": "healthy" if preferred_healthy else "degraded",
                "preferred_provider": self.preferred_provider.value if self.preferred_provider else None,
                "preferred_provider_healthy": preferred_healthy,
                "available_providers": list(provider_health.keys()),
                "provider_details": provider_health
            }
            
        except Exception as e:
            return {
                "adapter": "multi_provider_llm",
                "status": "unhealthy",
                "error": str(e)
            }


# Convenience function to create adapter with OpenAI as default
async def get_openai_llm_adapter() -> MultiProviderLLMAdapter:
    """Get LLM adapter configured for OpenAI."""
    return MultiProviderLLMAdapter(preferred_provider=AIProviderType.OPENAI)


# Convenience function to create adapter with Anthropic as default
async def get_anthropic_llm_adapter() -> MultiProviderLLMAdapter:
    """Get LLM adapter configured for Anthropic."""
    return MultiProviderLLMAdapter(preferred_provider=AIProviderType.ANTHROPIC)


# Convenience function to create adapter with automatic provider selection
async def get_auto_llm_adapter() -> MultiProviderLLMAdapter:
    """Get LLM adapter with automatic provider selection."""
    return MultiProviderLLMAdapter(preferred_provider=None)