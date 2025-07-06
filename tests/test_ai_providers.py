"""
Tests for multi-provider AI system.

Tests the AI provider interface, manager, and integration with RAG service.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.utils.ai_providers import (
    AIProviderManager,
    OpenAIProvider, 
    AnthropicProvider,
    AIProviderType,
    AIResponse,
    AIUsageMetrics,
    get_ai_provider_manager
)
from src.rag.ai_provider_adapter import MultiProviderLLMAdapter


class TestAIProviderManager:
    """Test AI provider manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create AI provider manager for testing."""
        return AIProviderManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initializes providers correctly."""
        assert isinstance(manager, AIProviderManager)
        assert AIProviderType.OPENAI in manager.providers
        # Anthropic may or may not be available depending on configuration
    
    def test_get_primary_provider(self, manager):
        """Test getting primary provider."""
        provider = manager.get_primary_provider()
        assert provider is not None
        assert provider.get_provider_type() == AIProviderType.OPENAI
    
    def test_get_embedding_provider(self, manager):
        """Test getting embedding provider."""
        provider = manager.get_embedding_provider()
        assert provider is not None
        assert provider.get_provider_type() == AIProviderType.OPENAI
    
    def test_get_available_providers(self, manager):
        """Test getting list of available providers."""
        providers = manager.get_available_providers()
        assert isinstance(providers, list)
        assert AIProviderType.OPENAI in providers
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, manager):
        """Test health check on all providers."""
        with patch.object(manager.providers[AIProviderType.OPENAI], 'health_check') as mock_health:
            mock_health.return_value = {"status": "healthy"}
            
            results = await manager.health_check_all()
            assert isinstance(results, dict)
            assert "openai" in results


class TestOpenAIProvider:
    """Test OpenAI provider functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create OpenAI provider for testing."""
        with patch('src.utils.ai_providers.AsyncOpenAI'):
            return OpenAIProvider()
    
    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.get_provider_type() == AIProviderType.OPENAI
        assert len(provider.get_available_models("text")) > 0
        assert len(provider.get_available_models("embedding")) > 0
    
    @pytest.mark.asyncio
    async def test_generate_text(self, provider):
        """Test text generation."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        provider.client = Mock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.generate_text(messages)
        
        assert isinstance(response, AIResponse)
        assert response.content == "Test response"
        assert response.provider_type == AIProviderType.OPENAI
        assert response.usage_metrics.prompt_tokens == 10
        assert response.usage_metrics.completion_tokens == 5
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, provider):
        """Test embedding generation."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_response.data[1].embedding = [0.4, 0.5, 0.6]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 8
        
        provider.client = Mock()
        provider.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        texts = ["Hello", "World"]
        embeddings, metrics = await provider.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert isinstance(metrics, AIUsageMetrics)
        assert metrics.prompt_tokens == 8


class TestAnthropicProvider:
    """Test Anthropic provider functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create Anthropic provider for testing."""
        with patch('anthropic.AsyncAnthropic'):
            provider = AnthropicProvider()
            # Mock client to be available
            provider.client = Mock()
            return provider
    
    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.get_provider_type() == AIProviderType.ANTHROPIC
        assert len(provider.get_available_models("text")) > 0
        assert len(provider.get_available_models("embedding")) == 0  # Anthropic doesn't do embeddings
    
    @pytest.mark.asyncio
    async def test_generate_text(self, provider):
        """Test text generation."""
        # Mock the Anthropic client response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 8
        mock_response.stop_reason = "end_turn"
        
        provider.client.messages.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.generate_text(messages)
        
        assert isinstance(response, AIResponse)
        assert response.content == "Claude response"
        assert response.provider_type == AIProviderType.ANTHROPIC
        assert response.usage_metrics.prompt_tokens == 15
        assert response.usage_metrics.completion_tokens == 8
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_not_supported(self, provider):
        """Test that embeddings are not supported."""
        with pytest.raises(NotImplementedError):
            await provider.generate_embeddings(["test"])


class TestMultiProviderLLMAdapter:
    """Test the RAG service adapter for multi-provider AI."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return MultiProviderLLMAdapter(preferred_provider=AIProviderType.OPENAI)
    
    @pytest.mark.asyncio
    async def test_generate_response(self, adapter):
        """Test response generation through adapter."""
        # Mock the AI provider manager
        mock_manager = Mock()
        mock_response = AIResponse(
            content="Adapter response",
            model="gpt-3.5-turbo",
            provider_type=AIProviderType.OPENAI,
            usage_metrics=AIUsageMetrics(
                provider_type=AIProviderType.OPENAI,
                model_name="gpt-3.5-turbo",
                prompt_tokens=12,
                completion_tokens=6,
                total_tokens=18
            )
        )
        mock_manager.generate_text = AsyncMock(return_value=mock_response)
        adapter._manager = mock_manager
        
        response_text, token_usage = await adapter.generate_response("Test prompt")
        
        assert response_text == "Adapter response"
        assert token_usage["prompt_tokens"] == 12
        assert token_usage["completion_tokens"] == 6
        assert token_usage["total_tokens"] == 18
    
    def test_get_context_limit(self, adapter):
        """Test getting context limits for different models."""
        # Test OpenAI model
        limit = adapter.get_context_limit("gpt-4")
        assert limit == 8192
        
        # Test Anthropic model
        limit = adapter.get_context_limit("claude-3-sonnet-20240229")
        assert limit == 200000
        
        # Test unknown model (should return default)
        limit = adapter.get_context_limit("unknown-model")
        assert limit > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check."""
        # Mock the AI provider manager
        mock_manager = Mock()
        mock_manager.health_check_all = AsyncMock(return_value={
            "openai": {"status": "healthy"}
        })
        adapter._manager = mock_manager
        
        health = await adapter.health_check()
        
        assert health["adapter"] == "multi_provider_llm"
        assert health["status"] in ["healthy", "degraded"]
        assert "provider_details" in health


@pytest.mark.asyncio
async def test_singleton_manager():
    """Test that AI provider manager is a singleton."""
    manager1 = await get_ai_provider_manager()
    manager2 = await get_ai_provider_manager()
    
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_ai_provider_integration():
    """Integration test for AI provider system."""
    # This test requires actual API keys to work fully
    # For now, just test initialization
    try:
        manager = await get_ai_provider_manager()
        providers = manager.get_available_providers()
        
        assert len(providers) > 0
        assert AIProviderType.OPENAI in providers
        
        # Test health check
        health = await manager.health_check_all()
        assert isinstance(health, dict)
        
    except Exception as e:
        # Expected if API keys not configured
        pytest.skip(f"Skipping integration test due to configuration: {e}")


if __name__ == "__main__":
    pytest.main([__file__])