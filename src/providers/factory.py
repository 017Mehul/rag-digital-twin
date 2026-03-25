"""
Centralized provider factory, validation, and fallback orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Type

from src.exceptions import (
    ConfigurationError,
    EmbeddingGenerationError,
    ErrorCode,
    ResponseGenerationError,
)

from .embedding_provider import (
    EmbeddingModel,
    HuggingFaceEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from .llm_provider import (
    HuggingFaceLLMProvider,
    LLMProvider,
    OpenAILLMProvider,
)


@dataclass(frozen=True)
class ProviderRegistration:
    """
    Metadata describing a registered provider implementation.
    """

    provider_class: Type[Any]
    default_model_name: str
    supported_kwargs: frozenset[str]


class FallbackEmbeddingProvider(EmbeddingModel):
    """
    Embedding provider that automatically fails over across a provider chain.
    """

    provider_name = "fallback"

    def __init__(self, providers: List[EmbeddingModel]) -> None:
        if not providers:
            raise ValueError("providers cannot be empty")

        primary = providers[0]
        for provider in providers[1:]:
            if provider.dimension != primary.dimension:
                raise ConfigurationError(
                    "Embedding fallback providers must use the same dimension",
                    ErrorCode.CONFIG_VALIDATION_FAILED,
                    "embedding_fallbacks",
                )

        super().__init__(model_name=primary.model_name, dimension=primary.dimension)
        self.providers = providers
        self._active_index = 0

    @property
    def active_provider(self) -> EmbeddingModel:
        return self.providers[self._active_index]

    def load_model(self) -> Any:
        return self._execute_with_fallback(
            operation=lambda provider: provider.load_model(),
            action_name="load embedding model",
        )

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        return self._execute_with_fallback(
            operation=lambda provider: provider.embed_text(text),
            action_name="generate embedding",
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        for text in texts:
            self._validate_text(text)

        return self._execute_with_fallback(
            operation=lambda provider: provider.embed_batch(texts),
            action_name="generate batch embeddings",
        )

    def get_config(self) -> Dict[str, Any]:
        active_config = self.active_provider.get_config()
        active_config["fallback_enabled"] = True
        active_config["fallback_chain"] = [
            {
                "provider": provider.provider_name,
                "model_name": provider.model_name,
            }
            for provider in self.providers
        ]
        active_config["active_provider_index"] = self._active_index
        return active_config

    def _execute_with_fallback(self, operation: Any, action_name: str) -> Any:
        errors: List[str] = []
        last_error: Optional[Exception] = None

        for index in range(self._active_index, len(self.providers)):
            provider = self.providers[index]
            try:
                result = operation(provider)
                self._active_index = index
                self.model_name = provider.model_name
                self.dimension = provider.dimension
                self._loaded_model = getattr(provider, "_loaded_model", None)
                self._loaded_backend = f"fallback:{provider.provider_name}"
                return result
            except Exception as exc:  # pragma: no cover - exercised through tests
                last_error = exc
                errors.append(f"{provider.provider_name}/{provider.model_name}: {exc}")

        error_code = (
            last_error.error_code
            if isinstance(last_error, EmbeddingGenerationError)
            else ErrorCode.EMBEDDING_GENERATION_FAILED
        )
        raise EmbeddingGenerationError(
            f"All embedding providers failed to {action_name}: {'; '.join(errors)}",
            error_code,
            self.active_provider.model_name,
            cause=last_error,
        ) from last_error


class FallbackLLMProvider(LLMProvider):
    """
    LLM provider that automatically fails over across a provider chain.
    """

    provider_name = "fallback"

    def __init__(self, providers: List[LLMProvider]) -> None:
        if not providers:
            raise ValueError("providers cannot be empty")

        primary = providers[0]
        super().__init__(model_name=primary.model_name)
        self.providers = providers
        self._active_index = 0

    @property
    def active_provider(self) -> LLMProvider:
        return self.providers[self._active_index]

    def load_model(self) -> Any:
        return self._execute_with_fallback(
            operation=lambda provider: provider.load_model(),
            action_name="load language model",
        )

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        self._validate_text(prompt)
        return self._execute_with_fallback(
            operation=lambda provider: provider.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            action_name="generate response",
        )

    def get_config(self) -> Dict[str, Any]:
        active_config = self.active_provider.get_config()
        active_config["fallback_enabled"] = True
        active_config["fallback_chain"] = [
            {
                "provider": provider.provider_name,
                "model_name": provider.model_name,
            }
            for provider in self.providers
        ]
        active_config["active_provider_index"] = self._active_index
        return active_config

    def _execute_with_fallback(self, operation: Any, action_name: str) -> Any:
        errors: List[str] = []
        last_error: Optional[Exception] = None

        for index in range(self._active_index, len(self.providers)):
            provider = self.providers[index]
            try:
                result = operation(provider)
                self._active_index = index
                self.model_name = provider.model_name
                self._loaded_model = getattr(provider, "_loaded_model", None)
                self._loaded_backend = f"fallback:{provider.provider_name}"
                return result
            except Exception as exc:  # pragma: no cover - exercised through tests
                last_error = exc
                errors.append(f"{provider.provider_name}/{provider.model_name}: {exc}")

        error_code = (
            last_error.error_code
            if isinstance(last_error, ResponseGenerationError)
            else ErrorCode.LLM_GENERATION_FAILED
        )
        raise ResponseGenerationError(
            f"All LLM providers failed to {action_name}: {'; '.join(errors)}",
            error_code,
            self.active_provider.model_name,
            cause=last_error,
        ) from last_error


class ProviderFactory:
    """
    Registry-backed provider creation and validation entry point.
    """

    _embedding_registry: Dict[str, ProviderRegistration] = {
        "openai": ProviderRegistration(
            provider_class=OpenAIEmbeddingProvider,
            default_model_name="text-embedding-3-small",
            supported_kwargs=frozenset({"api_key", "client", "dimension", "mock_embeddings"}),
        ),
        "huggingface": ProviderRegistration(
            provider_class=HuggingFaceEmbeddingProvider,
            default_model_name="sentence-transformers/all-MiniLM-L6-v2",
            supported_kwargs=frozenset({"model", "dimension", "mock_embeddings"}),
        ),
    }
    _llm_registry: Dict[str, ProviderRegistration] = {
        "openai": ProviderRegistration(
            provider_class=OpenAILLMProvider,
            default_model_name="gpt-4o-mini",
            supported_kwargs=frozenset({"api_key", "client", "mock_responses"}),
        ),
        "huggingface": ProviderRegistration(
            provider_class=HuggingFaceLLMProvider,
            default_model_name="distilgpt2",
            supported_kwargs=frozenset({"generator", "mock_responses"}),
        ),
    }

    @classmethod
    def register_embedding_provider(
        cls,
        provider_name: str,
        provider_class: Type[EmbeddingModel],
        default_model_name: str,
        supported_kwargs: Iterable[str],
    ) -> None:
        if not issubclass(provider_class, EmbeddingModel):
            raise ConfigurationError(
                "Embedding providers must inherit from EmbeddingModel",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                "embedding_provider",
            )

        cls._embedding_registry[provider_name] = ProviderRegistration(
            provider_class=provider_class,
            default_model_name=default_model_name,
            supported_kwargs=frozenset(supported_kwargs),
        )

    @classmethod
    def register_llm_provider(
        cls,
        provider_name: str,
        provider_class: Type[LLMProvider],
        default_model_name: str,
        supported_kwargs: Iterable[str],
    ) -> None:
        if not issubclass(provider_class, LLMProvider):
            raise ConfigurationError(
                "LLM providers must inherit from LLMProvider",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                "llm_provider",
            )

        cls._llm_registry[provider_name] = ProviderRegistration(
            provider_class=provider_class,
            default_model_name=default_model_name,
            supported_kwargs=frozenset(supported_kwargs),
        )

    @classmethod
    def get_supported_embedding_providers(cls) -> List[str]:
        return sorted(cls._embedding_registry.keys())

    @classmethod
    def get_supported_llm_providers(cls) -> List[str]:
        return sorted(cls._llm_registry.keys())

    @classmethod
    def get_embedding_provider_config_schema(cls, provider_name: str) -> Dict[str, Any]:
        registration = cls._get_registration("embedding", provider_name)
        return {
            "provider": provider_name,
            "default_model_name": registration.default_model_name,
            "supported_kwargs": sorted(registration.supported_kwargs),
        }

    @classmethod
    def get_llm_provider_config_schema(cls, provider_name: str) -> Dict[str, Any]:
        registration = cls._get_registration("llm", provider_name)
        return {
            "provider": provider_name,
            "default_model_name": registration.default_model_name,
            "supported_kwargs": sorted(registration.supported_kwargs),
        }

    @classmethod
    def validate_embedding_request(
        cls,
        provider_name: str,
        provider_config: Optional[Dict[str, Any]] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        cls._validate_provider_request(
            provider_type="embedding",
            provider_name=provider_name,
            provider_config=provider_config,
            fallback_providers=fallback_providers,
        )

    @classmethod
    def validate_llm_request(
        cls,
        provider_name: str,
        provider_config: Optional[Dict[str, Any]] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        cls._validate_provider_request(
            provider_type="llm",
            provider_name=provider_name,
            provider_config=provider_config,
            fallback_providers=fallback_providers,
        )

    @classmethod
    def create_embedding_provider(
        cls,
        provider_name: str,
        model_name: str,
        provider_config: Optional[Dict[str, Any]] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> EmbeddingModel:
        cls.validate_embedding_request(
            provider_name=provider_name,
            provider_config=provider_config,
            fallback_providers=fallback_providers,
        )
        providers = [
            cls._build_embedding_provider(
                provider_name=provider_name,
                model_name=model_name,
                provider_config=provider_config,
            )
        ]
        providers.extend(cls._build_embedding_fallback_chain(fallback_providers))

        if len(providers) == 1:
            return providers[0]
        return FallbackEmbeddingProvider(providers)

    @classmethod
    def create_llm_provider(
        cls,
        provider_name: str,
        model_name: str,
        provider_config: Optional[Dict[str, Any]] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMProvider:
        cls.validate_llm_request(
            provider_name=provider_name,
            provider_config=provider_config,
            fallback_providers=fallback_providers,
        )
        providers = [
            cls._build_llm_provider(
                provider_name=provider_name,
                model_name=model_name,
                provider_config=provider_config,
            )
        ]
        providers.extend(cls._build_llm_fallback_chain(fallback_providers))

        if len(providers) == 1:
            return providers[0]
        return FallbackLLMProvider(providers)

    @classmethod
    def _validate_provider_request(
        cls,
        provider_type: str,
        provider_name: str,
        provider_config: Optional[Dict[str, Any]],
        fallback_providers: Optional[List[Dict[str, Any]]],
    ) -> None:
        cls._validate_provider_config(provider_type, provider_name, provider_config)

        if fallback_providers is None:
            return
        if not isinstance(fallback_providers, list):
            raise ConfigurationError(
                "Fallback providers must be provided as a list",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"{provider_type}_fallbacks",
            )

        for index, fallback in enumerate(fallback_providers):
            if not isinstance(fallback, dict):
                raise ConfigurationError(
                    "Each fallback provider must be a dictionary",
                    ErrorCode.CONFIG_VALIDATION_FAILED,
                    f"{provider_type}_fallbacks[{index}]",
                )

            fallback_provider_name = fallback.get("provider")
            if not fallback_provider_name:
                raise ConfigurationError(
                    "Fallback provider entries must include a provider name",
                    ErrorCode.CONFIG_MISSING_REQUIRED,
                    f"{provider_type}_fallbacks[{index}].provider",
                )

            cls._validate_provider_config(
                provider_type=provider_type,
                provider_name=fallback_provider_name,
                provider_config=fallback.get("config"),
            )

    @classmethod
    def _validate_provider_config(
        cls,
        provider_type: str,
        provider_name: str,
        provider_config: Optional[Dict[str, Any]],
    ) -> None:
        registration = cls._get_registration(provider_type, provider_name)
        if provider_config is None:
            return
        if not isinstance(provider_config, dict):
            raise ConfigurationError(
                "Provider configuration must be a dictionary",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"{provider_type}_provider_config",
            )

        unsupported_keys = sorted(set(provider_config.keys()) - registration.supported_kwargs)
        if unsupported_keys:
            raise ConfigurationError(
                f"Unsupported {provider_type} provider settings for {provider_name}: {', '.join(unsupported_keys)}",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"{provider_type}_provider_config",
            )

    @classmethod
    def _build_embedding_provider(
        cls,
        provider_name: str,
        model_name: str,
        provider_config: Optional[Dict[str, Any]],
    ) -> EmbeddingModel:
        registration = cls._get_registration("embedding", provider_name)
        resolved_model_name = model_name or registration.default_model_name
        kwargs = dict(provider_config or {})
        return registration.provider_class(model_name=resolved_model_name, **kwargs)

    @classmethod
    def _build_llm_provider(
        cls,
        provider_name: str,
        model_name: str,
        provider_config: Optional[Dict[str, Any]],
    ) -> LLMProvider:
        registration = cls._get_registration("llm", provider_name)
        resolved_model_name = model_name or registration.default_model_name
        kwargs = dict(provider_config or {})
        return registration.provider_class(model_name=resolved_model_name, **kwargs)

    @classmethod
    def _build_embedding_fallback_chain(
        cls,
        fallback_providers: Optional[List[Dict[str, Any]]],
    ) -> List[EmbeddingModel]:
        providers: List[EmbeddingModel] = []
        for fallback in fallback_providers or []:
            providers.append(
                cls._build_embedding_provider(
                    provider_name=fallback["provider"],
                    model_name=fallback.get("model_name", ""),
                    provider_config=fallback.get("config"),
                )
            )
        return providers

    @classmethod
    def _build_llm_fallback_chain(
        cls,
        fallback_providers: Optional[List[Dict[str, Any]]],
    ) -> List[LLMProvider]:
        providers: List[LLMProvider] = []
        for fallback in fallback_providers or []:
            providers.append(
                cls._build_llm_provider(
                    provider_name=fallback["provider"],
                    model_name=fallback.get("model_name", ""),
                    provider_config=fallback.get("config"),
                )
            )
        return providers

    @classmethod
    def _get_registration(cls, provider_type: str, provider_name: str) -> ProviderRegistration:
        registry = cls._embedding_registry if provider_type == "embedding" else cls._llm_registry
        registration = registry.get(provider_name)
        if registration is None:
            raise ConfigurationError(
                f"Unsupported {provider_type} provider: {provider_name}",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"{provider_type}_provider",
            )
        return registration
