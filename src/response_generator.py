"""
Context-aware response generation using pluggable LLM providers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .exceptions import ErrorCode, ResponseGenerationError
from .models.rag_config import RAGConfig
from .models.search_results import GeneratedResponse, RetrievedContext
from .providers import HuggingFaceLLMProvider, LLMProvider, OpenAILLMProvider


class ResponseGenerator:
    """
    Generate validated, source-attributed responses from retrieved context.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_response_tokens: int = 500,
        temperature: float = 0.1,
        insufficient_context_message: str = "I do not have enough relevant context to answer this confidently.",
    ) -> None:
        if max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        self.provider = provider
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.insufficient_context_message = insufficient_context_message

    @classmethod
    def from_config(
        cls,
        config: RAGConfig,
        provider: Optional[LLMProvider] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        insufficient_context_message: str = "I do not have enough relevant context to answer this confidently.",
    ) -> "ResponseGenerator":
        if provider is None:
            provider = cls._create_provider(
                provider_name=config.llm_provider,
                model_name=config.llm_model,
                provider_kwargs=provider_kwargs,
            )

        return cls(
            provider=provider,
            max_response_tokens=config.max_response_tokens,
            temperature=config.temperature,
            insufficient_context_message=insufficient_context_message,
        )

    @staticmethod
    def _create_provider(
        provider_name: str,
        model_name: str,
        provider_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMProvider:
        kwargs = dict(provider_kwargs or {})

        if provider_name == "openai":
            return OpenAILLMProvider(model_name=model_name, **kwargs)
        if provider_name == "huggingface":
            return HuggingFaceLLMProvider(model_name=model_name, **kwargs)

        raise ResponseGenerationError(
            f"Unsupported LLM provider: {provider_name}",
            ErrorCode.LLM_MODEL_NOT_FOUND,
            model_name,
        )

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Return active provider metadata.
        """
        return self.provider.get_config()

    def create_prompt(self, query: str, context: str) -> str:
        """
        Build a grounded prompt that explicitly constrains the model to the context.
        """
        normalized_query = self._validate_query(query)
        normalized_context = self._validate_context(context)

        return (
            "You are a grounded RAG assistant. Answer the user's question using only the provided context. "
            "If the context is insufficient, say so clearly. Be concise, factual, and do not invent sources.\n\n"
            f"Context:\n{normalized_context}\n\n"
            f"Question:\n{normalized_query}\n\n"
            "Answer:"
        )

    def validate_response(self, response: str, context: RetrievedContext) -> bool:
        """
        Validate generated text for completeness, citations, and insufficient-context behavior.
        """
        if not isinstance(response, str) or not response.strip():
            return False

        if context.insufficient_context:
            return self._contains_uncertainty(response)

        citations = self.extract_citations(response)
        normalized_sources = self._normalize_sources(context.sources)

        if normalized_sources and set(citations) != set(normalized_sources):
            return False
        if self._looks_like_prompt_echo(response):
            return False

        return True

    def generate_response(self, query: str, context: RetrievedContext) -> GeneratedResponse:
        """
        Generate a context-aware response with source attribution and validation.
        """
        normalized_query = self._validate_query(query)
        start_time = time.perf_counter()

        if context.insufficient_context or not context.sources or not context.formatted_context.strip():
            response_text = self.insufficient_context_message
            return GeneratedResponse(
                response_text=response_text,
                sources=[],
                confidence_score=0.0,
                context_used=False,
                token_count=self.provider.count_tokens(response_text),
                model_used=self.provider.model_name,
                generation_time=time.perf_counter() - start_time,
            )

        prompt = self.create_prompt(normalized_query, context.formatted_context)

        try:
            raw_response = self.provider.generate(
                prompt,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature,
            )
        except ResponseGenerationError:
            raise
        except Exception as exc:
            raise ResponseGenerationError(
                f"Failed to generate LLM response: {exc}",
                ErrorCode.LLM_GENERATION_FAILED,
                self.provider.model_name,
                cause=exc,
            ) from exc

        response_text = self._finalize_response(raw_response, context)
        if not self.validate_response(response_text, context):
            response_text = self._attach_citations(
                self._fallback_answer_from_context(context),
                context.sources,
            )

        confidence_score = self._estimate_confidence(context, response_text)
        return GeneratedResponse(
            response_text=response_text,
            sources=self._normalize_sources(context.sources),
            confidence_score=confidence_score,
            context_used=True,
            token_count=self.provider.count_tokens(response_text),
            model_used=self.provider.model_name,
            generation_time=time.perf_counter() - start_time,
        )

    def extract_citations(self, response: str) -> List[str]:
        """
        Extract the normalized source list from the structured citation block.
        """
        marker = "\n\nSources:\n"
        if marker not in response:
            return []

        citation_block = response.split(marker, 1)[1]
        citations: List[str] = []
        for line in citation_block.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                citations.append(stripped[2:].strip())

        return citations

    def _finalize_response(self, raw_response: str, context: RetrievedContext) -> str:
        normalized = self._normalize_response(raw_response)
        if not normalized or self._looks_like_prompt_echo(normalized):
            normalized = self._fallback_answer_from_context(context)

        return self._attach_citations(normalized, context.sources)

    def _attach_citations(self, response: str, sources: List[str]) -> str:
        base_response = response.split("\n\nSources:\n", 1)[0].strip()
        normalized_sources = self._normalize_sources(sources)

        if not normalized_sources:
            return base_response

        citations = "\n".join(f"- {source}" for source in normalized_sources)
        return f"{base_response}\n\nSources:\n{citations}"

    @staticmethod
    def _normalize_sources(sources: List[str]) -> List[str]:
        normalized_sources: List[str] = []
        for source in sources:
            if source not in normalized_sources:
                normalized_sources.append(source)
        return normalized_sources

    def _fallback_answer_from_context(self, context: RetrievedContext) -> str:
        content_lines = [
            line.strip()
            for line in context.formatted_context.splitlines()
            if line.strip() and not line.strip().startswith("[Source ")
        ]
        summary = " ".join(content_lines).strip()
        if not summary:
            return "The retrieved context does not contain enough usable detail for a confident answer."

        preview = summary[:280].rstrip()
        if len(summary) > 280:
            preview += "..."
        return f"Based on the retrieved context, the available information indicates: {preview}"

    @staticmethod
    def _contains_uncertainty(response: str) -> bool:
        normalized = response.lower()
        return any(
            phrase in normalized
            for phrase in [
                "not enough",
                "insufficient context",
                "cannot answer confidently",
                "do not have enough",
                "don't have enough",
            ]
        )

    @staticmethod
    def _looks_like_prompt_echo(response: str) -> bool:
        normalized = response.lower()
        return "context:" in normalized and "question:" in normalized

    def _estimate_confidence(self, context: RetrievedContext, response: str) -> float:
        citation_ratio = 1.0 if self.extract_citations(response) else 0.0
        source_factor = min(len(self._normalize_sources(context.sources)) / 3.0, 1.0)
        token_factor = min(context.total_tokens / 200.0, 1.0)
        confidence = 0.35 + (0.25 * citation_ratio) + (0.2 * source_factor) + (0.2 * token_factor)
        return round(min(confidence, 0.95), 3)

    @staticmethod
    def _validate_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ResponseGenerationError(
                "Query cannot be empty for response generation",
                ErrorCode.LLM_GENERATION_FAILED,
            )
        return normalized_query

    @staticmethod
    def _validate_context(context: str) -> str:
        normalized_context = context.strip()
        if not normalized_context:
            raise ResponseGenerationError(
                "Context cannot be empty for response generation",
                ErrorCode.LLM_CONTEXT_TOO_LONG,
            )
        return normalized_context

    @staticmethod
    def _normalize_response(response: str) -> str:
        return " ".join(response.split()).strip()
