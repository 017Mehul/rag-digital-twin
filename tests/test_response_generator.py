"""
Tests for response generation and LLM provider abstraction.
"""

from typing import Any

import pytest
from hypothesis import given, strategies as st

from src.exceptions import ErrorCode, ResponseGenerationError
from src.models.rag_config import RAGConfig
from src.models.search_results import RetrievedContext
from src.providers import HuggingFaceLLMProvider, LLMProvider, OpenAILLMProvider
from src.response_generator import ResponseGenerator


class EchoLLMProvider(LLMProvider):
    provider_name = "echo"

    def __init__(self, response_text: str) -> None:
        super().__init__(model_name="echo-model")
        self.response_text = response_text

    def load_model(self) -> Any:
        self._loaded_backend = "test"
        return self

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        self._validate_text(prompt)
        return self.response_text


def _retrieved_context(sources: list[str], body: str = "FAISS supports semantic search.") -> RetrievedContext:
    formatted_sources = "\n\n".join(
        f"[Source {index}] {source}\n{body}"
        for index, source in enumerate(sources, start=1)
    )
    return RetrievedContext(
        formatted_context=formatted_sources or "No context available.",
        source_chunks=[],
        total_tokens=max(1, len(body.split()) * max(1, len(sources))),
        sources=sources,
        insufficient_context=False,
    )


class TestLLMProviders:
    def test_openai_provider_mock_mode(self):
        provider = OpenAILLMProvider(model_name="gpt-4o-mini", mock_responses=True)

        response = provider.generate("Explain vector search.")

        assert response
        assert provider.get_config()["provider"] == "openai"

    def test_huggingface_provider_mock_mode(self):
        provider = HuggingFaceLLMProvider(model_name="distilgpt2", mock_responses=True)

        response = provider.generate("Explain dense retrieval.")

        assert response
        assert provider.get_config()["provider"] == "huggingface"


class TestResponseGenerator:
    def test_create_generator_from_config_openai(self):
        config = RAGConfig(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            max_response_tokens=256,
            temperature=0.2,
        )

        generator = ResponseGenerator.from_config(
            config,
            provider_kwargs={"mock_responses": True},
        )

        info = generator.get_provider_info()

        assert info["provider"] == "openai"
        assert info["model_name"] == "gpt-4o-mini"
        assert generator.max_response_tokens == 256
        assert generator.temperature == 0.2

    def test_generate_response_adds_structured_citations(self):
        generator = ResponseGenerator(
            provider=OpenAILLMProvider(model_name="gpt-4o-mini", mock_responses=True)
        )
        context = _retrieved_context(["doc1.txt", "doc2.txt"])

        response = generator.generate_response("What does the context say?", context)

        assert response.context_used is True
        assert response.sources == ["doc1.txt", "doc2.txt"]
        assert set(generator.extract_citations(response.response_text)) == {"doc1.txt", "doc2.txt"}
        assert response.confidence_score > 0.0

    def test_generate_response_handles_insufficient_context(self):
        generator = ResponseGenerator(
            provider=HuggingFaceLLMProvider(model_name="distilgpt2", mock_responses=True)
        )
        context = RetrievedContext(
            formatted_context="Insufficient context to answer this query reliably.",
            source_chunks=[],
            total_tokens=7,
            sources=[],
            insufficient_context=True,
        )

        response = generator.generate_response("What is the answer?", context)

        assert response.context_used is False
        assert response.sources == []
        assert response.confidence_score == 0.0
        assert "enough relevant context" in response.response_text.lower()

    def test_validation_rejects_inaccurate_citations(self):
        generator = ResponseGenerator(provider=EchoLLMProvider("Answer.\n\nSources:\n- wrong.txt"))
        context = _retrieved_context(["doc1.txt"])

        response = generator.generate_response("Summarize the context.", context)

        assert generator.validate_response(response.response_text, context)
        assert generator.extract_citations(response.response_text) == ["doc1.txt"]

    def test_empty_query_raises_error(self):
        generator = ResponseGenerator(provider=EchoLLMProvider("Some answer"))

        with pytest.raises(ResponseGenerationError) as exc_info:
            generator.generate_response(
                "   ",
                _retrieved_context(["doc1.txt"]),
            )

        assert exc_info.value.error_code == ErrorCode.LLM_GENERATION_FAILED


class TestResponseGeneratorProperties:
    @given(
        sources=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd"),
                    whitelist_characters="._/-",
                ),
                min_size=3,
                max_size=20,
            ).filter(lambda value: value.strip() and "\n" not in value and "," not in value),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    def test_source_attribution_completeness(self, sources):
        """
        Property 16: Generated responses should include all context sources.
        """
        generator = ResponseGenerator(provider=OpenAILLMProvider(model_name="gpt-4o-mini", mock_responses=True))
        context = _retrieved_context(sources)

        response = generator.generate_response("What can be concluded?", context)
        citations = generator.extract_citations(response.response_text)

        assert set(citations) == set(sources)

    @given(
        sources=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd"),
                    whitelist_characters="._/-",
                ),
                min_size=3,
                max_size=20,
            ).filter(lambda value: value.strip() and "\n" not in value and "," not in value),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    def test_citation_accuracy(self, sources):
        """
        Property 17: Cited sources should be accurate and drawn from the retrieved context.
        """
        provider = EchoLLMProvider("Model draft answer.\n\nSources:\n- fabricated.txt")
        generator = ResponseGenerator(provider=provider)
        context = _retrieved_context(sources)

        response = generator.generate_response("Provide an answer.", context)
        citations = generator.extract_citations(response.response_text)

        assert all(citation in sources for citation in citations)
        assert set(citations) == set(sources)
