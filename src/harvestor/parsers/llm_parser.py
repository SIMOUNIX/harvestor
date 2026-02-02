"""
LLM-based document parser using provider abstraction.

Supports multiple LLM providers (Anthropic, OpenAI, Ollama) for extracting
structured data from text.
"""

import json
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from ..core.cost_tracker import cost_tracker
from ..providers import DEFAULT_MODEL, BaseLLMProvider, get_provider
from ..schemas.base import ExtractionResult, ExtractionStrategy
from ..schemas.prompt_builder import PromptBuilder


class LLMParser:
    """
    LLM-based parser for extracting structured data from text.

    Features:
    - Multi-provider support (Anthropic, OpenAI, Ollama)
    - Structured output with Pydantic validation
    - Automatic retry on validation errors
    - Cost tracking integration
    - Smart truncation for long documents
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_input_chars: int = 8000,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM parser.

        Args:
            model: Model to use (e.g., 'claude-haiku', 'gpt-4o-mini', 'llama3')
            api_key: API key (uses env var if not provided, not needed for Ollama)
            max_retries: Maximum retry attempts for failed extractions
            max_input_chars: Maximum characters to send to LLM
            base_url: Optional base URL override
        """
        self.model_name = model
        self.max_retries = max_retries
        self.max_input_chars = max_input_chars

        # Get provider for this model
        self.provider: BaseLLMProvider = get_provider(
            model=model, api_key=api_key, base_url=base_url
        )

        # Get model info for cost tracking
        self.model_info = self.provider.get_model_info()

        # Determine strategy based on provider
        self.strategy = self._get_strategy()

    def _get_strategy(self) -> ExtractionStrategy:
        """Determine extraction strategy based on provider."""
        provider_name = self.model_info.provider
        if provider_name == "anthropic":
            return ExtractionStrategy.LLM_ANTHROPIC
        elif provider_name == "openai":
            return ExtractionStrategy.LLM_OPENAI
        elif provider_name == "ollama":
            return ExtractionStrategy.LLM_OLLAMA
        return ExtractionStrategy.LLM_ANTHROPIC

    def truncate_text(self, text: str, max_chars: Optional[int] = None) -> str:
        """
        Smart truncation of text to fit token limits.

        Keeps beginning and end, truncates middle (where line items usually are).

        Args:
            text: Full text to truncate
            max_chars: Maximum characters (uses self.max_input_chars if not provided)

        Returns:
            Truncated text with marker where content was removed
        """
        max_chars = max_chars or self.max_input_chars

        if len(text) <= max_chars:
            return text

        # Keep first 60% and last 30% (drop middle line items)
        keep_start = int(max_chars * 0.6)
        keep_end = int(max_chars * 0.3)

        start = text[:keep_start]
        end = text[-keep_end:]

        removed_chars = len(text) - (keep_start + keep_end)
        removed_lines = text[keep_start:-keep_end].count("\n")

        truncation_marker = (
            f"\n\n[... {removed_lines} lines removed ({removed_chars} chars) ...]\n\n"
        )

        return start + truncation_marker + end

    def create_prompt(self, text: str, doc_type: str, schema: Type[BaseModel]) -> str:
        """
        Create extraction prompt from schema.

        Args:
            text: Document text to extract from
            doc_type: Type of document (invoice, receipt, etc.)
            schema: Pydantic model defining the output structure

        Returns:
            Prompt string with schema-derived fields
        """
        builder = PromptBuilder(schema)
        return builder.build_text_prompt(text, doc_type)

    def extract(
        self,
        text: str,
        schema: Type[BaseModel],
        doc_type: str = "document",
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from text using LLM.

        Args:
            text: Text to extract from
            schema: Pydantic model for structured output
            doc_type: Document type (defaults to "document")
            document_id: Optional document ID for cost tracking

        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()
        original_length = len(text)

        # Truncate if needed
        text = self.truncate_text(text)
        was_truncated = len(text) < original_length

        # Create prompt from schema
        prompt = self.create_prompt(text, doc_type, schema)

        # Try extraction with retries
        for attempt in range(self.max_retries):
            try:
                result = self._extract_with_provider(
                    prompt=prompt, schema=schema, document_id=document_id
                )

                processing_time = time.time() - start_time

                return ExtractionResult(
                    success=True,
                    data=result["data"],
                    raw_text=text[:500],
                    strategy=self.strategy,
                    confidence=result.get("confidence", 0.85),
                    processing_time=processing_time,
                    cost=result["cost"],
                    tokens_used=result["tokens"],
                    metadata={
                        "model": self.model_info.model_id,
                        "provider": self.model_info.provider,
                        "attempt": attempt + 1,
                        "truncated": was_truncated,
                    },
                )

            except ValidationError as e:
                if attempt < self.max_retries - 1:
                    continue
                else:
                    processing_time = time.time() - start_time
                    return ExtractionResult(
                        success=False,
                        data={},
                        strategy=self.strategy,
                        confidence=0.0,
                        processing_time=processing_time,
                        error=f"Validation failed after {self.max_retries} attempts: {str(e)}",
                    )

            except Exception as e:
                processing_time = time.time() - start_time
                return ExtractionResult(
                    success=False,
                    data={},
                    strategy=self.strategy,
                    confidence=0.0,
                    processing_time=processing_time,
                    error=f"Extraction failed: {str(e)}",
                )

        # Should not reach here, but handle edge case
        return ExtractionResult(
            success=False,
            data={},
            strategy=self.strategy,
            confidence=0.0,
            processing_time=time.time() - start_time,
            error="Extraction failed: max retries exceeded",
        )

    def _extract_with_provider(
        self, prompt: str, schema: Type[BaseModel], document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using the configured provider.

        Args:
            prompt: Prompt text
            schema: Pydantic schema for validation
            document_id: Optional document ID

        Returns:
            Dict with data, cost, and tokens
        """
        # Call provider
        result = self.provider.complete(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
        )

        if not result.success:
            raise RuntimeError(result.error or "Provider returned unsuccessful result")

        # Track cost
        cost = cost_tracker.track_call(
            model=self.model_name,
            strategy=self.strategy,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            document_id=document_id,
            success=True,
        )

        # Parse response
        response_text = result.content

        # Try to extract JSON from response
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)

            # Validate against schema
            validated_data = schema(**data)

            return {
                "data": validated_data.model_dump(),
                "cost": cost,
                "tokens": result.total_tokens,
                "confidence": 0.85,
            }

        except json.JSONDecodeError as e:
            raise ValidationError(f"Failed to parse JSON: {str(e)}")

    def extract_vision(
        self,
        image_data: bytes,
        schema: Type[BaseModel],
        doc_type: str = "document",
        document_id: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> ExtractionResult:
        """
        Extract structured data from an image using vision API.

        Args:
            image_data: Raw image bytes
            schema: Pydantic model for structured output
            doc_type: Document type
            document_id: Optional document ID for cost tracking
            media_type: Image MIME type

        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()

        if not self.provider.supports_vision():
            return ExtractionResult(
                success=False,
                data={},
                strategy=self.strategy,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=f"Model {self.model_name} does not support vision",
            )

        # Create vision prompt
        builder = PromptBuilder(schema)
        prompt = builder.build_vision_prompt(doc_type)

        try:
            result = self.provider.complete_vision(
                prompt=prompt,
                image_data=image_data,
                media_type=media_type,
                max_tokens=2048,
                temperature=0.0,
            )

            if not result.success:
                raise RuntimeError(result.error or "Vision API failed")

            # Track cost
            cost = cost_tracker.track_call(
                model=self.model_name,
                strategy=self.strategy,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                document_id=document_id,
                success=True,
            )

            # Parse JSON response
            response_text = result.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)

            validated_data = schema(**data)

            processing_time = time.time() - start_time

            return ExtractionResult(
                success=True,
                data=validated_data.model_dump(),
                raw_text=response_text[:500],
                strategy=self.strategy,
                confidence=0.85,
                processing_time=processing_time,
                cost=cost,
                tokens_used=result.total_tokens,
                metadata={
                    "model": self.model_info.model_id,
                    "provider": self.model_info.provider,
                    "vision": True,
                    "media_type": media_type,
                },
            )

        except json.JSONDecodeError as e:
            return ExtractionResult(
                success=False,
                data={},
                strategy=self.strategy,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=f"Failed to parse JSON response: {str(e)}",
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                data={},
                strategy=self.strategy,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=f"Vision extraction failed: {str(e)}",
            )
