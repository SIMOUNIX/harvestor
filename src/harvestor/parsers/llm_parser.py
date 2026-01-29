"""
LLM-based document parser using LangChain and Anthropic.

Uses Claude Haiku (or other models) for extracting structured data from text.
"""

import json
import time
from typing import Any, Dict, Optional, Type

from anthropic import Anthropic
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, ValidationError

from ..core.cost_tracker import cost_tracker
from ..schemas.base import ExtractionResult, ExtractionStrategy


class InvoiceData(BaseModel):
    """Default invoice schema for structured extraction."""

    invoice_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    line_items: Optional[list] = None
    tax_amount: Optional[float] = None
    subtotal: Optional[float] = None


class LLMParser:
    """
    LLM-based parser for extracting structured data from text.

    Features:
    - Uses Claude Haiku by default (cheapest)
    - Structured output with Pydantic validation
    - Automatic retry on validation errors
    - Cost tracking integration
    - Smart truncation for long documents
    """

    def __init__(
        self,
        model: str = "Claude Haiku 3",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_input_chars: int = 8000,
    ):
        """
        Initialize LLM parser.

        Args:
            model: Model to use (default: Claude Haiku)
            api_key: Anthropic API key (uses env var if not provided)
            max_retries: Maximum retry attempts for failed extractions
            max_input_chars: Maximum characters to send to LLM
        """
        self.model = model
        self.max_retries = max_retries
        self.max_input_chars = max_input_chars

        # Initialize LangChain LLM
        self.llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.0,  # Deterministic for data extraction do not hallucanite
        )

        # Initialize Anthropic client for direct API access
        self.anthropic_client = Anthropic(api_key=api_key)

        # Determine strategy based on model
        if "haiku" in model.lower():
            self.strategy = ExtractionStrategy.LLM_HAIKU
        elif "sonnet" in model.lower():
            self.strategy = ExtractionStrategy.LLM_SONNET
        elif "gpt" in model.lower():
            self.strategy = ExtractionStrategy.LLM_GPT35
        else:
            self.strategy = ExtractionStrategy.LLM_HAIKU

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

    def create_prompt(self, text: str, doc_type: str = "invoice") -> str:
        """
        Create minimal prompt for extraction.

        Args:
            text: Document text to extract from
            doc_type: Type of document (invoice, receipt, etc.), based on your use case.

        Returns:
            Prompt string
        """
        # Minimal prompt to save tokens
        # TODO: Make this customer customisable with class, json, etc.
        if doc_type == "invoice":
            fields = "invoice_number, date, due_date, total_amount, currency, vendor_name, customer_name, line_items, tax_amount, subtotal"
        else:
            fields = "all relevant data fields"

        return f"""Extract structured data from this {doc_type}.

            Return JSON with fields: {fields}

            Document text:
            {text}

            JSON:"""

    def extract(
        self,
        text: str,
        doc_type: str = "invoice",
        schema: Type[BaseModel] = InvoiceData,
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from text using LLM.

        Args:
            text: Text to extract from
            doc_type: Document type
            schema: Pydantic model for structured output
            document_id: Optional document ID for cost tracking

        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()

        # Truncate if needed
        text = self.truncate_text(text)

        # Create prompt
        prompt = self.create_prompt(text, doc_type)

        # Try extraction with retries
        for attempt in range(self.max_retries):
            try:
                result = self._extract_with_anthropic(
                    prompt=prompt, schema=schema, document_id=document_id
                )

                processing_time = time.time() - start_time

                return ExtractionResult(
                    success=True,
                    data=result["data"],
                    raw_text=text[:500],  # Store first 500 chars
                    strategy=self.strategy,
                    confidence=result.get("confidence", 0.85),
                    processing_time=processing_time,
                    cost=result["cost"],
                    tokens_used=result["tokens"],
                    metadata={
                        "model": self.model,
                        "attempt": attempt + 1,
                        "truncated": len(text) < len(text),
                    },
                )

            except ValidationError as e:
                if attempt < self.max_retries - 1:
                    # Retry with error feedback
                    continue
                else:
                    # Final attempt failed
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

    def _extract_with_anthropic(
        self, prompt: str, schema: Type[BaseModel], document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using Anthropic API with structured output.

        Args:
            prompt: Prompt text
            schema: Pydantic schema for validation
            document_id: Optional document ID

        Returns:
            Dict with data, cost, and tokens
        """
        # Call Anthropic API
        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract tokens and calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost = cost_tracker.track_call(
            model=self.model,
            strategy=self.strategy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            document_id=document_id,
            success=True,
        )

        # Parse response
        response_text = response.content[0].text

        # Try to extract JSON from response
        try:
            # Find JSON in response (might have explanatory text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                # Try parsing entire response as JSON
                data = json.loads(response_text)

            # Validate against schema
            validated_data = schema(**data)

            return {
                "data": validated_data.dict(),
                "cost": cost,
                "tokens": input_tokens + output_tokens,
                "confidence": 0.85,  # Default confidence for LLM extraction
            }

        except json.JSONDecodeError as e:
            raise ValidationError(f"Failed to parse JSON: {str(e)}")

    def extract_with_langchain(
        self, text: str, doc_type: str = "invoice", document_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Alternative extraction using LangChain chains.

        This is simpler but doesn't use structured output.
        Good for experimentation.

        Might use this for experimentation with agentic AI @Koweez.
        Args:
            text: Text to extract from
            doc_type: Document type
            document_id: Optional document ID

        Returns:
            ExtractionResult
        """
        start_time = time.time()

        # Truncate if needed
        text = self.truncate_text(text)

        # Create LangChain prompt template
        prompt_template = PromptTemplate(
            input_variables=["text"], template=self.create_prompt("{text}", doc_type)
        )

        try:
            # Use LangChain LLM
            result = self.llm.predict(prompt_template.format(text=text))

            # Parse JSON response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(result)

            processing_time = time.time() - start_time

            # Note: We don't have token counts with LangChain predict
            # This is a limitation of using the simplified API
            estimated_cost = 0.02  # Rough estimate

            return ExtractionResult(
                success=True,
                data=data,
                raw_text=text[:500],
                strategy=self.strategy,
                confidence=0.80,
                processing_time=processing_time,
                cost=estimated_cost,
                tokens_used=0,  # Unknown with LangChain
                warnings=["Using LangChain predict - token counts unavailable"],
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                data={},
                strategy=self.strategy,
                confidence=0.0,
                processing_time=processing_time,
                error=f"LangChain extraction failed: {str(e)}",
            )
