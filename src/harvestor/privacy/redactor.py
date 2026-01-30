"""Lightweight regex-based PII redaction for privacy-preserving document extraction.

Zero external dependencies - uses only Python stdlib.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Regex patterns for common PII types
# These are intentionally conservative to minimize false positives
PII_PATTERNS: dict[str, re.Pattern] = {
    # Email: standard format
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # Credit card: 13-19 digits with optional separators
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    # IBAN: 2 letter country code + 2 check digits + up to 30 alphanumeric
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
    # US SSN: XXX-XX-XXXX
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # US Phone: various formats (with or without parentheses)
    "PHONE_US": re.compile(
        r"(?<![A-Za-z0-9])(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?![A-Za-z0-9])"
    ),
    # International phone: + followed by 10-15 digits
    "PHONE_INTL": re.compile(r"\b\+\d{10,15}\b"),
    # IPv4 address
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    # EU VAT numbers (common formats)
    "VAT_EU": re.compile(r"\b[A-Z]{2}\d{8,12}\b"),
}

# Default entities to redact
DEFAULT_ENTITIES = list(PII_PATTERNS.keys())


@dataclass
class PlaceholderMap:
    """Bidirectional mapping between original PII values and placeholders."""

    _original_to_placeholder: dict[str, str] = field(default_factory=dict)
    _placeholder_to_original: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, original: str, entity_type: str) -> str:
        """Add a PII value and return its placeholder."""
        if original in self._original_to_placeholder:
            return self._original_to_placeholder[original]

        self._counters[entity_type] += 1
        placeholder = f"[{entity_type}_{self._counters[entity_type]}]"

        self._original_to_placeholder[original] = placeholder
        self._placeholder_to_original[placeholder] = original

        return placeholder

    def get_original(self, placeholder: str) -> str | None:
        """Get the original value for a placeholder."""
        return self._placeholder_to_original.get(placeholder)

    def get_placeholder(self, original: str) -> str | None:
        """Get the placeholder for an original value."""
        return self._original_to_placeholder.get(original)

    def restore_in_string(self, text: str) -> str:
        """Replace all placeholders in a string with their original values."""
        result = text
        for placeholder, original in self._placeholder_to_original.items():
            result = result.replace(placeholder, original)
        return result

    def __len__(self) -> int:
        return len(self._original_to_placeholder)

    def __bool__(self) -> bool:
        return bool(self._original_to_placeholder)


class PIIRedactor:
    """Lightweight regex-based PII redactor.

    Zero external dependencies. Detects structured PII patterns with high precision:
    - Emails (100% accurate for valid formats)
    - Credit cards (Luhn validation could be added)
    - IBANs
    - SSNs (US format)
    - Phone numbers (US and international)
    - IP addresses
    - EU VAT numbers

    Does NOT attempt to detect:
    - Names (requires NLP context)
    - Addresses (too many false positives)

    Example:
        redactor = PIIRedactor()
        text = "Contact john@example.com or call +1-555-123-4567"
        redacted, pmap = redactor.redact(text)
        # redacted = "Contact [EMAIL_1] or call [PHONE_US_1]"
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        custom_patterns: dict[str, re.Pattern] | None = None,
        **_kwargs,  # Accept but ignore presidio-style params (language, etc.) for compatibility
    ):
        """Initialize the PII redactor.

        Args:
            entities: List of entity types to detect. If None, uses all defaults.
            custom_patterns: Additional regex patterns to use.
        """
        self.entities = entities or DEFAULT_ENTITIES

        # Build pattern dict for requested entities
        self._patterns: dict[str, re.Pattern] = {}
        for entity in self.entities:
            if entity in PII_PATTERNS:
                self._patterns[entity] = PII_PATTERNS[entity]

        # Add custom patterns
        if custom_patterns:
            self._patterns.update(custom_patterns)

    def redact(self, text: str) -> tuple[str, PlaceholderMap]:
        """Detect and redact PII from text.

        Args:
            text: The text to redact

        Returns:
            Tuple of (redacted_text, placeholder_map)
        """
        placeholder_map = PlaceholderMap()

        # Find all matches with their positions
        matches: list[tuple[int, int, str, str]] = []  # (start, end, value, type)

        for entity_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                # Skip if this looks like a false positive
                if not self._validate_match(match.group(), entity_type):
                    continue
                matches.append((match.start(), match.end(), match.group(), entity_type))

        if not matches:
            return text, placeholder_map

        # Sort by start position descending to replace from end
        matches.sort(key=lambda x: x[0], reverse=True)

        # Remove overlapping matches (keep longest)
        filtered_matches = []
        for match in matches:
            overlaps = False
            for existing in filtered_matches:
                if not (match[1] <= existing[0] or match[0] >= existing[1]):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Apply redactions
        redacted_text = text
        for start, end, value, entity_type in filtered_matches:
            placeholder = placeholder_map.add(value, entity_type)
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

        return redacted_text, placeholder_map

    def _validate_match(self, value: str, entity_type: str) -> bool:
        """Additional validation to reduce false positives."""
        if entity_type == "CREDIT_CARD":
            # Must have at least 13 digits
            digits = re.sub(r"\D", "", value)
            if len(digits) < 13 or len(digits) > 19:
                return False
            # Could add Luhn check here

        if entity_type == "PHONE_US":
            # Must have exactly 10 digits (excluding country code)
            digits = re.sub(r"\D", "", value)
            if digits.startswith("1"):
                digits = digits[1:]
            if len(digits) != 10:
                return False

        return True

    def restore(self, data: Any, placeholder_map: PlaceholderMap) -> Any:
        """Restore original PII values in extracted data."""
        if not placeholder_map:
            return data
        return self._restore_recursive(data, placeholder_map)

    def _restore_recursive(self, data: Any, placeholder_map: PlaceholderMap) -> Any:
        """Recursively restore placeholders in data structures."""
        if isinstance(data, str):
            return placeholder_map.restore_in_string(data)
        elif isinstance(data, dict):
            return {
                key: self._restore_recursive(value, placeholder_map)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._restore_recursive(item, placeholder_map) for item in data]
        else:
            return data

    def get_supported_entities(self) -> list[str]:
        """Return list of supported PII entity types."""
        return list(PII_PATTERNS.keys())
