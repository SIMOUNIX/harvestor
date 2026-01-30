"""Lightweight regex-based PII redaction for privacy-preserving document extraction.

Zero external dependencies - uses only Python stdlib.
"""

from harvestor.privacy.redactor import DEFAULT_ENTITIES, PIIRedactor, PlaceholderMap

__all__ = ["PIIRedactor", "PlaceholderMap", "DEFAULT_ENTITIES"]
