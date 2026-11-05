"""
Validation engine for extracted document data.

Provides rule-based validation, fraud detection, and anomaly flagging
for data extracted by Harvestor.
"""

from ..schemas.base import ValidationResult
from .base import BaseValidationRule, RuleFinding, RuleSeverity
from .engine import ValidationEngine


def validate(data, schema, rules=None, include_defaults=True) -> ValidationResult:
    """
    One-liner validation function.

    Args:
        data: Extracted data dict
        schema: Pydantic schema class
        rules: Optional custom rules
        include_defaults: Include built-in rules (default: True)

    Returns:
        ValidationResult
    """
    engine = ValidationEngine(rules=rules, include_defaults=include_defaults)
    return engine.validate(data, schema)


__all__ = [
    "BaseValidationRule",
    "RuleFinding",
    "RuleSeverity",
    "ValidationEngine",
    "validate",
]
