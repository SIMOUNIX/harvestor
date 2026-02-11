"""Base abstractions for validation rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel


class RuleSeverity(str, Enum):
    """Severity level for a rule finding."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class RuleFinding:
    """A single finding from a validation rule."""

    rule_name: str
    severity: RuleSeverity
    message: str
    field_name: Optional[str] = None
    confidence_impact: float = 0.0
    is_fraud_signal: bool = False
    fraud_weight: float = 0.0


class BaseValidationRule(ABC):
    """Abstract base class for all validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this rule."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this rule checks."""
        ...

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        """Set of schema types this rule applies to. None means all schemas."""
        return None

    def applies_to(self, schema: Type[BaseModel]) -> bool:
        """Check if this rule applies to the given schema."""
        supported = self.supported_schemas
        if supported is None:
            return True
        return any(issubclass(schema, s) for s in supported)

    @abstractmethod
    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        """
        Run this rule against extracted data.

        Args:
            data: The extracted data dict (from HarvestResult.data)
            schema: The Pydantic schema class used for extraction

        Returns:
            List of findings (empty list means rule passed)
        """
        ...
