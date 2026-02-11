"""Anomaly detection rules for fraud signals."""

from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel

from ...schemas.defaults import InvoiceData, ReceiptData
from ..base import BaseValidationRule, RuleFinding, RuleSeverity


class RoundNumberRule(BaseValidationRule):
    """Flag suspiciously round total amounts."""

    def __init__(self, min_amount: float = 1000.0):
        self.min_amount = min_amount

    @property
    def name(self) -> str:
        return "round_number_anomaly"

    @property
    def description(self) -> str:
        return "Flags suspiciously round total amounts above threshold"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        total_field = "total_amount" if issubclass(schema, InvoiceData) else "total"
        total = data.get(total_field)

        if total is None or not isinstance(total, (int, float)):
            return findings

        if total < self.min_amount:
            return findings

        # Check if the amount is exactly round (no cents, divisible by 1000)
        if total == int(total) and int(total) % 1000 == 0:
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=(
                        f"Total amount ({total:.2f}) is a suspiciously round number"
                    ),
                    field_name=total_field,
                    confidence_impact=0.05,
                    is_fraud_signal=True,
                    fraud_weight=0.15,
                )
            )

        return findings


class DuplicateLineItemRule(BaseValidationRule):
    """Detect duplicate line items with identical name and amount."""

    @property
    def name(self) -> str:
        return "duplicate_line_items"

    @property
    def description(self) -> str:
        return "Detects line items with identical name and amount"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)
        if not items or not isinstance(items, list):
            return findings

        seen = {}
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            key = (item.get("name"), item.get("amount"))
            if key[0] is None and key[1] is None:
                continue
            if key in seen:
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.WARNING,
                        message=(
                            f"Duplicate line item: '{key[0]}' with amount {key[1]} "
                            f"appears at positions {seen[key]} and {i}"
                        ),
                        field_name=f"{items_key}[{i}]",
                        confidence_impact=0.05,
                        is_fraud_signal=True,
                        fraud_weight=0.2,
                    )
                )
            else:
                seen[key] = i

        return findings


class ExtremeQuantityRule(BaseValidationRule):
    """Flag line items with extreme quantities."""

    def __init__(self, max_quantity: float = 10_000.0):
        self.max_quantity = max_quantity

    @property
    def name(self) -> str:
        return "extreme_quantity"

    @property
    def description(self) -> str:
        return "Flags line items with quantities outside normal range"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)
        if not items or not isinstance(items, list):
            return findings

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            quantity = item.get("quantity")
            if quantity is None or not isinstance(quantity, (int, float)):
                continue

            item_name = item.get("name", f"item #{i + 1}")

            if quantity < 0:
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.WARNING,
                        message=f"Line item '{item_name}' has negative quantity: {quantity}",
                        field_name=f"{items_key}[{i}].quantity",
                        confidence_impact=0.1,
                        is_fraud_signal=True,
                        fraud_weight=0.15,
                    )
                )
            elif quantity > self.max_quantity:
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.WARNING,
                        message=(
                            f"Line item '{item_name}' has extreme quantity: "
                            f"{quantity} (threshold: {self.max_quantity})"
                        ),
                        field_name=f"{items_key}[{i}].quantity",
                        confidence_impact=0.05,
                        is_fraud_signal=True,
                        fraud_weight=0.15,
                    )
                )

        return findings
