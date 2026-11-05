"""Cross-field arithmetic consistency rules."""

from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel

from ...schemas.defaults import InvoiceData, ReceiptData
from ..base import BaseValidationRule, RuleFinding, RuleSeverity


class LineItemsSumRule(BaseValidationRule):
    """Check that line item amounts sum to subtotal."""

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "line_items_sum_to_subtotal"

    @property
    def description(self) -> str:
        return "Verifies that the sum of line item amounts equals the subtotal"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)
        subtotal = data.get("subtotal")

        if items is None or subtotal is None:
            return findings

        computed_sum = sum(
            item.get("amount", 0) or 0 for item in items if isinstance(item, dict)
        )

        diff = abs(computed_sum - subtotal)
        if diff > self.tolerance:
            severity = RuleSeverity.ERROR if diff > 1.0 else RuleSeverity.WARNING
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=severity,
                    message=(
                        f"Line items sum ({computed_sum:.2f}) does not match "
                        f"subtotal ({subtotal:.2f}), diff={diff:.2f}"
                    ),
                    field_name="subtotal",
                    confidence_impact=0.15 if severity == RuleSeverity.ERROR else 0.05,
                    is_fraud_signal=severity == RuleSeverity.ERROR,
                    fraud_weight=0.2 if severity == RuleSeverity.ERROR else 0.0,
                )
            )

        return findings


class SubtotalTaxTotalRule(BaseValidationRule):
    """Check that subtotal + tax - discount = total."""

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "subtotal_plus_tax_equals_total"

    @property
    def description(self) -> str:
        return "Verifies that subtotal + tax - discount equals total"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        subtotal = data.get("subtotal")

        # InvoiceData uses tax_amount/total_amount/discount, ReceiptData uses tax/total
        if issubclass(schema, InvoiceData):
            tax = data.get("tax_amount")
            total = data.get("total_amount")
            discount = data.get("discount") or 0.0
        else:
            tax = data.get("tax")
            total = data.get("total")
            discount = 0.0

        if subtotal is None or total is None:
            return findings

        tax = tax or 0.0
        expected_total = subtotal + tax - discount

        diff = abs(expected_total - total)
        if diff > self.tolerance:
            severity = RuleSeverity.ERROR if diff > 1.0 else RuleSeverity.WARNING
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=severity,
                    message=(
                        f"Subtotal ({subtotal:.2f}) + tax ({tax:.2f}) - discount ({discount:.2f}) "
                        f"= {expected_total:.2f}, but total is {total:.2f}, diff={diff:.2f}"
                    ),
                    field_name="total_amount"
                    if issubclass(schema, InvoiceData)
                    else "total",
                    confidence_impact=0.15 if severity == RuleSeverity.ERROR else 0.05,
                    is_fraud_signal=severity == RuleSeverity.ERROR,
                    fraud_weight=0.25 if severity == RuleSeverity.ERROR else 0.0,
                )
            )

        return findings


class LineItemMathRule(BaseValidationRule):
    """Check that quantity * unit_price ~= amount for each line item."""

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "line_item_internal_math"

    @property
    def description(self) -> str:
        return "Verifies that quantity * unit_price equals amount for each line item"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)
        if not items:
            return findings

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            quantity = item.get("quantity")
            amount = item.get("amount")
            unit_price = item.get("unit_price_with_taxes") or item.get(
                "unit_price_without_taxes"
            )

            if quantity is not None and unit_price is not None and amount is not None:
                expected = quantity * unit_price
                diff = abs(expected - amount)
                if diff > self.tolerance:
                    item_name = item.get("name", f"item #{i + 1}")
                    findings.append(
                        RuleFinding(
                            rule_name=self.name,
                            severity=RuleSeverity.WARNING,
                            message=(
                                f"Line item '{item_name}': quantity ({quantity}) * "
                                f"unit_price ({unit_price:.2f}) = {expected:.2f}, "
                                f"but amount is {amount:.2f}"
                            ),
                            field_name=f"{items_key}[{i}].amount",
                            confidence_impact=0.05,
                        )
                    )

        return findings


class TaxConsistencyRule(BaseValidationRule):
    """Check that taxes match taxes_percentage * base price for line items."""

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "tax_percentage_consistency"

    @property
    def description(self) -> str:
        return "Verifies that taxes match taxes_percentage * base price for line items"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)
        if not items:
            return findings

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            taxes = item.get("taxes")
            taxes_pct = item.get("taxes_percentage")
            base_price = item.get("unit_price_without_taxes") or item.get("amount")

            if taxes is not None and taxes_pct is not None and base_price is not None:
                expected_taxes = base_price * taxes_pct / 100.0
                diff = abs(expected_taxes - taxes)
                if diff > self.tolerance:
                    item_name = item.get("name", f"item #{i + 1}")
                    findings.append(
                        RuleFinding(
                            rule_name=self.name,
                            severity=RuleSeverity.WARNING,
                            message=(
                                f"Line item '{item_name}': expected taxes "
                                f"{expected_taxes:.2f} ({taxes_pct}% of {base_price:.2f}), "
                                f"but got {taxes:.2f}"
                            ),
                            field_name=f"{items_key}[{i}].taxes",
                            confidence_impact=0.05,
                        )
                    )

        return findings
