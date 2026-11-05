"""Business logic validation rules."""

from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel

from ...schemas.defaults import InvoiceData, ReceiptData
from ..base import BaseValidationRule, RuleFinding, RuleSeverity


def _try_parse_date(value: str):
    """Try to parse a date string into a comparable tuple (year, month, day)."""
    import re
    from datetime import datetime

    if not isinstance(value, str):
        return None

    # Try ISO format first: YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Try common strptime formats
    for fmt in [
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %b %Y",
        "%d.%m.%Y",
    ]:
        try:
            dt = datetime.strptime(value.strip().replace(",", ","), fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue

    return None


class RequiredFieldsRule(BaseValidationRule):
    """Check that critical fields are present and non-null."""

    @property
    def name(self) -> str:
        return "required_fields_present"

    @property
    def description(self) -> str:
        return "Verifies that critical fields are present and non-null"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        if issubclass(schema, InvoiceData):
            required = ["invoice_number", "date", "total_amount", "vendor_name"]
        else:
            required = ["merchant_name", "date", "total"]

        for field_name in required:
            value = data.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.WARNING,
                        message=f"Required field '{field_name}' is missing or empty",
                        field_name=field_name,
                        confidence_impact=0.05,
                    )
                )

        return findings


class DueDateAfterIssueDateRule(BaseValidationRule):
    """Check that due_date is on or after issue date."""

    @property
    def name(self) -> str:
        return "due_date_after_issue_date"

    @property
    def description(self) -> str:
        return "Verifies that due_date is on or after the issue date"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []
        date_str = data.get("date")
        due_date_str = data.get("due_date")

        if date_str is None or due_date_str is None:
            return findings

        issue_date = _try_parse_date(str(date_str))
        due_date = _try_parse_date(str(due_date_str))

        if issue_date is None or due_date is None:
            return findings

        if due_date < issue_date:
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"Due date ({due_date_str}) is before issue date ({date_str})",
                    field_name="due_date",
                    confidence_impact=0.1,
                    is_fraud_signal=True,
                    fraud_weight=0.15,
                )
            )

        return findings


class NegativeAmountsRule(BaseValidationRule):
    """Check that monetary amounts are non-negative."""

    @property
    def name(self) -> str:
        return "no_negative_amounts"

    @property
    def description(self) -> str:
        return "Verifies that monetary amounts are non-negative"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        if issubclass(schema, InvoiceData):
            amount_fields = ["total_amount", "subtotal", "tax_amount"]
        else:
            amount_fields = ["total", "subtotal", "tax"]

        for field_name in amount_fields:
            value = data.get(field_name)
            if value is not None and isinstance(value, (int, float)) and value < 0:
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.ERROR,
                        message=f"Field '{field_name}' has negative value: {value}",
                        field_name=field_name,
                        confidence_impact=0.15,
                        is_fraud_signal=True,
                        fraud_weight=0.3,
                    )
                )

        return findings


class AmountThresholdRule(BaseValidationRule):
    """Flag documents with unusually high total amounts."""

    def __init__(self, threshold: float = 100_000.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "amount_threshold"

    @property
    def description(self) -> str:
        return f"Flags documents with total amount exceeding {self.threshold}"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        total_field = "total_amount" if issubclass(schema, InvoiceData) else "total"
        total = data.get(total_field)

        if (
            total is not None
            and isinstance(total, (int, float))
            and total > self.threshold
        ):
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"Total amount ({total:.2f}) exceeds threshold ({self.threshold:.2f})",
                    field_name=total_field,
                    confidence_impact=0.05,
                    is_fraud_signal=True,
                    fraud_weight=0.1,
                )
            )

        return findings


class EmptyLineItemsRule(BaseValidationRule):
    """Check that line items list is not empty when present."""

    @property
    def name(self) -> str:
        return "line_items_not_empty"

    @property
    def description(self) -> str:
        return "Verifies that line items list is not empty when present"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []

        items_key = "line_items" if "line_items" in data else "items"
        items = data.get(items_key)

        if items is not None and isinstance(items, list) and len(items) == 0:
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"'{items_key}' is present but empty",
                    field_name=items_key,
                    confidence_impact=0.05,
                )
            )

        return findings
