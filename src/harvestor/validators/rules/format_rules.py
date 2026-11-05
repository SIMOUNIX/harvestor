"""Date, currency, tax ID, and card format validation rules."""

import re
from typing import Any, Dict, List, Optional, Set, Type

from pydantic import BaseModel

from ...schemas.defaults import InvoiceData, ReceiptData
from ..base import BaseValidationRule, RuleFinding, RuleSeverity

# Common date patterns that LLMs produce
_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",  # 2024-01-15
    r"\d{2}/\d{2}/\d{4}",  # 01/15/2024 or 15/01/2024
    r"\d{2}-\d{2}-\d{4}",  # 01-15-2024
    r"\d{1,2}\s+\w+\s+\d{4}",  # 15 January 2024
    r"\w+\s+\d{1,2},?\s+\d{4}",  # January 15, 2024
    r"\d{2}\.\d{2}\.\d{4}",  # 15.01.2024
]

_DATE_REGEX = re.compile("|".join(f"(?:{p})" for p in _DATE_PATTERNS))

# Common ISO 4217 currency codes
_VALID_CURRENCIES = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "CNY",
        "HKD",
        "SGD",
        "SEK",
        "NOK",
        "DKK",
        "KRW",
        "INR",
        "BRL",
        "MXN",
        "ZAR",
        "RUB",
        "TRY",
        "PLN",
        "CZK",
        "HUF",
        "ILS",
        "THB",
        "MYR",
        "PHP",
        "IDR",
        "TWD",
        "AED",
        "SAR",
        "ARS",
        "CLP",
        "COP",
        "PEN",
        "EGP",
        "NGN",
        "KES",
        "MAD",
    }
)


class DateFormatRule(BaseValidationRule):
    """Check that date fields are parseable."""

    @property
    def name(self) -> str:
        return "date_format_valid"

    @property
    def description(self) -> str:
        return "Verifies that date fields match common date patterns"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData, ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []
        date_fields = ["date"]
        if issubclass(schema, InvoiceData):
            date_fields.append("due_date")

        for field_name in date_fields:
            value = data.get(field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                continue
            if not _DATE_REGEX.search(value):
                findings.append(
                    RuleFinding(
                        rule_name=self.name,
                        severity=RuleSeverity.WARNING,
                        message=f"Field '{field_name}' value '{value}' does not match common date formats",
                        field_name=field_name,
                        confidence_impact=0.05,
                    )
                )

        return findings


class CurrencyCodeRule(BaseValidationRule):
    """Check that currency code is a valid ISO 4217 code."""

    @property
    def name(self) -> str:
        return "currency_code_valid"

    @property
    def description(self) -> str:
        return "Verifies that currency field is a valid ISO 4217 code"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []
        currency = data.get("currency")
        if currency is None:
            return findings

        if not isinstance(currency, str):
            return findings

        normalized = currency.strip().upper()
        if normalized not in _VALID_CURRENCIES:
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"Currency code '{currency}' is not a recognized ISO 4217 code",
                    field_name="currency",
                    confidence_impact=0.05,
                )
            )

        return findings


class TaxIdFormatRule(BaseValidationRule):
    """Check that vendor tax ID has a reasonable format."""

    @property
    def name(self) -> str:
        return "tax_id_format_valid"

    @property
    def description(self) -> str:
        return "Verifies that vendor tax ID is non-empty and has minimum length"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {InvoiceData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []
        tax_id = data.get("vendor_tax_id")
        if tax_id is None:
            return findings

        if not isinstance(tax_id, str):
            return findings

        cleaned = tax_id.strip()
        if len(cleaned) < 5:
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"Vendor tax ID '{tax_id}' is suspiciously short (< 5 characters)",
                    field_name="vendor_tax_id",
                    confidence_impact=0.05,
                    is_fraud_signal=True,
                    fraud_weight=0.1,
                )
            )

        return findings


class CardLastFourRule(BaseValidationRule):
    """Check that card_last_four is exactly 4 digits."""

    @property
    def name(self) -> str:
        return "card_last_four_format"

    @property
    def description(self) -> str:
        return "Verifies that card_last_four is exactly 4 digits"

    @property
    def supported_schemas(self) -> Optional[Set[Type[BaseModel]]]:
        return {ReceiptData}

    def validate(
        self, data: Dict[str, Any], schema: Type[BaseModel]
    ) -> List[RuleFinding]:
        findings = []
        card = data.get("card_last_four")
        if card is None:
            return findings

        card_str = str(card).strip()
        if not re.fullmatch(r"\d{4}", card_str):
            findings.append(
                RuleFinding(
                    rule_name=self.name,
                    severity=RuleSeverity.WARNING,
                    message=f"card_last_four '{card}' is not exactly 4 digits",
                    field_name="card_last_four",
                    confidence_impact=0.05,
                )
            )

        return findings
