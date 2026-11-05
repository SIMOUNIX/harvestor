"""Tests for the validation rules engine."""

from unittest.mock import MagicMock, patch

from harvestor import InvoiceData
from harvestor.schemas.base import ValidationResult
from harvestor.schemas.defaults import ReceiptData
from harvestor.validators import ValidationEngine, validate
from harvestor.validators.base import BaseValidationRule, RuleFinding, RuleSeverity
from harvestor.validators.rules.anomaly_rules import (
    DuplicateLineItemRule,
    ExtremeQuantityRule,
    RoundNumberRule,
)
from harvestor.validators.rules.business_rules import (
    AmountThresholdRule,
    DueDateAfterIssueDateRule,
    EmptyLineItemsRule,
    NegativeAmountsRule,
    RequiredFieldsRule,
)
from harvestor.validators.rules.format_rules import (
    CardLastFourRule,
    CurrencyCodeRule,
    DateFormatRule,
    TaxIdFormatRule,
)
from harvestor.validators.rules.math_rules import (
    LineItemMathRule,
    LineItemsSumRule,
    SubtotalTaxTotalRule,
    TaxConsistencyRule,
)


# ---------------------------------------------------------------------------
# Base abstractions
# ---------------------------------------------------------------------------


class TestRuleFindingAndSeverity:
    def test_severity_values(self):
        assert RuleSeverity.ERROR == "error"
        assert RuleSeverity.WARNING == "warning"
        assert RuleSeverity.INFO == "info"

    def test_rule_finding_construction(self):
        f = RuleFinding(
            rule_name="test",
            severity=RuleSeverity.ERROR,
            message="something wrong",
            field_name="total",
            confidence_impact=0.1,
            is_fraud_signal=True,
            fraud_weight=0.3,
        )
        assert f.rule_name == "test"
        assert f.severity == RuleSeverity.ERROR
        assert f.is_fraud_signal is True

    def test_rule_finding_defaults(self):
        f = RuleFinding(rule_name="x", severity=RuleSeverity.INFO, message="ok")
        assert f.field_name is None
        assert f.confidence_impact == 0.0
        assert f.is_fraud_signal is False
        assert f.fraud_weight == 0.0


# ---------------------------------------------------------------------------
# Validation Engine
# ---------------------------------------------------------------------------


class _DummyRule(BaseValidationRule):
    """A simple test rule that always produces one warning."""

    @property
    def name(self):
        return "dummy_rule"

    @property
    def description(self):
        return "dummy"

    def validate(self, data, schema):
        return [
            RuleFinding(
                rule_name=self.name,
                severity=RuleSeverity.WARNING,
                message="dummy warning",
            )
        ]


class _CrashingRule(BaseValidationRule):
    @property
    def name(self):
        return "crashing_rule"

    @property
    def description(self):
        return "always crashes"

    def validate(self, data, schema):
        raise RuntimeError("boom")


class TestValidationEngine:
    def test_engine_loads_default_rules(self):
        engine = ValidationEngine()
        assert len(engine.rules) > 0

    def test_engine_custom_rules_only(self):
        engine = ValidationEngine(rules=[_DummyRule()], include_defaults=False)
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "dummy_rule"

    def test_engine_add_remove_rule(self):
        engine = ValidationEngine(include_defaults=False)
        assert len(engine.rules) == 0

        engine.add_rule(_DummyRule())
        assert len(engine.rules) == 1

        engine.remove_rule("dummy_rule")
        assert len(engine.rules) == 0

    def test_engine_returns_validation_result(self, valid_invoice_data):
        engine = ValidationEngine()
        result = engine.validate(valid_invoice_data, InvoiceData)
        assert isinstance(result, ValidationResult)

    def test_engine_handles_crashing_rule(self, valid_invoice_data):
        engine = ValidationEngine(rules=[_CrashingRule()], include_defaults=False)
        result = engine.validate(valid_invoice_data, InvoiceData)
        assert isinstance(result, ValidationResult)
        assert any("exception" in w for w in result.warnings)

    def test_engine_records_rules_checked(self, valid_invoice_data):
        engine = ValidationEngine(rules=[_DummyRule()], include_defaults=False)
        result = engine.validate(valid_invoice_data, InvoiceData)
        assert "dummy_rule" in result.rules_checked

    def test_clean_data_is_valid(self, valid_invoice_data):
        engine = ValidationEngine()
        result = engine.validate(valid_invoice_data, InvoiceData)
        assert result.is_valid is True
        assert result.fraud_risk == "clean"


# ---------------------------------------------------------------------------
# Math Rules
# ---------------------------------------------------------------------------


class TestMathRules:
    def test_line_items_sum_valid(self, valid_invoice_data):
        rule = LineItemsSumRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_line_items_sum_mismatch_error(self, valid_invoice_data):
        valid_invoice_data["subtotal"] = 999.99
        rule = LineItemsSumRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].severity == RuleSeverity.ERROR

    def test_line_items_sum_small_mismatch_warning(self, valid_invoice_data):
        valid_invoice_data["subtotal"] = 300.50  # diff = 0.50, < 1.0
        rule = LineItemsSumRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].severity == RuleSeverity.WARNING

    def test_line_items_sum_skips_missing_fields(self):
        data = {"invoice_number": "INV-001"}
        rule = LineItemsSumRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0

    def test_line_items_sum_receipt(self, valid_receipt_data):
        rule = LineItemsSumRule()
        findings = rule.validate(valid_receipt_data, ReceiptData)
        assert len(findings) == 0

    def test_subtotal_tax_total_valid(self, valid_invoice_data):
        rule = SubtotalTaxTotalRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_subtotal_tax_total_mismatch(self, valid_invoice_data):
        valid_invoice_data["total_amount"] = 9999.00
        rule = SubtotalTaxTotalRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].severity == RuleSeverity.ERROR

    def test_subtotal_tax_total_receipt(self, valid_receipt_data):
        rule = SubtotalTaxTotalRule()
        findings = rule.validate(valid_receipt_data, ReceiptData)
        assert len(findings) == 0

    def test_line_item_math_valid(self, valid_invoice_data):
        rule = LineItemMathRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_line_item_math_mismatch(self, valid_invoice_data):
        valid_invoice_data["line_items"][0]["amount"] = 999.00  # 2 * 50 != 999
        rule = LineItemMathRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 1
        assert "Service A" in findings[0].message

    def test_tax_consistency_valid(self):
        data = {
            "line_items": [
                {
                    "name": "Item",
                    "unit_price_without_taxes": 100.00,
                    "taxes": 20.00,
                    "taxes_percentage": 20.0,
                }
            ]
        }
        rule = TaxConsistencyRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0

    def test_tax_consistency_mismatch(self):
        data = {
            "line_items": [
                {
                    "name": "Item",
                    "unit_price_without_taxes": 100.00,
                    "taxes": 50.00,  # should be 20.00
                    "taxes_percentage": 20.0,
                }
            ]
        }
        rule = TaxConsistencyRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1

    def test_tolerance_boundary(self):
        data = {
            "line_items": [{"name": "A", "amount": 100.01}],
            "subtotal": 100.00,  # diff = 0.01, within default tolerance
        }
        rule = LineItemsSumRule(tolerance=0.02)
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Format Rules
# ---------------------------------------------------------------------------


class TestFormatRules:
    def test_valid_date_formats(self, valid_invoice_data):
        rule = DateFormatRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_invalid_date_format(self):
        data = {"date": "not-a-date", "due_date": "also bad"}
        rule = DateFormatRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 2

    def test_various_valid_date_formats(self):
        for date_str in [
            "2024-01-15",
            "01/15/2024",
            "15 January 2024",
            "January 15, 2024",
            "15.01.2024",
        ]:
            data = {"date": date_str}
            rule = DateFormatRule()
            findings = rule.validate(data, InvoiceData)
            assert len(findings) == 0, f"Failed for date format: {date_str}"

    def test_valid_currency_code(self, valid_invoice_data):
        rule = CurrencyCodeRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_invalid_currency_code(self):
        data = {"currency": "FAKE"}
        rule = CurrencyCodeRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1

    def test_currency_case_insensitive(self):
        data = {"currency": "usd"}
        rule = CurrencyCodeRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0

    def test_tax_id_valid(self, valid_invoice_data):
        rule = TaxIdFormatRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_tax_id_too_short(self):
        data = {"vendor_tax_id": "AB"}
        rule = TaxIdFormatRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_valid_card_last_four(self, valid_receipt_data):
        rule = CardLastFourRule()
        findings = rule.validate(valid_receipt_data, ReceiptData)
        assert len(findings) == 0

    def test_invalid_card_last_four(self):
        data = {"card_last_four": "12AB"}
        rule = CardLastFourRule()
        findings = rule.validate(data, ReceiptData)
        assert len(findings) == 1

    def test_card_last_four_too_short(self):
        data = {"card_last_four": "12"}
        rule = CardLastFourRule()
        findings = rule.validate(data, ReceiptData)
        assert len(findings) == 1


# ---------------------------------------------------------------------------
# Business Rules
# ---------------------------------------------------------------------------


class TestBusinessRules:
    def test_required_fields_all_present(self, valid_invoice_data):
        rule = RequiredFieldsRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_required_fields_missing(self):
        data = {
            "subtotal": 100.00
        }  # missing invoice_number, date, total_amount, vendor_name
        rule = RequiredFieldsRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 4

    def test_required_fields_receipt(self, valid_receipt_data):
        rule = RequiredFieldsRule()
        findings = rule.validate(valid_receipt_data, ReceiptData)
        assert len(findings) == 0

    def test_required_fields_receipt_missing(self):
        data = {}
        rule = RequiredFieldsRule()
        findings = rule.validate(data, ReceiptData)
        assert len(findings) == 3  # merchant_name, date, total

    def test_due_date_after_issue_date_valid(self, valid_invoice_data):
        rule = DueDateAfterIssueDateRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_due_date_before_issue_date(self):
        data = {"date": "2024-06-15", "due_date": "2024-01-01"}
        rule = DueDateAfterIssueDateRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_due_date_skips_unparseable(self):
        data = {"date": "not-a-date", "due_date": "also-bad"}
        rule = DueDateAfterIssueDateRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0

    def test_negative_amount_detected(self):
        data = {"total_amount": -500.00}
        rule = NegativeAmountsRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].severity == RuleSeverity.ERROR
        assert findings[0].is_fraud_signal is True
        assert findings[0].fraud_weight == 0.3

    def test_positive_amounts_clean(self, valid_invoice_data):
        rule = NegativeAmountsRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_amount_threshold_exceeded(self):
        data = {"total_amount": 200_000.00}
        rule = AmountThresholdRule(threshold=100_000.0)
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_amount_threshold_within_range(self, valid_invoice_data):
        rule = AmountThresholdRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_empty_line_items_warning(self):
        data = {"line_items": []}
        rule = EmptyLineItemsRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1

    def test_non_empty_line_items_clean(self, valid_invoice_data):
        rule = EmptyLineItemsRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Anomaly Rules
# ---------------------------------------------------------------------------


class TestAnomalyRules:
    def test_round_number_flagged(self):
        data = {"total_amount": 10_000.00}
        rule = RoundNumberRule(min_amount=1000.0)
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_non_round_number_clean(self, valid_invoice_data):
        rule = RoundNumberRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_round_number_below_min_amount_clean(self):
        data = {"total_amount": 100.00}  # round but below min_amount
        rule = RoundNumberRule(min_amount=1000.0)
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 0

    def test_duplicate_line_items_detected(self):
        data = {
            "line_items": [
                {"name": "Widget", "amount": 50.00},
                {"name": "Widget", "amount": 50.00},
            ]
        }
        rule = DuplicateLineItemRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_unique_line_items_clean(self, valid_invoice_data):
        rule = DuplicateLineItemRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0

    def test_extreme_quantity_flagged(self):
        data = {"line_items": [{"name": "Bolts", "quantity": 50_000, "amount": 100.00}]}
        rule = ExtremeQuantityRule(max_quantity=10_000.0)
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1
        assert findings[0].is_fraud_signal is True

    def test_negative_quantity_flagged(self):
        data = {"line_items": [{"name": "Refund", "quantity": -5, "amount": 100.00}]}
        rule = ExtremeQuantityRule()
        findings = rule.validate(data, InvoiceData)
        assert len(findings) == 1

    def test_normal_quantity_clean(self, valid_invoice_data):
        rule = ExtremeQuantityRule()
        findings = rule.validate(valid_invoice_data, InvoiceData)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Fraud Risk Calculation
# ---------------------------------------------------------------------------


class TestFraudRiskCalculation:
    def test_clean_data_clean_risk(self, valid_invoice_data):
        result = validate(valid_invoice_data, InvoiceData)
        assert result.fraud_risk == "clean"

    def test_single_low_signal(self):
        data = {
            "total_amount": 150_123.45,
            "subtotal": 135_123.45,
            "tax_amount": 15_000.00,
            "date": "2024-01-01",
            "invoice_number": "INV-001",
            "vendor_name": "Vendor Co",
            "line_items": [{"name": "Big project", "amount": 135_123.45}],
        }
        result = validate(data, InvoiceData)
        # Only amount_threshold rule fires (fraud_weight=0.1) -> "low"
        assert result.fraud_risk == "low"

    def test_multiple_fraud_signals_escalate_risk(self):
        data = {
            "total_amount": -500.00,  # negative = fraud_weight 0.3
            "subtotal": -500.00,  # another negative = 0.3
            "tax_amount": -50.00,  # another negative = 0.3
            "date": "2024-01-01",
            "invoice_number": "X",
            "vendor_name": "V",
        }
        result = validate(data, InvoiceData)
        assert result.fraud_risk in ("high", "critical")
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    def test_validate_function(self, valid_invoice_data):
        result = validate(valid_invoice_data, InvoiceData)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_with_custom_rules(self, valid_invoice_data):
        result = validate(
            valid_invoice_data,
            InvoiceData,
            rules=[_DummyRule()],
            include_defaults=False,
        )
        assert len(result.warnings) == 1
        assert "dummy warning" in result.warnings[0]

    def test_validate_receipt(self, valid_receipt_data):
        result = validate(valid_receipt_data, ReceiptData)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Integration with Harvestor
# ---------------------------------------------------------------------------


class TestHarvestorValidationIntegration:
    @patch("harvestor.providers.anthropic.Anthropic")
    def test_validation_populates_harvest_result(
        self, mock_anthropic, sample_invoice_text, mock_anthropic_response, api_key
    ):
        from harvestor import Harvestor
        from harvestor.schemas.base import HarvestResult

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvestor = Harvestor(api_key=api_key, validate=True)
        result = harvestor.harvest_text(
            sample_invoice_text, schema=InvoiceData, doc_type="invoice"
        )

        assert isinstance(result, HarvestResult)
        assert result.validation is not None
        assert isinstance(result.validation, ValidationResult)
        assert result.validation.fraud_checked is True
        assert len(result.validation.rules_checked) > 0

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_validation_disabled_by_default(
        self, mock_anthropic, sample_invoice_text, mock_anthropic_response, api_key
    ):
        from harvestor import Harvestor

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_text(
            sample_invoice_text, schema=InvoiceData, doc_type="invoice"
        )

        assert result.validation is None

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_custom_rule_in_harvest(
        self, mock_anthropic, sample_invoice_text, mock_anthropic_response, api_key
    ):
        from harvestor import Harvestor

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvestor = Harvestor(
            api_key=api_key, validate=True, validation_rules=[_DummyRule()]
        )
        result = harvestor.harvest_text(
            sample_invoice_text, schema=InvoiceData, doc_type="invoice"
        )

        assert result.validation is not None
        assert "dummy_rule" in result.validation.rules_checked
