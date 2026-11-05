"""Validation engine that runs rules and produces ValidationResult."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ..schemas.base import ValidationResult
from .base import BaseValidationRule, RuleFinding, RuleSeverity


class ValidationEngine:
    """
    Runs validation rules against extracted data and produces a ValidationResult.

    Usage:
        engine = ValidationEngine()  # loads built-in rules
        result = engine.validate(data, schema=InvoiceData)

        # Custom rules only:
        engine = ValidationEngine(rules=[MyRule()], include_defaults=False)
    """

    def __init__(
        self,
        rules: Optional[List[BaseValidationRule]] = None,
        include_defaults: bool = True,
    ):
        self._rules: List[BaseValidationRule] = []
        if include_defaults:
            self._rules.extend(self._get_default_rules())
        if rules:
            self._rules.extend(rules)

    @staticmethod
    def _get_default_rules() -> List[BaseValidationRule]:
        from .rules import get_all_default_rules

        return get_all_default_rules()

    @property
    def rules(self) -> List[BaseValidationRule]:
        return list(self._rules)

    def add_rule(self, rule: BaseValidationRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        self._rules = [r for r in self._rules if r.name != rule_name]

    def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
    ) -> ValidationResult:
        all_findings: List[RuleFinding] = []
        rules_checked: List[str] = []

        for rule in self._rules:
            if not rule.applies_to(schema):
                continue
            rules_checked.append(rule.name)
            try:
                findings = rule.validate(data, schema)
                all_findings.extend(findings)
            except Exception:
                all_findings.append(
                    RuleFinding(
                        rule_name=rule.name,
                        severity=RuleSeverity.WARNING,
                        message=f"Rule '{rule.name}' raised an exception and was skipped",
                    )
                )

        return self._build_result(all_findings, rules_checked)

    def _build_result(
        self,
        findings: List[RuleFinding],
        rules_checked: List[str],
    ) -> ValidationResult:
        errors = [f.message for f in findings if f.severity == RuleSeverity.ERROR]
        warnings = [f.message for f in findings if f.severity == RuleSeverity.WARNING]

        confidence = 1.0
        for f in findings:
            confidence -= f.confidence_impact
        confidence = max(0.0, min(1.0, confidence))

        fraud_findings = [f for f in findings if f.is_fraud_signal]
        fraud_checked = len(rules_checked) > 0
        fraud_reasons = [f.message for f in fraud_findings]
        fraud_risk = self._calculate_fraud_risk(fraud_findings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            fraud_checked=fraud_checked,
            fraud_risk=fraud_risk,
            fraud_reasons=fraud_reasons,
            cost=0.0,
            rules_checked=rules_checked,
            timestamp=datetime.now(),
        )

    @staticmethod
    def _calculate_fraud_risk(fraud_findings: List[RuleFinding]) -> str:
        if not fraud_findings:
            return "clean"

        total_weight = min(1.0, sum(f.fraud_weight for f in fraud_findings))

        if total_weight < 0.01:
            return "clean"
        elif total_weight < 0.2:
            return "low"
        elif total_weight < 0.5:
            return "medium"
        elif total_weight < 0.8:
            return "high"
        else:
            return "critical"
