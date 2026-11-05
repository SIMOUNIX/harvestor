"""Built-in validation rules."""

from .anomaly_rules import DuplicateLineItemRule, ExtremeQuantityRule, RoundNumberRule
from .business_rules import (
    AmountThresholdRule,
    DueDateAfterIssueDateRule,
    EmptyLineItemsRule,
    NegativeAmountsRule,
    RequiredFieldsRule,
)
from .format_rules import (
    CardLastFourRule,
    CurrencyCodeRule,
    DateFormatRule,
    TaxIdFormatRule,
)
from .math_rules import (
    LineItemMathRule,
    LineItemsSumRule,
    SubtotalTaxTotalRule,
    TaxConsistencyRule,
)


def get_all_default_rules():
    """Instantiate all built-in rules with default configuration."""
    return [
        # Math
        LineItemsSumRule(),
        SubtotalTaxTotalRule(),
        LineItemMathRule(),
        TaxConsistencyRule(),
        # Format
        DateFormatRule(),
        CurrencyCodeRule(),
        TaxIdFormatRule(),
        CardLastFourRule(),
        # Business
        RequiredFieldsRule(),
        DueDateAfterIssueDateRule(),
        NegativeAmountsRule(),
        AmountThresholdRule(),
        EmptyLineItemsRule(),
        # Anomaly
        RoundNumberRule(),
        DuplicateLineItemRule(),
        ExtremeQuantityRule(),
    ]
