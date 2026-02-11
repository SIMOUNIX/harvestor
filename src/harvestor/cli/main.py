"""
CLI to use the tool in the command line.
"""

import argparse
import json
import sys
from pathlib import Path

from harvestor import DEFAULT_MODEL, harvest, list_models
from harvestor.schemas.defaults import InvoiceData, ReceiptData


def build_parser():
    parser = argparse.ArgumentParser(
        prog="harvestor",
        description="Extract structured data from documents using AI",
    )

    parser.add_argument(
        "file_path",
        type=Path,
        nargs="?",
        help="Path to the document to process",
    )
    parser.add_argument(
        "schema",
        nargs="?",
        help="Schema to use (e.g., InvoiceData, ReceiptData)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-schemas",
        action="store_true",
        help="List available schemas and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation rules on extracted data",
    )

    return parser


def get_schema(schema_name: str):
    """Resolve schema name to actual schema class."""
    schemas = {
        "InvoiceData": InvoiceData,
        "ReceiptData": ReceiptData,
    }

    if schema_name not in schemas:
        available = ", ".join(schemas.keys())
        raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

    return schemas[schema_name]


def print_models():
    """Print available models grouped by provider."""
    models = list_models()

    providers = {}
    for name, info in models.items():
        provider = info.get("provider", "unknown")
        if provider not in providers:
            providers[provider] = []
        providers[provider].append((name, info))

    print("\nAvailable models:")
    print("=" * 50)

    for provider, model_list in sorted(providers.items()):
        print(f"\n{provider.upper()}:")
        for name, info in sorted(model_list):
            vision = " (vision)" if info.get("supports_vision") else ""
            cost = info.get("input_cost", 0)
            if cost == 0:
                cost_str = "free"
            else:
                cost_str = f"${cost}/M tokens"
            print(f"  {name:<20} {cost_str}{vision}")

    print(f"\nDefault: {DEFAULT_MODEL}")
    print()


def print_schemas():
    """Print available schemas."""
    schemas = {
        "InvoiceData": InvoiceData,
        "ReceiptData": ReceiptData,
    }

    print("\nAvailable schemas:")
    print("=" * 50)

    for name, schema in schemas.items():
        doc = schema.__doc__ or "No description"
        print(f"  {name}: {doc.strip().split(chr(10))[0]}")

    print()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print_models()
        sys.exit(0)

    if args.list_schemas:
        print_schemas()
        sys.exit(0)

    if not args.file_path:
        parser.error("file_path is required")

    if not args.schema:
        parser.error("schema is required")

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        schema = get_schema(args.schema)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = harvest(
        source=args.file_path,
        schema=schema,
        model=args.model,
        validate=args.validate,
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None

    if result.validation:
        full_output = {
            "data": result.data,
            "validation": {
                "is_valid": result.validation.is_valid,
                "confidence": result.validation.confidence,
                "fraud_risk": result.validation.fraud_risk,
                "errors": result.validation.errors,
                "warnings": result.validation.warnings,
                "fraud_reasons": result.validation.fraud_reasons,
                "rules_checked": result.validation.rules_checked,
            },
        }
        output = json.dumps(full_output, indent=indent, default=str)
    else:
        output = json.dumps(result.data, indent=indent, default=str)

    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
