"""
CLI to use the tool in the command line.
"""

import argparse
import json
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="harvestor",
        description="Extract structured data from documents using AI",
    )

    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the document to process",
    )
    parser.add_argument(
        "schema",
        help="Schema to use (e.g., InvoiceData, ReceiptData)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Claude Haiku 3",
        help="Model to use (default: Claude Haiku 3)",
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

    return parser


def get_schema(schema_name: str):
    """Resolve schema name to actual schema class."""
    from harvestor.schemas.defaults import InvoiceData, ReceiptData

    schemas = {
        "InvoiceData": InvoiceData,
        "ReceiptData": ReceiptData,
    }

    if schema_name not in schemas:
        available = ", ".join(schemas.keys())
        raise ValueError(f"Unknown schema: {schema_name}. Available: {available}")

    return schemas[schema_name]


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        schema = get_schema(args.schema)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from harvestor import harvest

    result = harvest(
        source=args.file_path,
        schema=schema,
        model=args.model,
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None
    output = json.dumps(result.data, indent=indent, default=str)

    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
