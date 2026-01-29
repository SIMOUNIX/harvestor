"""
Dynamic prompt generation from Pydantic schemas.

Inspects Pydantic models to generate extraction prompts with field names,
types, and descriptions.
"""

from typing import Any, List, Type, Union, get_args, get_origin

from pydantic import BaseModel


class PromptBuilder:
    """
    Generates extraction prompts from Pydantic schemas.

    Features:
    - Extracts field names, types, and descriptions from Pydantic models
    - Supports nested models and Optional types
    - Generates both text and vision API prompts
    """

    def __init__(self, schema: Type[BaseModel]):
        """
        Initialize with a Pydantic schema.

        Args:
            schema: Pydantic BaseModel class to generate prompts from
        """
        self.schema = schema
        self._field_specs = self._extract_field_specs()

    def _extract_field_specs(self) -> List[dict]:
        """
        Extract field specifications from the schema.

        Returns:
            List of dicts with field name, type, description, required status
        """
        specs = []

        for field_name, field_info in self.schema.model_fields.items():
            type_hint = field_info.annotation

            # Handle Optional types
            is_optional = False
            origin = get_origin(type_hint)
            if origin is Union:
                args = get_args(type_hint)
                if type(None) in args:
                    is_optional = True
                    # Get the non-None type
                    non_none_types = [a for a in args if a is not type(None)]
                    if non_none_types:
                        type_hint = non_none_types[0]

            # Get description from field info
            description = field_info.description
            if not description:
                # Generate default description from field name
                description = field_name.replace("_", " ").capitalize()

            spec = {
                "name": field_name,
                "type": self._format_type(type_hint),
                "description": description,
                "required": not is_optional and field_info.is_required(),
            }
            specs.append(spec)

        return specs

    def _format_type(self, type_hint: Any) -> str:
        """
        Convert Python type hint to human-readable string.

        Args:
            type_hint: Python type annotation

        Returns:
            Human-readable type string
        """
        origin = get_origin(type_hint)

        if origin is list or origin is List:
            args = get_args(type_hint)
            if args:
                inner_type = self._format_type(args[0])
                return f"list of {inner_type}"
            return "list"

        if origin is dict:
            return "object"

        if origin is Union:
            args = get_args(type_hint)
            non_none_types = [a for a in args if a is not type(None)]
            if non_none_types:
                return self._format_type(non_none_types[0])
            return "any"

        # Handle basic types
        if type_hint is str:
            return "string"
        if type_hint is int:
            return "integer"
        if type_hint is float:
            return "number"
        if type_hint is bool:
            return "boolean"

        # Handle Pydantic models (nested)
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return f"object ({type_hint.__name__})"

        # Fallback to type name
        if hasattr(type_hint, "__name__"):
            return type_hint.__name__.lower()

        return str(type_hint)

    def build_text_prompt(self, text: str, doc_type: str) -> str:
        """
        Build prompt for text extraction.

        Args:
            text: Document text to extract from
            doc_type: Human-readable document type

        Returns:
            Complete prompt string
        """
        fields_section = self._build_fields_section()

        return f"""Extract structured data from this {doc_type}.

Return a JSON object with the following fields:
{fields_section}

Extract all available information. If a field is not found, use null.
Return only the JSON object, no other text.

Document text:
{text}

JSON:"""

    def build_vision_prompt(self, doc_type: str) -> str:
        """
        Build prompt for vision API extraction.

        Args:
            doc_type: Human-readable document type

        Returns:
            Complete prompt string for vision API
        """
        fields_section = self._build_fields_section()

        return f"""Extract structured data from this {doc_type} image.

Return a JSON object with the following fields:
{fields_section}

Extract all available information. If a field is not found, use null.
Return only the JSON object, no other text."""

    def _build_fields_section(self) -> str:
        """
        Build the fields section for prompts.

        Returns:
            Formatted field list string
        """
        lines = []
        for spec in self._field_specs:
            required_marker = "(required)" if spec["required"] else "(optional)"
            lines.append(
                f"- {spec['name']}: {spec['description']} [{spec['type']}] {required_marker}"
            )
        return "\n".join(lines)

    def get_json_schema(self) -> dict:
        """
        Get JSON schema representation for structured output.

        Returns:
            JSON schema dict
        """
        return self.schema.model_json_schema()
