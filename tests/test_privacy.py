"""Test PII redaction functionality (regex-based, zero dependencies)."""

from harvestor.privacy import PIIRedactor, PlaceholderMap


class TestPlaceholderMap:
    """Test PlaceholderMap functionality."""

    def test_add_creates_placeholder(self):
        """Test that add() creates a placeholder."""
        pmap = PlaceholderMap()
        placeholder = pmap.add("john@example.com", "EMAIL")

        assert placeholder == "[EMAIL_1]"

    def test_add_same_value_returns_same_placeholder(self):
        """Test that adding the same value returns the same placeholder."""
        pmap = PlaceholderMap()
        p1 = pmap.add("john@example.com", "EMAIL")
        p2 = pmap.add("john@example.com", "EMAIL")

        assert p1 == p2
        assert len(pmap) == 1

    def test_add_different_values_increments_counter(self):
        """Test that different values get incremented placeholders."""
        pmap = PlaceholderMap()
        p1 = pmap.add("john@example.com", "EMAIL")
        p2 = pmap.add("jane@example.com", "EMAIL")

        assert p1 == "[EMAIL_1]"
        assert p2 == "[EMAIL_2]"

    def test_add_different_entity_types(self):
        """Test that different entity types have separate counters."""
        pmap = PlaceholderMap()
        email_p = pmap.add("john@example.com", "EMAIL")
        phone_p = pmap.add("555-123-4567", "PHONE_US")

        assert email_p == "[EMAIL_1]"
        assert phone_p == "[PHONE_US_1]"

    def test_get_original(self):
        """Test retrieving original value from placeholder."""
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")

        original = pmap.get_original("[EMAIL_1]")
        assert original == "john@example.com"

    def test_get_original_not_found(self):
        """Test get_original returns None for unknown placeholder."""
        pmap = PlaceholderMap()
        assert pmap.get_original("[UNKNOWN_1]") is None

    def test_get_placeholder(self):
        """Test retrieving placeholder from original value."""
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")

        placeholder = pmap.get_placeholder("john@example.com")
        assert placeholder == "[EMAIL_1]"

    def test_restore_in_string(self):
        """Test restoring placeholders in a string."""
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")
        pmap.add("555-123-4567", "PHONE_US")

        text = "Contact [EMAIL_1] or call [PHONE_US_1]"
        restored = pmap.restore_in_string(text)

        assert restored == "Contact john@example.com or call 555-123-4567"

    def test_len_and_bool(self):
        """Test __len__ and __bool__ methods."""
        pmap = PlaceholderMap()
        assert len(pmap) == 0
        assert not pmap

        pmap.add("test@example.com", "EMAIL")
        assert len(pmap) == 1
        assert pmap


class TestPIIRedactor:
    """Test PIIRedactor functionality."""

    def test_init_default(self):
        """Test default initialization."""
        redactor = PIIRedactor()
        assert "EMAIL" in redactor.entities
        assert "CREDIT_CARD" in redactor.entities

    def test_init_custom_entities(self):
        """Test initialization with custom entities."""
        redactor = PIIRedactor(entities=["EMAIL", "PHONE_US"])
        assert redactor.entities == ["EMAIL", "PHONE_US"]

    def test_redact_email(self):
        """Test email redaction."""
        redactor = PIIRedactor()
        text = "Contact john@example.com for details."
        redacted, pmap = redactor.redact(text)

        assert "john@example.com" not in redacted
        assert "[EMAIL_1]" in redacted
        assert pmap.get_original("[EMAIL_1]") == "john@example.com"

    def test_redact_phone_us(self):
        """Test US phone number redaction."""
        redactor = PIIRedactor()
        text = "Call us at (555) 123-4567 today."
        redacted, pmap = redactor.redact(text)

        assert "(555) 123-4567" not in redacted
        assert "[PHONE_US_1]" in redacted

    def test_redact_phone_with_country_code(self):
        """Test phone with +1 country code."""
        redactor = PIIRedactor()
        text = "Call +1-555-123-4567 for support."
        redacted, pmap = redactor.redact(text)

        assert "+1-555-123-4567" not in redacted
        assert len(pmap) >= 1

    def test_redact_multiple_emails(self):
        """Test redacting multiple emails."""
        redactor = PIIRedactor()
        text = "Email john@example.com or support@company.org."
        redacted, pmap = redactor.redact(text)

        assert "john@example.com" not in redacted
        assert "support@company.org" not in redacted
        assert len(pmap) == 2

    def test_redact_no_pii(self):
        """Test redacting text with no PII."""
        redactor = PIIRedactor()
        text = "This is a simple invoice for services."
        redacted, pmap = redactor.redact(text)

        assert redacted == text
        assert len(pmap) == 0

    def test_redact_credit_card(self):
        """Test credit card number redaction."""
        redactor = PIIRedactor()
        text = "Card number: 4111-1111-1111-1111"
        redacted, pmap = redactor.redact(text)

        assert "4111-1111-1111-1111" not in redacted
        assert "[CREDIT_CARD_1]" in redacted

    def test_redact_ssn(self):
        """Test SSN redaction."""
        redactor = PIIRedactor()
        text = "SSN: 123-45-6789"
        redacted, pmap = redactor.redact(text)

        assert "123-45-6789" not in redacted
        assert "[SSN_1]" in redacted

    def test_redact_ip_address(self):
        """Test IP address redaction."""
        redactor = PIIRedactor()
        text = "Server IP: 192.168.1.100"
        redacted, pmap = redactor.redact(text)

        assert "192.168.1.100" not in redacted
        assert "[IP_ADDRESS_1]" in redacted

    def test_restore_dict(self):
        """Test restoring placeholders in a dict."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")
        pmap.add("(555) 123-4567", "PHONE_US")

        data = {"email": "[EMAIL_1]", "phone": "[PHONE_US_1]"}
        restored = redactor.restore(data, pmap)

        assert restored["email"] == "john@example.com"
        assert restored["phone"] == "(555) 123-4567"

    def test_restore_nested_dict(self):
        """Test restoring placeholders in nested dict."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")

        data = {"contact": {"email": "[EMAIL_1]", "name": "John"}}
        restored = redactor.restore(data, pmap)

        assert restored["contact"]["email"] == "john@example.com"
        assert restored["contact"]["name"] == "John"

    def test_restore_list(self):
        """Test restoring placeholders in a list."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")
        pmap.add("jane@example.com", "EMAIL")

        data = ["[EMAIL_1]", "[EMAIL_2]"]
        restored = redactor.restore(data, pmap)

        assert restored == ["john@example.com", "jane@example.com"]

    def test_restore_mixed_structure(self):
        """Test restoring in complex nested structure."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()
        pmap.add("john@example.com", "EMAIL")
        pmap.add("(555) 123-4567", "PHONE_US")

        data = {
            "customer": {
                "email": "[EMAIL_1]",
                "phones": ["[PHONE_US_1]"],
            },
            "items": [{"name": "Product", "price": 100}],
        }
        restored = redactor.restore(data, pmap)

        assert restored["customer"]["email"] == "john@example.com"
        assert restored["customer"]["phones"] == ["(555) 123-4567"]
        assert restored["items"][0]["name"] == "Product"

    def test_restore_empty_map(self):
        """Test restore with empty placeholder map."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()

        data = {"email": "test@example.com"}
        restored = redactor.restore(data, pmap)

        assert restored == data

    def test_restore_preserves_primitives(self):
        """Test that restore preserves non-string primitives."""
        redactor = PIIRedactor()
        pmap = PlaceholderMap()

        data = {"count": 42, "active": True, "value": None, "price": 99.99}
        restored = redactor.restore(data, pmap)

        assert restored["count"] == 42
        assert restored["active"] is True
        assert restored["value"] is None
        assert restored["price"] == 99.99

    def test_get_supported_entities(self):
        """Test getting supported entity types."""
        redactor = PIIRedactor()
        entities = redactor.get_supported_entities()

        assert isinstance(entities, list)
        assert "EMAIL" in entities
        assert "PHONE_US" in entities
        assert "CREDIT_CARD" in entities

    def test_custom_patterns(self):
        """Test adding custom regex patterns."""
        import re

        custom = {"CUSTOM_ID": re.compile(r"ID-\d{6}")}
        redactor = PIIRedactor(custom_patterns=custom)

        text = "Your reference: ID-123456"
        redacted, pmap = redactor.redact(text)

        assert "ID-123456" not in redacted
        assert "[CUSTOM_ID_1]" in redacted

    def test_backwards_compatible_params(self):
        """Test that presidio-style params are accepted but ignored."""
        # Should not raise
        redactor = PIIRedactor(language="en", score_threshold=0.5)
        assert redactor is not None


class TestFullRedactionCycle:
    """Test complete redaction and restoration cycle."""

    def test_full_cycle_invoice(self):
        """Test full redaction cycle with invoice-like text."""
        redactor = PIIRedactor()

        original_text = """
        INVOICE #12345

        Bill To:
        Email: john.smith@example.com
        Phone: (555) 123-4567

        Amount Due: $1,500.00
        """

        # Redact
        redacted_text, pmap = redactor.redact(original_text)

        # Verify redaction
        assert "john.smith@example.com" not in redacted_text
        assert "(555) 123-4567" not in redacted_text
        assert len(pmap) == 2

        # Simulate LLM extraction result
        extracted_data = {
            "invoice_number": "12345",
            "customer_email": "[EMAIL_1]",
            "customer_phone": "[PHONE_US_1]",
            "amount": 1500.00,
        }

        # Restore
        restored_data = redactor.restore(extracted_data, pmap)

        assert restored_data["customer_email"] == "john.smith@example.com"
        assert restored_data["customer_phone"] == "(555) 123-4567"
        assert restored_data["amount"] == 1500.00

    def test_redaction_preserves_structure(self):
        """Test that redaction preserves document structure."""
        redactor = PIIRedactor()

        original = "Contact: john@test.com | Phone: (555) 000-1234"
        redacted, _ = redactor.redact(original)

        # Structure should be preserved
        assert "Contact:" in redacted
        assert "|" in redacted
        assert "Phone:" in redacted
