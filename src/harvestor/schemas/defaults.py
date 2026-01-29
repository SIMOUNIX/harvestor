"""
Default Pydantic schemas for common document types.

These schemas serve as examples and can be used directly or as
templates for custom schemas.

Only supported default types for now: invoice & receipt.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """Individual line item in an invoice or receipt."""

    name: str = Field(description="Item name")
    amount: Optional[float] = Field(None, description="Total amount for this item")
    quantity: Optional[float] = Field(None, description="Quantity of items")

    unit_price_with_taxes: Optional[float] = Field(
        None, description="Price per unit with taxes"
    )
    unit_price_without_taxes: Optional[float] = Field(
        None, description="Price per unit without taxes"
    )

    discount: Optional[float] = Field(None, description="Discount amount per item")
    discount_percentage: Optional[float] = Field(
        None, description="Discount percentage per item"
    )

    taxes: Optional[float] = Field(None, description="Taxes amount per item")
    taxes_percentage: Optional[float] = Field(
        None, description="Taxes percentage per item"
    )


class InvoiceData(BaseModel):
    """
    Standard invoice extraction schema.

    Covers common invoice fields found in most business invoices.
    """

    invoice_number: Optional[str] = Field(
        None, description="The unique invoice number or ID"
    )
    date: Optional[str] = Field(None, description="Invoice issue date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    po_number: Optional[str] = Field(
        None, description="Purchase order number if referenced"
    )

    vendor_name: Optional[str] = Field(None, description="Vendor/seller company name")
    vendor_address: Optional[str] = Field(None, description="Vendor address")
    vendor_email: Optional[str] = Field(None, description="Vendor email address")
    vendor_tax_id: Optional[str] = Field(
        None, description="Vendor tax ID (GSTIN, VAT, EIN, etc.)"
    )

    customer_name: Optional[str] = Field(
        None, description="Customer/buyer company or person name"
    )
    customer_address: Optional[str] = Field(
        None, description="Customer billing address"
    )
    customer_email: Optional[str] = Field(None, description="Customer email address")
    customer_phone: Optional[str] = Field(None, description="Customer phone number")

    line_items: Optional[List[LineItem]] = Field(
        None, description="List of invoice line items"
    )
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    tax_amount: Optional[float] = Field(None, description="Total tax amount")
    discount: Optional[float] = Field(None, description="Discount amount if any")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    currency: Optional[str] = Field(None, description="Currency code (USD, EUR, etc.)")

    bank_name: Optional[str] = Field(None, description="Bank name for payment")
    bank_account: Optional[str] = Field(None, description="Bank account number")
    bank_routing: Optional[str] = Field(None, description="Bank routing/SWIFT/BIC code")

    notes: Optional[str] = Field(None, description="Special notes or payment terms")


class ReceiptData(BaseModel):
    """Standard receipt extraction schema."""

    merchant_name: Optional[str] = Field(None, description="Store or merchant name")
    merchant_address: Optional[str] = Field(None, description="Store address")
    date: Optional[str] = Field(None, description="Transaction date")
    time: Optional[str] = Field(None, description="Transaction time")
    items: Optional[List[LineItem]] = Field(None, description="List of purchased items")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    tax: Optional[float] = Field(None, description="Tax amount")
    total: Optional[float] = Field(None, description="Total amount paid")
    payment_method: Optional[str] = Field(None, description="Payment method used")
    card_last_four: Optional[str] = Field(
        None, description="Last 4 digits of card if applicable"
    )
