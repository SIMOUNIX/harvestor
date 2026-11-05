from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from harvestor import Harvestor, list_models


load_dotenv()


class SimpleInvoiceSchema(BaseModel):
    """
    Implement the schema you want as output. Customize for each document type.
    """

    vendor: Optional[str] = Field(None, description="The vendor name")
    customer_firstname: Optional[str] = Field(
        None, description="The customer firstname"
    )
    customer_lastname: Optional[str] = Field(None, description="The customer lastname")
    invoice_total_price_with_taxes: Optional[float] = Field(
        None, description="The total price with taxes"
    )
    invoice_total_price_without_taxes: Optional[float] = Field(
        None, description="The total price without taxes"
    )


# List available models
print("Available models:", list(list_models().keys()))

# Use default model (claude-haiku)
h = Harvestor(model="claude-haiku", validate=True)

output = h.harvest_file(
    source="data/uploads/keep_for_test.jpg", schema=SimpleInvoiceSchema
)

print(output.to_summary())
print(output.validation)
# Alternative: use OpenAI
# h_openai = Harvestor(model="gpt-4o-mini")
# output = h_openai.harvest_file("data/uploads/keep_for_test.jpg", schema=SimpleInvoiceSchema)

# Alternative: use local Ollama (free) or cloud Ollama
# h_ollama = Harvestor(model="gemma3:4b-cloud")
# output = h_ollama.harvest_file(
#     "data/uploads/keep_for_test.jpg", schema=SimpleInvoiceSchema
# )
# print(output.to_summary())
