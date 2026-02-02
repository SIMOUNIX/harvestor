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


# List available models
print("Available models:", list(list_models().keys()))

# Use default model (claude-haiku)
h = Harvestor(model="claude-haiku")

output = h.harvest_file(
    source="data/uploads/keep_for_test.jpg", schema=SimpleInvoiceSchema
)

print(output.to_summary())

# Alternative: use OpenAI
# h_openai = Harvestor(model="gpt-4o-mini")
# output = h_openai.harvest_file("invoice.jpg", schema=SimpleInvoiceSchema)

# Alternative: use local Ollama (free)
# h_ollama = Harvestor(model="llava")
# output = h_ollama.harvest_file("invoice.jpg", schema=SimpleInvoiceSchema)
