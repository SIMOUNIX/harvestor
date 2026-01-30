from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from harvestor import Harvestor  # , harvest
import os

load_dotenv()


class SimpleInoviceModelSchema(BaseModel):
    """
    Implement the schema you want as output. Customise for each document types.
    """

    vendor: Optional[str] = Field(None, description="The vendor name")
    customer_firstname: Optional[str] = Field(
        None, description="The customer firstname"
    )
    customer_lastname: Optional[str] = Field(None, description="The customer lastname")


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

h = Harvestor(api_key=ANTHROPIC_API_KEY, model="Claude Haiku 3")

output = h.harvest_file(
    source="data/uploads/keep_for_test.jpg", schema=SimpleInoviceModelSchema
)

print(output.to_summary())

# output_2 = harvest("data/uploads/keep_for_test.jpg", schema=SimpleInoviceModelSchema)

# print(output_2.to_summary())
