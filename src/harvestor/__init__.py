"""
🌾 Harvestor - Harvest intelligence from any document

Extract structured data from any document with AI-powered extraction.
"""

__version__ = "0.1.0"

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("Harvestor requires Python 3.10 or higher")

# Will implement these later
# from .core.harvester import Harvester, harvest
# from .core.pipeline import Pipeline

__all__ = ["__version__"]
