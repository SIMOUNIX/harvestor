"""
Main Harvester class for document data extraction.

This is the primary public API for Harvestor.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.cost_tracker import cost_tracker
from ..parsers.llm_parser import InvoiceData, LLMParser
from ..schemas.base import ExtractionStrategy, HarvestResult


class Harvester:
    """
    Main document extraction class.

    Features:
    - Extract structured data from documents
    - Multiple extraction strategies (LLM, OCR, layout analysis)
    - Cost optimization (free methods first, LLM fallback)
    - Batch processing support
    - Progress tracking and reporting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        cost_limit_per_doc: float = 0.10,
        daily_cost_limit: Optional[float] = None
    ):
        """
        Initialize Harvester.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: LLM model to use (default: Claude Haiku for cost optimization)
            cost_limit_per_doc: Maximum cost per document (default: $0.10)
            daily_cost_limit: Optional daily cost limit
        """
        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        self.model = model

        # Set cost limits
        cost_tracker.set_limits(
            daily_limit=daily_cost_limit,
            per_document_limit=cost_limit_per_doc
        )

        # Initialize LLM parser
        self.llm_parser = LLMParser(
            model=model,
            api_key=self.api_key
        )

    def harvest_text(
        self,
        text: str,
        doc_type: str = "invoice",
        document_id: Optional[str] = None,
        language: str = "en"
    ) -> HarvestResult:
        """
        Extract structured data from text.

        This is the simplest method - just LLM extraction on provided text.

        Args:
            text: Document text to extract from
            doc_type: Type of document (invoice, receipt, etc.)
            document_id: Unique identifier for this document
            language: Document language (for future use)

        Returns:
            HarvestResult with extracted data and metadata
        """
        import time
        from datetime import datetime

        start_time = time.time()

        # Generate document ID if not provided
        if not document_id:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract using LLM
        extraction_result = self.llm_parser.extract(
            text=text,
            doc_type=doc_type,
            document_id=document_id
        )

        total_time = time.time() - start_time

        # Build harvest result
        return HarvestResult(
            success=extraction_result.success,
            document_id=document_id,
            document_type=doc_type,
            data=extraction_result.data,
            extraction_results=[extraction_result],
            final_strategy=extraction_result.strategy,
            final_confidence=extraction_result.confidence,
            total_cost=extraction_result.cost,
            cost_breakdown={
                extraction_result.strategy.value: extraction_result.cost
            },
            total_time=total_time,
            error=extraction_result.error,
            language=language
        )

    def harvest_file(
        self,
        file_path: Union[str, Path],
        doc_type: str = "invoice",
        document_id: Optional[str] = None,
        language: str = "en"
    ) -> HarvestResult:
        """
        Extract structured data from a file.

        Currently supports:
        - Text files (.txt)
        - PDF files (.pdf) - extracts text first

        Future: Will implement smart extraction cascade (native PDF → layout → OCR → LLM)

        Args:
            file_path: Path to document file
            doc_type: Type of document
            document_id: Unique identifier
            language: Document language

        Returns:
            HarvestResult with extracted data
        """
        import time

        start_time = time.time()
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            return HarvestResult(
                success=False,
                document_id=document_id or file_path.stem,
                document_type=doc_type,
                data={},
                error=f"File not found: {file_path}",
                file_path=str(file_path),
                total_time=time.time() - start_time
            )

        # Get file info
        file_size = file_path.stat().st_size
        document_id = document_id or file_path.stem

        # Extract text based on file type
        try:
            text = self._extract_text_from_file(file_path)
        except Exception as e:
            return HarvestResult(
                success=False,
                document_id=document_id,
                document_type=doc_type,
                data={},
                error=f"Failed to extract text: {str(e)}",
                file_path=str(file_path),
                file_size_bytes=file_size,
                total_time=time.time() - start_time
            )

        # Use harvest_text for actual extraction
        result = self.harvest_text(
            text=text,
            doc_type=doc_type,
            document_id=document_id,
            language=language
        )

        # Add file metadata
        result.file_path = str(file_path)
        result.file_size_bytes = file_size

        return result

    def _extract_text_from_file(self, file_path: Path) -> str:
        """
        Extract text from file based on type.

        Args:
            file_path: Path to file

        Returns:
            Extracted text

        Raises:
            ValueError: If file type is not supported
        """
        suffix = file_path.suffix.lower()

        if suffix == '.txt':
            # Plain text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif suffix == '.pdf':
            # PDF file - use pdfplumber for native text extraction
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                if not text_parts:
                    raise ValueError("No text found in PDF (might need OCR)")

                return "\n\n".join(text_parts)

            except ImportError:
                raise ValueError("pdfplumber not installed. Install with: pip install pdfplumber")

        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .pdf")

    def harvest_batch(
        self,
        files: List[Union[str, Path]],
        doc_type: str = "invoice",
        show_progress: bool = True
    ) -> List[HarvestResult]:
        """
        Process multiple documents.

        Args:
            files: List of file paths to process
            doc_type: Document type for all files
            show_progress: Show progress bar

        Returns:
            List of HarvestResult objects
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(files, desc="Processing documents")
            except ImportError:
                iterator = files
        else:
            iterator = files

        for file_path in iterator:
            result = self.harvest_file(
                file_path=file_path,
                doc_type=doc_type
            )
            results.append(result)

        return results

    def print_summary(self):
        """Print cost summary."""
        cost_tracker.print_summary()


def harvest(
    file_path: Union[str, Path],
    doc_type: str = "invoice",
    language: str = "en",
    model: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None
) -> HarvestResult:
    """
    One-liner function for quick extraction.

    Example:
        ```python
        from harvestor import harvest

        result = harvest("invoice.pdf", doc_type="invoice")
        print(f"Invoice #: {result.data.get('invoice_number')}")
        print(f"Total: ${result.data.get('total_amount')}")
        print(f"Cost: ${result.total_cost:.4f}")
        ```

    Args:
        file_path: Path to document
        doc_type: Document type
        language: Document language
        model: LLM model to use
        api_key: API key (uses env var if not provided)

    Returns:
        HarvestResult with extracted data
    """
    harvester = Harvester(api_key=api_key, model=model)
    return harvester.harvest_file(
        file_path=file_path,
        doc_type=doc_type,
        language=language
    )
