"""
Base storage abstraction for extraction results.

Defines the interface that all storage backends (SQLite, PostgreSQL, MySQL, etc.)
must implement. This file is pure Python — no SQLAlchemy or other DB-specific imports.

Key concepts:
- BaseStorage is an Abstract Base Class (ABC). Any concrete backend (e.g. SQLAlchemyStorage)
  must subclass it and implement every @abstractmethod. If they forget one, Python raises
  TypeError at instantiation time.
- EntityProfile and FraudContext are plain dataclasses that hold query results.
  They have no DB dependencies so they work with any backend.
- DEFAULT_FIELD_MAPPINGS maps generic DB column names to schema-specific field names.
  This is what makes fraud detection work across invoices, receipts, or any custom schema.

Usage pattern:
    from harvestor.storage import get_storage

    storage = get_storage(url="sqlite:///my.db")
    storage.store_harvest(result)
    ctx = storage.build_fraud_context(
        document_number="INV-001",
        entity_name="Acme Corp",
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# TYPE_CHECKING is False at runtime, True when a type checker (pyright, mypy) analyzes
# the code. This lets us import HarvestResult for type hints without creating a circular
# import at runtime (since schemas.base doesn't need to know about storage).
if TYPE_CHECKING:
    from harvestor.schemas.base import HarvestResult


# ---------------------------------------------------------------------------
# Dataclasses for fraud query results
# ---------------------------------------------------------------------------
# These are plain data containers — no logic, no DB dependency.
# They use @dataclass so you get __init__, __repr__, __eq__ for free.
# For mutable defaults (lists), always use field(default_factory=list),
# never = [] — otherwise all instances would share the SAME list object.
# ---------------------------------------------------------------------------


@dataclass
class EntityProfile:
    """
    Aggregated history for a single entity (vendor, merchant, counterparty, etc.).

    Built from all past documents involving this entity. Used by the fraud agent
    to understand what's "normal" for this entity.

    Attributes:
        entity_name:          The entity's name as extracted from documents.
        total_documents:      How many documents we've seen from this entity.
        total_amount:         Sum of all total_amount values across documents.
        avg_amount:           Mean total_amount across documents.
        stddev_amount:        Standard deviation of total_amount. A new document
                              with an amount far from avg (high z-score) is suspicious.
        min_amount:           Smallest total_amount seen.
        max_amount:           Largest total_amount seen.
        known_bank_accounts:  All distinct bank account numbers seen for this entity.
                              A change here is a major fraud signal.
        known_bank_routings:  All distinct routing/SWIFT codes seen.
        known_entity_ids:     All distinct tax/business IDs seen. Should usually be 1.
        first_seen:           Timestamp of the earliest document from this entity.
        last_seen:            Timestamp of the most recent document.
    """

    entity_name: str
    total_documents: int
    total_amount: float
    avg_amount: float
    stddev_amount: float
    min_amount: float
    max_amount: float
    known_bank_accounts: List[str] = field(default_factory=list)
    known_bank_routings: List[str] = field(default_factory=list)
    known_entity_ids: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class FraudContext:
    """
    Pre-built fraud context for a single document.

    This is the bundle of information an LLM agent receives before deciding
    whether a document is fraudulent. Each field maps to a specific fraud signal:

    Attributes:
        entity_profile:       Full history of the entity (see EntityProfile).
                              None if this is the first document from this entity.

        duplicate_documents:  List of past documents with the same document_number
                              AND entity_name. Any match = potential duplicate fraud.

        bank_detail_changes:  Historical bank details for this entity. Each entry has
                              bank_account, bank_routing, first_seen, last_seen, count.
                              A new bank account appearing is a key fraud indicator.

        amount_zscore:        How many standard deviations the current document's amount
                              is from this entity's historical mean. None if we can't
                              calculate it (first document, or zero stddev).
                              |zscore| > 2-3 = suspicious.

        entity_id_conflicts:  Other entity names using the same tax/business ID.
                              Should normally be empty or contain only name variations.
                              Different companies sharing an ID = suspicious.
    """

    entity_profile: Optional[EntityProfile] = None
    duplicate_documents: List[Dict[str, Any]] = field(default_factory=list)
    bank_detail_changes: List[Dict[str, Any]] = field(default_factory=list)
    amount_zscore: Optional[float] = None
    entity_id_conflicts: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------
# ABC = Abstract Base Class. You can't instantiate BaseStorage directly.
# You must subclass it and implement every @abstractmethod.
#
# Why an ABC instead of just a regular class?
# - It enforces the contract at instantiation time, not at call time.
#   If SQLAlchemyStorage forgets to implement store_harvest(), you get
#   TypeError("Can't instantiate abstract class") immediately — not a
#   cryptic AttributeError later when someone tries to use it.
# - It makes the interface explicit and discoverable.
#
# Note: register_field_mapping() is NOT abstract — it's the same logic
# for every backend (just updating a dict), so we implement it once here.
# ---------------------------------------------------------------------------


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.

    All storage backends must subclass this and implement every @abstractmethod.
    The backend is responsible for persisting HarvestResults and answering
    fraud-context queries.

    Non-abstract methods:
        register_field_mapping() — same for all backends, implemented here.

    Subclass contract:
        - initialize() must be idempotent (safe to call multiple times)
        - store_harvest() must use self.field_mappings to populate generic columns
        - All query methods return plain dicts (not ORM objects) to stay backend-agnostic
        - build_fraud_context() should call the individual fraud query methods
    """

    def __init__(self) -> None:
        # Copy the default mappings so each storage instance has its own dict.
        # dict() creates a shallow copy — good enough since the inner dicts
        # are only read, never mutated in place.
        self.field_mappings: Dict[str, Dict[str, Optional[str]]] = dict()

    def register_field_mapping(
        self, doc_type: str, mapping: Dict[str, Optional[str]]
    ) -> None:
        """
        Register a field mapping for a custom document type.

        This tells the storage layer how to extract generic fields (entity_name,
        total_amount, etc.) from the extracted data dict of your custom schema.

        Args:
            doc_type:  The document type string (must match what Harvestor.get_doc_type_from_schema
                       returns for your schema, e.g. "contract" for ContractData).
            mapping:   Dict mapping generic column names to your schema's field names.
                       Use None for fields that don't apply to your document type.

        Example:
            storage.register_field_mapping("contract", {
                "entity_name": "counterparty_name",
                "entity_id": "counterparty_tax_id",
                "document_number": "contract_number",
                "total_amount": "contract_value",
                "currency": "currency",
                "bank_account": None,
                "bank_routing": None,
            })
        """
        self.field_mappings[doc_type] = mapping

    # --- Lifecycle -----------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the storage backend (create tables, indexes, etc.).

        Must be idempotent — calling it twice should not fail or duplicate anything.
        For SQL backends this means CREATE TABLE IF NOT EXISTS / checkfirst=True.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close connections and release resources.

        Called when the storage is no longer needed. After close(), the storage
        instance should not be used.
        """
        pass

    # --- Write Operations ----------------------------------------------------

    @abstractmethod
    def store_harvest(self, result: "HarvestResult") -> str:
        """
        Persist a HarvestResult and its nested ExtractionResults/ValidationResult.

        The implementation must:
        1. Look up self.field_mappings[result.document_type] to find the mapping
        2. Use the mapping to extract generic field values from result.data
           e.g. entity_name = result.data.get(mapping.get("entity_name"))
        3. Store both the full data (as JSON) and the denormalized generic fields

        Args:
            result: The HarvestResult from a harvest operation.
                    (imported from harvestor.schemas.base at runtime)

        Returns:
            The storage ID (primary key) of the stored record, as a string.
        """
        pass

    # --- Read Operations -----------------------------------------------------

    @abstractmethod
    def get_harvest(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored harvest result by its document_id.

        Args:
            document_id: The business key (e.g. "doc_20240101_120000" or filename stem).

        Returns:
            Dict representation of the harvest, or None if not found.
        """
        pass

    @abstractmethod
    def list_harvests(
        self,
        document_type: Optional[str] = None,
        entity_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List stored harvests with optional filters.

        All parameters are optional — calling list_harvests() with no args
        returns the most recent 100 harvests.

        Args:
            document_type: Filter by type ("invoice", "receipt", etc.)
            entity_name:   Filter by entity (vendor, merchant, etc.)
            since:         Only return harvests after this timestamp
            limit:         Max number of results (default 100)
            offset:        Skip this many results (for pagination)

        Returns:
            List of dict representations, ordered by most recent first.
        """
        pass

    # --- Fraud Context Queries -----------------------------------------------
    # These methods power the fraud detection agent. Each one answers a specific
    # fraud-related question about the current document.
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_entity_profile(self, entity_name: str) -> Optional[EntityProfile]:
        """
        Get aggregated profile for an entity across all their past documents.

        Used to establish what's "normal" for this entity — average amounts,
        known bank accounts, typical frequency, etc.

        Args:
            entity_name: The entity name to look up.

        Returns:
            EntityProfile with aggregated stats, or None if entity has no history.
        """
        pass

    @abstractmethod
    def find_duplicate_documents(
        self, document_number: str, entity_name: str
    ) -> List[Dict[str, Any]]:
        """
        Find documents with the same document_number AND entity_name.

        A match here means the same entity submitted the same document number
        more than once — a common fraud pattern (duplicate invoice submission).

        Args:
            document_number: The document reference (invoice #, receipt #, etc.)
            entity_name:     The entity who issued it.

        Returns:
            List of matching harvest dicts. Empty list = no duplicates found.
        """
        pass

    @abstractmethod
    def get_bank_detail_history(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all bank account details ever used by this entity.

        Each entry should include:
        - bank_account: the account number
        - bank_routing: the routing/SWIFT code
        - first_seen: when this bank detail first appeared
        - last_seen: when it was last used
        - count: how many documents used these details

        A new bank account appearing for a known entity is a top fraud signal
        (business email compromise / payment redirection attack).

        Args:
            entity_name: The entity to look up.

        Returns:
            List of dicts with bank detail history, ordered by first_seen.
        """
        pass

    @abstractmethod
    def find_entity_id_conflicts(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Find different entity names using the same entity_id (tax ID, business reg, etc.).

        Each entry should include:
        - entity_name: the name associated with this ID
        - document_count: how many documents from this name
        - first_seen / last_seen: time range

        Normally an entity_id maps to exactly one entity. Multiple different names
        sharing the same ID suggests either name variations (benign) or identity
        fraud (malicious).

        Args:
            entity_id: The tax/business ID to check.

        Returns:
            List of dicts, one per distinct entity_name using this ID.
        """
        pass

    @abstractmethod
    def build_fraud_context(
        self,
        document_number: str,
        entity_name: str,
        entity_id: Optional[str] = None,
        total_amount: Optional[float] = None,
    ) -> FraudContext:
        """
        Build a complete FraudContext for a single document.

        This is the main method the fraud agent calls. It orchestrates all the
        individual fraud queries into one FraudContext object.

        A typical implementation calls:
        1. get_entity_profile(entity_name)
        2. find_duplicate_documents(document_number, entity_name)
        3. get_bank_detail_history(entity_name)
        4. Calculate amount z-score from profile stats + total_amount
        5. find_entity_id_conflicts(entity_id) if entity_id is provided

        Args:
            document_number: The document reference to check for duplicates.
            entity_name:     The counterparty name.
            entity_id:       Their tax/business ID (optional — some doc types don't have this).
            total_amount:    The document's total amount (optional — used for z-score).

        Returns:
            FraudContext with all fraud signals populated.
        """
        pass

    # --- Metadata ------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_backend_name(cls) -> str:
        """
        Get the backend identifier string.

        Used by the registry/factory to map names to classes.
        e.g. "sqlalchemy", "mongodb", "dynamodb"

        Returns:
            Backend name string.
        """
        pass
