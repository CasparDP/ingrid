"""Database manager for document storage with SQLite + SQLAlchemy."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, or_, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from ingrid.storage.models import Base, Document

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for document storage."""

    def __init__(self, database_path: Path, journal_mode: str = "WAL"):
        """Initialize database manager.

        Args:
            database_path: Path to SQLite database file
            journal_mode: SQLite journal mode (WAL or DELETE)
        """
        self.database_path = database_path
        self.journal_mode = journal_mode

        # Ensure parent directory exists
        database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite-specific settings
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            echo=False,
            connect_args={
                "timeout": 30,  # 30 second timeout
                "check_same_thread": False,  # Allow multithreading
            },
        )

        # Set WAL mode if specified
        if journal_mode == "WAL":
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        logger.info(f"Database initialized at {database_path} (mode: {journal_mode})")

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def create_document(self, doc_data: dict[str, Any]) -> Document:
        """Create a new document record.

        Args:
            doc_data: Dictionary containing document fields

        Returns:
            Created Document object

        Raises:
            Exception: If document creation fails
        """
        with self._get_session() as session:
            document = Document(**doc_data)
            session.add(document)
            session.commit()
            session.refresh(document)
            logger.info(f"Created document: {document.id} ({document.filename})")
            return document

    def get_document(self, doc_id: str) -> Document | None:
        """Get document by ID.

        Args:
            doc_id: Document UUID

        Returns:
            Document object or None if not found
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if document:
                # Detach from session before returning
                session.expunge(document)
            return document

    def get_document_by_filename(self, filename: str) -> Document | None:
        """Get document by filename.

        Args:
            filename: Original filename

        Returns:
            Document object or None if not found
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.filename == filename).first()
            if document:
                session.expunge(document)
            return document

    def list_documents(
        self,
        doc_type: str | None = None,
        content_type: str | None = None,
        flagged_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents with optional filters.

        Args:
            doc_type: Filter by document type (letter, newspaper_article, other)
            content_type: Filter by content type (handwritten, typed, mixed)
            flagged_only: Show only flagged documents
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of Document objects
        """
        with self._get_session() as session:
            query = session.query(Document)

            # Apply filters
            if doc_type:
                query = query.filter(Document.doc_type == doc_type)
            if content_type:
                query = query.filter(Document.content_type == content_type)
            if flagged_only:
                query = query.filter(Document.flagged_for_review == True)

            # Order by creation date (newest first)
            query = query.order_by(Document.created_at.desc())

            # Apply pagination
            query = query.limit(limit).offset(offset)

            # Execute and detach results
            documents = query.all()
            for doc in documents:
                session.expunge(doc)

            return documents

    def update_document(self, doc_id: str, updates: dict[str, Any]) -> Document | None:
        """Update document fields.

        Args:
            doc_id: Document UUID
            updates: Dictionary of fields to update

        Returns:
            Updated Document object or None if not found
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(document, key):
                    setattr(document, key, value)

            session.commit()
            session.refresh(document)
            session.expunge(document)

            logger.info(f"Updated document: {doc_id}")
            return document

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the database.

        Args:
            doc_id: Document UUID

        Returns:
            True if deleted, False if document not found
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return False

            session.delete(document)
            session.commit()

            logger.info(f"Deleted document: {doc_id}")
            return True

    def add_tag(self, doc_id: str, tag: str) -> bool:
        """Add a manual tag to document.

        Args:
            doc_id: Document UUID
            tag: Tag to add

        Returns:
            True if successful, False if document not found
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return False

            # Initialize tags list if None
            if document.manual_tags is None:
                document.manual_tags = []

            # Add tag if not already present
            if tag not in document.manual_tags:
                document.manual_tags.append(tag)
                # Mark the JSON field as modified so SQLAlchemy knows to update it
                flag_modified(document, "manual_tags")
                session.commit()
                logger.info(f"Added tag '{tag}' to document {doc_id}")

            return True

    def remove_tag(self, doc_id: str, tag: str) -> bool:
        """Remove a manual tag from document.

        Args:
            doc_id: Document UUID
            tag: Tag to remove

        Returns:
            True if successful, False if document not found or tag not present
        """
        with self._get_session() as session:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document or not document.manual_tags:
                return False

            if tag in document.manual_tags:
                document.manual_tags.remove(tag)
                # Mark the JSON field as modified so SQLAlchemy knows to update it
                flag_modified(document, "manual_tags")
                session.commit()
                logger.info(f"Removed tag '{tag}' from document {doc_id}")
                return True

            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics:
            - total: Total document count
            - by_type: Counts by doc_type
            - by_content_type: Counts by content_type
            - by_language: Counts by language
            - flagged_count: Number of flagged documents
            - avg_extraction_time: Average extraction time
            - avg_classification_time: Average classification time
            - avg_processing_time: Average processing time
            - avg_storage_time: Average storage time
        """
        with self._get_session() as session:
            # Total documents
            total = session.query(func.count(Document.id)).scalar()

            # Counts by doc_type
            by_type = {}
            type_counts = (
                session.query(Document.doc_type, func.count(Document.id))
                .group_by(Document.doc_type)
                .all()
            )
            for doc_type, count in type_counts:
                by_type[doc_type] = count

            # Counts by content_type
            by_content_type = {}
            content_counts = (
                session.query(Document.content_type, func.count(Document.id))
                .group_by(Document.content_type)
                .all()
            )
            for content_type, count in content_counts:
                by_content_type[content_type] = count

            # Counts by language (documents can have multiple languages)
            by_language = defaultdict(int)
            documents = session.query(Document.languages).all()
            for (languages,) in documents:
                if languages:
                    for lang in languages:
                        by_language[lang] += 1

            # Flagged documents
            flagged_count = (
                session.query(func.count(Document.id))
                .filter(Document.flagged_for_review == True)
                .scalar()
            )

            # Average processing times
            avg_times = (
                session.query(
                    func.avg(Document.extraction_time),
                    func.avg(Document.classification_time),
                    func.avg(Document.processing_time),
                    func.avg(Document.storage_time),
                )
                .filter(Document.extraction_time.isnot(None))  # Only documents with times
                .first()
            )

            return {
                "total": total or 0,
                "by_type": by_type,
                "by_content_type": by_content_type,
                "by_language": dict(by_language),
                "flagged_count": flagged_count or 0,
                "avg_extraction_time": round(avg_times[0], 1) if avg_times[0] else 0,
                "avg_classification_time": (round(avg_times[1], 1) if avg_times[1] else 0),
                "avg_processing_time": round(avg_times[2], 1) if avg_times[2] else 0,
                "avg_storage_time": round(avg_times[3], 1) if avg_times[3] else 0,
            }

    def search_documents(self, query: str, limit: int = 10) -> list[Document]:
        """Simple text search across document fields.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching Document objects

        Note:
            This is a basic SQL LIKE search. For semantic search, use ChromaDB.
        """
        with self._get_session() as session:
            search_pattern = f"%{query}%"
            documents = (
                session.query(Document)
                .filter(
                    or_(
                        Document.cleaned_text.like(search_pattern),
                        Document.summary.like(search_pattern),
                        Document.summary_english.like(search_pattern),
                        Document.sender.like(search_pattern),
                        Document.recipient.like(search_pattern),
                        Document.location.like(search_pattern),
                    )
                )
                .limit(limit)
                .all()
            )

            for doc in documents:
                session.expunge(doc)

            return documents

    def verify_database(self) -> tuple[bool, list[str]]:
        """Verify database health and schema.

        Returns:
            Tuple of (success, errors_list)
        """
        errors = []

        try:
            # Check database file exists
            if not self.database_path.exists():
                errors.append(f"Database file does not exist: {self.database_path}")
                return False, errors

            # Try to connect and query
            with self._get_session() as session:
                try:
                    # Check if documents table exists and is queryable
                    count = session.query(func.count(Document.id)).scalar()
                    logger.debug(f"Database verified: {count} documents")
                except Exception as e:
                    errors.append(f"Database query failed: {e}")
                    return False, errors

            # Check table schema
            from sqlalchemy import inspect

            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            if "documents" not in tables:
                errors.append("Documents table does not exist")

            if errors:
                return False, errors

            logger.info("Database verification passed")
            return True, []

        except Exception as e:
            errors.append(f"Verification failed: {e}")
            return False, errors

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")
