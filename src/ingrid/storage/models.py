"""SQLAlchemy database models for document storage."""

from datetime import datetime
from typing import Any
import json
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class JSONType(TypeDecorator):
    """Type decorator for JSON columns in SQLite."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        """Convert Python object to JSON string for storage."""
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: str | None, dialect: Any) -> Any:
        """Convert JSON string to Python object on retrieval."""
        if value is None:
            return None
        return json.loads(value)


class Document(Base):
    """Main document table storing all metadata and references."""

    __tablename__ = "documents"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # File information
    filename = Column(String(255), nullable=False, index=True)
    filepath = Column(String(512), nullable=False)
    markdown_path = Column(String(512), nullable=True)

    # Classification
    doc_type = Column(
        String(50), nullable=False, index=True
    )  # letter, newspaper_article, other
    content_type = Column(
        String(50), nullable=False, index=True
    )  # handwritten, typed, mixed
    languages = Column(JSONType, nullable=False)  # List of language codes

    # Content
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # Original language summary
    summary_english = Column(Text, nullable=True)  # English translation
    summary_language = Column(String(10), nullable=True)  # Language code of summary

    # Metadata (extracted by LLM)
    date = Column(String(50), nullable=True, index=True)  # Flexible date format
    sender = Column(String(255), nullable=True, index=True)
    recipient = Column(String(255), nullable=True, index=True)
    location = Column(String(255), nullable=True, index=True)
    topics = Column(JSONType, nullable=True)  # List of topic strings
    people_mentioned = Column(JSONType, nullable=True)  # List of names
    organizations_mentioned = Column(JSONType, nullable=True)  # List of organizations

    # Confidence scores (JSON dict with field-level scores)
    confidence_scores = Column(JSONType, nullable=True)
    # Expected structure:
    # {
    #   "doc_type": 0.0-1.0,
    #   "content_type": 0.0-1.0,
    #   "date": 0.0-1.0,
    #   "sender": 0.0-1.0,
    #   "recipient": 0.0-1.0,
    #   "location": 0.0-1.0
    # }

    # Embedding references (ChromaDB IDs)
    text_embedding_id = Column(String(100), nullable=True)
    summary_embedding_id = Column(String(100), nullable=True)
    summary_english_embedding_id = Column(String(100), nullable=True)
    image_embedding_id = Column(String(100), nullable=True)

    # Embedded images (list of paths to extracted images)
    embedded_images = Column(JSONType, nullable=True)  # List of image paths/metadata

    # Processing status flags
    extraction_success = Column(Boolean, nullable=False, default=False)
    classification_success = Column(Boolean, nullable=False, default=False)
    processing_success = Column(Boolean, nullable=False, default=False)
    storage_success = Column(Boolean, nullable=False, default=False)
    flagged_for_review = Column(Boolean, nullable=False, default=False, index=True)

    # Processing times (seconds)
    extraction_time = Column(Float, nullable=True)
    classification_time = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    storage_time = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Manual overrides and tags
    manual_tags = Column(JSONType, nullable=True, default=list)  # List of tag strings
    manual_notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of document."""
        return (
            f"<Document(id={self.id[:8]}..., "
            f"filename={self.filename}, "
            f"doc_type={self.doc_type}, "
            f"content_type={self.content_type})>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "filepath": self.filepath,
            "markdown_path": self.markdown_path,
            "doc_type": self.doc_type,
            "content_type": self.content_type,
            "languages": self.languages,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "summary": self.summary,
            "summary_english": self.summary_english,
            "summary_language": self.summary_language,
            "date": self.date,
            "sender": self.sender,
            "recipient": self.recipient,
            "location": self.location,
            "topics": self.topics,
            "people_mentioned": self.people_mentioned,
            "organizations_mentioned": self.organizations_mentioned,
            "confidence_scores": self.confidence_scores,
            "text_embedding_id": self.text_embedding_id,
            "summary_embedding_id": self.summary_embedding_id,
            "summary_english_embedding_id": self.summary_english_embedding_id,
            "image_embedding_id": self.image_embedding_id,
            "embedded_images": self.embedded_images,
            "extraction_success": self.extraction_success,
            "classification_success": self.classification_success,
            "processing_success": self.processing_success,
            "storage_success": self.storage_success,
            "flagged_for_review": self.flagged_for_review,
            "extraction_time": self.extraction_time,
            "classification_time": self.classification_time,
            "processing_time": self.processing_time,
            "storage_time": self.storage_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "manual_tags": self.manual_tags,
            "manual_notes": self.manual_notes,
        }
