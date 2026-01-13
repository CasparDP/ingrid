"""Storage module for document persistence and semantic search.

This module orchestrates database operations (SQLite) and vector store operations
(ChromaDB) for document storage, indexing, and retrieval.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.base import BaseLLMProvider, LLMError
from ..processing.models import ProcessingResult

from .database import DatabaseManager
from .vectorstore import VectorStoreManager

# Public exports
__all__ = [
    "StorageOrchestrator",
    "DatabaseManager",
    "VectorStoreManager",
]

logger = logging.getLogger(__name__)


class StorageOrchestrator:
    """Orchestrates document storage in database and vector store.

    The orchestrator:
    1. Generates UUID for document
    2. Generates and stores text embeddings (cleaned text, summaries)
    3. Generates and stores image embedding (CLIP)
    4. Creates database record with all metadata and embedding references
    5. Returns document ID and storage status

    Attributes:
        config: Configuration object
        db: DatabaseManager instance
        vectorstore: VectorStoreManager instance
    """

    def __init__(self, config: Config, embedding_provider: BaseLLMProvider):
        """Initialize storage orchestrator.

        Args:
            config: Configuration object with storage settings
            embedding_provider: LLM provider for text embeddings
        """
        self.config = config

        # Initialize database
        self.db = DatabaseManager(
            database_path=config.storage.database_path,
            journal_mode=(
                config.database.journal_mode if hasattr(config, "database") else "WAL"
            ),
        )

        # Initialize vector store
        self.vectorstore = VectorStoreManager(
            chroma_path=config.storage.chroma_path,
            embedding_provider=embedding_provider,
            clip_model_name="openai/clip-vit-base-patch32",
            similarity_metric=(
                config.chromadb.similarity_metric
                if hasattr(config, "chromadb")
                else "cosine"
            ),
        )

        logger.info("StorageOrchestrator initialized")

    def store_document(
        self, processing_result: ProcessingResult, image_path: Path
    ) -> tuple[str, float, bool, str | None]:
        """Store document in database and generate/store embeddings.

        Args:
            processing_result: Processing result from Phase 4
            image_path: Path to original scan image

        Returns:
            Tuple of (doc_id, storage_time, success, error_message)
        """
        start_time = time.time()
        doc_id = ""

        try:
            # Validate processing result
            if not processing_result.success:
                error = "Cannot store: processing failed"
                logger.error(error)
                return "", time.time() - start_time, False, error

            # Generate UUID for document
            doc_id = str(uuid.uuid4())
            logger.info(f"Storing document {doc_id}: {image_path.name}")

            # Build metadata for vector store
            metadata = self._build_vectorstore_metadata(processing_result, doc_id)

            # Step 1: Generate and store text embeddings
            logger.debug("Generating text embeddings")
            text_emb_id, summary_emb_id, summary_en_emb_id = (
                self.vectorstore.add_text_embeddings(
                    doc_id=doc_id,
                    cleaned_text=processing_result.cleaned_text,
                    summary=processing_result.summary,
                    summary_english=processing_result.summary_english,
                    metadata=metadata,
                )
            )

            # Step 2: Generate and store image embedding
            logger.debug("Generating image embedding")
            image_emb_id = self.vectorstore.add_image_embedding(
                doc_id=doc_id, image_path=image_path, metadata=metadata
            )

            # Step 3: Build database record
            doc_data = self._build_document_data(
                doc_id=doc_id,
                processing_result=processing_result,
                image_path=image_path,
                text_emb_id=text_emb_id,
                summary_emb_id=summary_emb_id,
                summary_en_emb_id=summary_en_emb_id,
                image_emb_id=image_emb_id,
            )

            # Step 4: Store in database
            logger.debug("Storing document in database")
            document = self.db.create_document(doc_data)

            storage_time = time.time() - start_time
            logger.info(f"Document {doc_id} stored successfully in {storage_time:.1f}s")

            return doc_id, storage_time, True, None

        except LLMError as e:
            storage_time = time.time() - start_time
            error = f"Embedding generation failed: {str(e)}"
            logger.error(error)
            return doc_id, storage_time, False, error

        except Exception as e:
            storage_time = time.time() - start_time
            error = f"Storage failed: {str(e)}"
            logger.error(error, exc_info=True)
            return doc_id, storage_time, False, error

    def _build_vectorstore_metadata(
        self, processing_result: ProcessingResult, doc_id: str
    ) -> dict[str, Any]:
        """Build metadata dict for ChromaDB storage.

        Args:
            processing_result: Processing result
            doc_id: Document UUID

        Returns:
            Metadata dictionary
        """
        classification = processing_result.classification_job.primary_result
        metadata_obj = processing_result.metadata

        metadata = {
            "doc_id": doc_id,
            "filename": processing_result.image_path.name,
        }

        if classification:
            metadata["doc_type"] = classification.doc_type.value
            metadata["content_type"] = classification.content_type.value
            metadata["languages"] = ",".join(classification.languages)

        if metadata_obj:
            if metadata_obj.date:
                metadata["date"] = metadata_obj.date
            if metadata_obj.sender:
                metadata["sender"] = metadata_obj.sender
            if metadata_obj.recipient:
                metadata["recipient"] = metadata_obj.recipient
            if metadata_obj.location:
                metadata["location"] = metadata_obj.location

        return metadata

    def _build_document_data(
        self,
        doc_id: str,
        processing_result: ProcessingResult,
        image_path: Path,
        text_emb_id: str,
        summary_emb_id: str,
        summary_en_emb_id: str,
        image_emb_id: str,
    ) -> dict[str, Any]:
        """Build document data dictionary for database storage.

        Args:
            doc_id: Document UUID
            processing_result: Processing result
            image_path: Original image path
            text_emb_id: Text embedding ID
            summary_emb_id: Summary embedding ID
            summary_en_emb_id: English summary embedding ID
            image_emb_id: Image embedding ID

        Returns:
            Document data dictionary
        """
        extraction = processing_result.extraction_job
        classification = processing_result.classification_job
        primary_extraction = extraction.primary_result
        primary_classification = classification.primary_result
        metadata_obj = processing_result.metadata

        # Build confidence scores dict
        confidence_scores = {}
        if primary_classification:
            confidence_scores.update(primary_classification.confidence_scores)
        if metadata_obj:
            confidence_scores.update(
                {
                    "date": metadata_obj.date_confidence,
                    "sender": metadata_obj.sender_confidence,
                    "recipient": metadata_obj.recipient_confidence,
                    "location": metadata_obj.location_confidence,
                }
            )

        # Build document data
        doc_data = {
            "id": doc_id,
            "filename": image_path.name,
            "filepath": str(image_path),
            "markdown_path": str(processing_result.markdown_path)
            if processing_result.markdown_path
            else None,
            # Classification
            "doc_type": (
                primary_classification.doc_type.value
                if primary_classification
                else "other"
            ),
            "content_type": (
                primary_classification.content_type.value
                if primary_classification
                else "unknown"
            ),
            "languages": (
                primary_classification.languages if primary_classification else []
            ),
            # Content
            "raw_text": primary_extraction.text if primary_extraction else "",
            "cleaned_text": processing_result.cleaned_text,
            "summary": processing_result.summary,
            "summary_english": processing_result.summary_english,
            "summary_language": processing_result.summary_language,
            # Metadata
            "date": metadata_obj.date if metadata_obj else None,
            "sender": metadata_obj.sender if metadata_obj else None,
            "recipient": metadata_obj.recipient if metadata_obj else None,
            "location": metadata_obj.location if metadata_obj else None,
            "topics": metadata_obj.topics if metadata_obj else [],
            "people_mentioned": (
                metadata_obj.people_mentioned if metadata_obj else []
            ),
            "organizations_mentioned": (
                metadata_obj.organizations_mentioned if metadata_obj else []
            ),
            # Confidence scores
            "confidence_scores": confidence_scores,
            # Embedding references
            "text_embedding_id": text_emb_id,
            "summary_embedding_id": summary_emb_id,
            "summary_english_embedding_id": summary_en_emb_id,
            "image_embedding_id": image_emb_id,
            # Embedded images (placeholder for future)
            "embedded_images": [],
            # Processing status
            "extraction_success": extraction.success,
            "classification_success": classification.success,
            "processing_success": processing_result.success,
            "storage_success": True,  # Will be set by database
            "flagged_for_review": classification.flagged_for_review,
            # Processing times
            "extraction_time": extraction.total_processing_time,
            "classification_time": classification.total_processing_time,
            "processing_time": processing_result.total_processing_time,
            "storage_time": 0.0,  # Will be updated
            # Manual overrides
            "manual_tags": [],
            "manual_notes": None,
        }

        return doc_data
