"""Vector store manager using ChromaDB for semantic search."""

import logging
from pathlib import Path
from typing import Any

import chromadb
import torch
from chromadb.config import Settings
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ingrid.llm.base import BaseLLMProvider, LLMError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector embeddings and semantic search with ChromaDB."""

    def __init__(
        self,
        chroma_path: Path,
        embedding_provider: BaseLLMProvider,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        similarity_metric: str = "cosine",
    ):
        """Initialize vector store manager.

        Args:
            chroma_path: Path to ChromaDB storage directory
            embedding_provider: LLM provider for text embeddings
            clip_model_name: HuggingFace CLIP model name for image embeddings
            similarity_metric: Distance metric (cosine, l2, ip)
        """
        self.chroma_path = chroma_path
        self.embedding_provider = embedding_provider
        self.clip_model_name = clip_model_name
        self.similarity_metric = similarity_metric

        # Ensure storage directory exists
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChromaDB storage directory: {chroma_path}")

        # Verify directory is writable
        if not chroma_path.exists():
            raise RuntimeError(f"Failed to create ChromaDB directory: {chroma_path}")
        if not chroma_path.is_dir():
            raise RuntimeError(f"ChromaDB path exists but is not a directory: {chroma_path}")

        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB PersistentClient...")
        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialized successfully")

        # Initialize CLIP model for image embeddings
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Set device (prefer MPS on Mac, CUDA on GPU, fallback to CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.clip_model.to(self.device)
        logger.info(f"CLIP model loaded on device: {self.device}")

        # Initialize collections
        self.text_collection = self._get_or_create_collection(
            name="ingrid_cleaned_text",
            metadata={"description": "Full cleaned text embeddings"},
        )

        self.summary_collection = self._get_or_create_collection(
            name="ingrid_summaries",
            metadata={"description": "Original language summaries"},
        )

        self.summary_en_collection = self._get_or_create_collection(
            name="ingrid_summaries_english",
            metadata={"description": "English summary translations"},
        )

        self.image_collection = self._get_or_create_collection(
            name="ingrid_images",
            metadata={"description": "CLIP image embeddings"},
        )

        # Log collection counts for verification
        collection_stats = self.get_collection_stats()
        logger.info(
            f"VectorStoreManager initialized: {chroma_path}, "
            f"similarity={similarity_metric}, device={self.device}"
        )
        logger.info(
            f"Collection counts: "
            f"text={collection_stats['ingrid_cleaned_text']}, "
            f"summaries={collection_stats['ingrid_summaries']}, "
            f"summaries_en={collection_stats['ingrid_summaries_english']}, "
            f"images={collection_stats['ingrid_images']}"
        )

    def _get_or_create_collection(self, name: str, metadata: dict[str, Any]) -> chromadb.Collection:
        """Get existing collection or create new one.

        Args:
            name: Collection name
            metadata: Collection metadata

        Returns:
            ChromaDB collection object
        """
        try:
            collection = self.client.get_collection(name=name)
            logger.debug(f"Found existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata,
                # ChromaDB will use cosine similarity by default
            )
            logger.info(f"Created new collection: {name}")

        return collection

    def add_text_embeddings(
        self,
        doc_id: str,
        cleaned_text: str,
        summary: str,
        summary_english: str,
        metadata: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Generate and store text embeddings in three collections.

        Args:
            doc_id: Document UUID
            cleaned_text: Full cleaned text
            summary: Original language summary
            summary_english: English translation of summary
            metadata: Document metadata

        Returns:
            Tuple of (text_emb_id, summary_emb_id, summary_en_emb_id)

        Raises:
            LLMError: If embedding generation fails
        """
        try:
            # Generate embeddings for cleaned text
            logger.debug(f"Generating text embedding for doc {doc_id}")
            text_embedding = self._generate_text_embedding(cleaned_text)
            text_emb_id = f"{doc_id}_text"

            self.text_collection.add(
                ids=[text_emb_id],
                embeddings=[text_embedding],
                documents=[cleaned_text[:1000]],  # Store preview (first 1000 chars)
                metadatas=[metadata],
            )

            # Generate embeddings for summary (if available)
            summary_emb_id = ""
            if summary:
                logger.debug(f"Generating summary embedding for doc {doc_id}")
                summary_embedding = self._generate_text_embedding(summary)
                summary_emb_id = f"{doc_id}_summary"

                self.summary_collection.add(
                    ids=[summary_emb_id],
                    embeddings=[summary_embedding],
                    documents=[summary],
                    metadatas=[metadata],
                )

            # Generate embeddings for English summary (if available)
            summary_en_emb_id = ""
            if summary_english:
                logger.debug(f"Generating English summary embedding for doc {doc_id}")
                summary_en_embedding = self._generate_text_embedding(summary_english)
                summary_en_emb_id = f"{doc_id}_summary_en"

                self.summary_en_collection.add(
                    ids=[summary_en_emb_id],
                    embeddings=[summary_en_embedding],
                    documents=[summary_english],
                    metadatas=[metadata],
                )

            logger.info(
                f"Text embeddings stored for doc {doc_id}: "
                f"text={text_emb_id}, summary={summary_emb_id}, "
                f"summary_en={summary_en_emb_id}"
            )

            return text_emb_id, summary_emb_id, summary_en_emb_id

        except LLMError as e:
            logger.error(f"Failed to generate text embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding text embeddings: {e}", exc_info=True)
            raise

    def add_image_embedding(self, doc_id: str, image_path: Path, metadata: dict[str, Any]) -> str:
        """Generate CLIP embedding for image and store.

        Args:
            doc_id: Document UUID
            image_path: Path to image file
            metadata: Document metadata

        Returns:
            Image embedding ID

        Raises:
            Exception: If image processing or embedding fails
        """
        try:
            logger.debug(f"Generating CLIP embedding for {image_path.name}")

            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Generate CLIP embedding
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embedding
                image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding_list = image_embedding.cpu().numpy().flatten().tolist()

            # Store in ChromaDB
            image_emb_id = f"{doc_id}_image"
            self.image_collection.add(
                ids=[image_emb_id],
                embeddings=[embedding_list],
                documents=[str(image_path)],  # Store path as document
                metadatas=[metadata],
            )

            logger.info(
                f"Image embedding stored for doc {doc_id}: {image_emb_id} "
                f"(dim={len(embedding_list)})"
            )

            return image_emb_id

        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}", exc_info=True)
            raise

    def search_text(
        self, query: str, collection_name: str = "ingrid_cleaned_text", top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Semantic search across text collections.

        Args:
            query: Search query
            collection_name: Collection to search (without "ingrid_" prefix)
            top_k: Number of results to return

        Returns:
            List of result dictionaries with keys:
            - id: Document ID
            - score: Similarity score (distance)
            - document: Document text
            - metadata: Document metadata

        Raises:
            LLMError: If query embedding generation fails
        """
        try:
            # Map collection name to full name
            full_collection_name = (
                collection_name
                if collection_name.startswith("ingrid_")
                else f"ingrid_{collection_name}"
            )

            # Get collection
            if full_collection_name == "ingrid_cleaned_text":
                collection = self.text_collection
            elif full_collection_name == "ingrid_summaries":
                collection = self.summary_collection
            elif full_collection_name == "ingrid_summaries_english":
                collection = self.summary_en_collection
            else:
                raise ValueError(f"Unknown collection: {collection_name}")

            logger.debug(f"Searching {full_collection_name} for: '{query}'")

            # Generate query embedding
            query_embedding = self._generate_text_embedding(query)

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "score": (
                                1 - results["distances"][0][i]
                            ),  # Convert distance to similarity
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                        }
                    )

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except LLMError as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def search_by_image(self, image_path: Path, top_k: int = 10) -> list[dict[str, Any]]:
        """Visual similarity search using CLIP.

        Args:
            image_path: Path to query image
            top_k: Number of results to return

        Returns:
            List of result dictionaries (same format as search_text)

        Raises:
            Exception: If image processing fails
        """
        try:
            logger.debug(f"Image similarity search for: {image_path.name}")

            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Generate CLIP embedding for query image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                query_embedding = image_embedding.cpu().numpy().flatten().tolist()

            # Search in image collection
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "score": 1 - results["distances"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                        }
                    )

            logger.info(f"Image search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Image search failed: {e}", exc_info=True)
            raise

    def get_collection_stats(self) -> dict[str, int]:
        """Get statistics for all collections.

        Returns:
            Dictionary mapping collection names to document counts
        """
        stats = {
            "ingrid_cleaned_text": self.text_collection.count(),
            "ingrid_summaries": self.summary_collection.count(),
            "ingrid_summaries_english": self.summary_en_collection.count(),
            "ingrid_images": self.image_collection.count(),
        }

        logger.debug(f"Collection stats: {stats}")
        return stats

    def delete_embeddings(self, doc_id: str) -> dict[str, bool]:
        """Delete all embeddings for a document from all collections.

        Args:
            doc_id: Document UUID to delete

        Returns:
            Dictionary mapping collection names to deletion success status
        """
        results = {}
        collections = [
            ("ingrid_cleaned_text", self.text_collection),
            ("ingrid_summaries", self.summary_collection),
            ("ingrid_summaries_english", self.summary_en_collection),
            ("ingrid_images", self.image_collection),
        ]

        for name, collection in collections:
            try:
                # Check if document exists in collection
                existing = collection.get(ids=[doc_id])
                if existing and existing.get("ids"):
                    collection.delete(ids=[doc_id])
                    results[name] = True
                    logger.debug(f"Deleted embedding for {doc_id} from {name}")
                else:
                    # Document not in this collection, still success
                    results[name] = True
                    logger.debug(f"Document {doc_id} not found in {name}")
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to delete {doc_id} from {name}: {e}")

        deleted_count = sum(1 for success in results.values() if success)
        logger.info(
            f"Deleted embeddings for {doc_id}: {deleted_count}/{len(collections)} collections"
        )

        return results

    def delete_embeddings_batch(self, doc_ids: list[str]) -> dict[str, int]:
        """Delete embeddings for multiple documents.

        Args:
            doc_ids: List of document UUIDs to delete

        Returns:
            Dictionary with 'deleted' and 'failed' counts
        """
        deleted = 0
        failed = 0

        for doc_id in doc_ids:
            results = self.delete_embeddings(doc_id)
            # Consider success if at least cleaned_text was successful
            if results.get("ingrid_cleaned_text", False):
                deleted += 1
            else:
                failed += 1

        logger.info(f"Batch delete complete: {deleted} deleted, {failed} failed")
        return {"deleted": deleted, "failed": failed}

    def verify_storage(self) -> tuple[bool, list[str]]:
        """Verify vector store health and persistence.

        Returns:
            Tuple of (success, errors_list)
        """
        errors = []

        try:
            # Check storage directory exists
            if not self.chroma_path.exists():
                errors.append(f"ChromaDB directory does not exist: {self.chroma_path}")
                return False, errors

            # Check collections are accessible
            collections_to_check = [
                ("ingrid_cleaned_text", self.text_collection),
                ("ingrid_summaries", self.summary_collection),
                ("ingrid_summaries_english", self.summary_en_collection),
                ("ingrid_images", self.image_collection),
            ]

            for name, collection in collections_to_check:
                try:
                    count = collection.count()
                    logger.debug(f"Collection {name} verified: {count} documents")
                except Exception as e:
                    errors.append(f"Collection {name} not accessible: {e}")

            # Check CLIP model is loaded
            if self.clip_model is None:
                errors.append("CLIP model not loaded")

            if errors:
                return False, errors

            logger.info("Vector store verification passed")
            return True, []

        except Exception as e:
            errors.append(f"Verification failed: {e}")
            return False, errors

    def _generate_text_embedding(self, text: str, max_chars: int = 28000) -> list[float]:
        """Generate embedding for text using embedding provider.

        Args:
            text: Text to embed
            max_chars: Maximum characters to embed (default 28000, safe for 8192 tokens)
                      nomic-embed-text supports 8192 tokens (~4 chars/token = ~32k chars)
                      We use 28000 to leave margin for safety.

        Returns:
            Embedding vector as list of floats

        Raises:
            LLMError: If embedding generation fails
        """
        # Truncate long text to avoid context length errors
        if len(text) > max_chars:
            logger.warning(
                f"Text too long for embedding ({len(text)} chars), truncating to {max_chars} chars"
            )
            text = text[:max_chars]

        try:
            response = self.embedding_provider.embed(text)
            if isinstance(response, list):
                # Provider returned list of embeddings
                return response[0] if response else []
            else:
                # Provider returned EmbeddingResponse object
                return response.embedding
        except LLMError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise LLMError(f"Embedding error: {e}")
