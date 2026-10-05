"""
Embedding generation module.

This module converts incident descriptions into dense vector embeddings
using a pre-trained Sentence Transformer model.

Model Used:
    all-MiniLM-L6-v2

Output:
    List of 384-dimensional embeddings
"""

from typing import List, Optional, Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.config.settings import settings
from app.utils.logger import logger


from app.interfaces.embedding_interface import EmbeddingInterface

class SemanticEmbeddingGenerator(EmbeddingInterface):
    """
    Generates semantic embeddings for incidents.
    """

    def __init__(self) -> None:
        """
        Load the Sentence Transformer model only once.

        Loading the model is expensive, so we initialize it
        during object creation and reuse it.
        """
        try:
            logger.info("Loading Sentence Transformer model locally...")
            
            model_path_str = settings.EMBEDDING_MODEL_PATH
            model_path = Path(model_path_str).resolve()
            
            if not model_path.exists():
                raise RuntimeError(f"Embedding model not found at: {model_path}")
                
            if not model_path.is_dir():
                raise RuntimeError(f"Embedding model path is not a directory: {model_path}")
                
            required_files = ["config.json", "model.safetensors"]
            for req_file in required_files:
                if not (model_path / req_file).exists():
                    raise RuntimeError(f"Required model file missing: {req_file} in {model_path}")

            self.model = SentenceTransformer(
                str(model_path),
                local_files_only=settings.EMBEDDING_MODEL_LOCAL_ONLY
            )
            
            # Validate embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            if test_embedding.shape[1] != 384:
                raise RuntimeError(f"Invalid embedding dimension. Expected 384, got {test_embedding.shape[1]}")

            logger.info("Embedding model loaded successfully from %s", model_path)

        except Exception as e:
            logger.exception("Failed to load embedding model.")
            raise RuntimeError(f"Failed to load local embedding model: {str(e)}") from e

    def generate_embeddings(
        self,
        texts: List[str],
        check_cancellation: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of incident texts.

        Args:
            texts:
                List containing incident descriptions.
            check_cancellation:
                Optional callable to cooperatively abort the operation.

        Returns:
            NumPy array of embedding vectors.
        """

        try:

            total = len(texts)
            batch_size = settings.EMBEDDING_BATCH_SIZE

            logger.info(
                "Generating embeddings for %s incidents in batches of %s...",
                total, batch_size
            )

            all_embeddings = []

            for start_idx in range(0, total, batch_size):
                if check_cancellation:
                    check_cancellation()

                batch = texts[start_idx:start_idx + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)

            if check_cancellation:
                check_cancellation()

            logger.info("Embedding generation completed.")

            return np.vstack(all_embeddings) if all_embeddings else np.array([])

        except Exception as e:
            if type(e).__name__ == "JobCancelledException":
                logger.info("Embedding generation was cancelled.")
                raise

            logger.exception(
                "Embedding generation failed."
            )

            raise
