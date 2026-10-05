import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer

from app.clustering.embedding import SemanticEmbeddingGenerator
from app.config.settings import settings

def test_local_model_path_resolution(tmp_path):
    """Test that embedding generator correctly resolves the model path."""
    with patch("app.config.settings.settings.EMBEDDING_MODEL_PATH", str(tmp_path)):
        # Since it's an empty temp dir, initialization should fail
        with pytest.raises(RuntimeError, match="Required model file missing"):
            SemanticEmbeddingGenerator()

def test_local_model_invalid_path():
    """Test missing model path error."""
    with patch("app.config.settings.settings.EMBEDDING_MODEL_PATH", "/does/not/exist/ever/123"):
        with pytest.raises(RuntimeError, match="Embedding model not found"):
            SemanticEmbeddingGenerator()

def test_local_model_not_a_directory(tmp_path):
    """Test model path is not a directory."""
    test_file = tmp_path / "not_a_dir.txt"
    test_file.touch()
    
    with patch("app.config.settings.settings.EMBEDDING_MODEL_PATH", str(test_file)):
        with pytest.raises(RuntimeError, match="Embedding model path is not a directory"):
            SemanticEmbeddingGenerator()

def test_local_model_missing_files(tmp_path):
    """Test missing required model files."""
    # Create empty directory
    with patch("app.config.settings.settings.EMBEDDING_MODEL_PATH", str(tmp_path)):
        with pytest.raises(RuntimeError, match="Required model file missing"):
            SemanticEmbeddingGenerator()

@patch("app.clustering.embedding.SentenceTransformer")
def test_local_files_only_behavior(mock_transformer):
    """Test local_files_only=True behavior."""
    # Mock to avoid loading real model during this specific check
    mock_instance = MagicMock()
    # Mock encode to return a correct-dimension embedding
    mock_instance.encode.return_value = np.zeros((1, 384))
    mock_transformer.return_value = mock_instance
    
    # We must point to a real valid directory so the pre-checks pass
    # Since we are mocking SentenceTransformer, we can use the default valid path
    
    # Run the generator
    generator = SemanticEmbeddingGenerator()
    
    # Verify SentenceTransformer was called with local_files_only
    mock_transformer.assert_called_once()
    kwargs = mock_transformer.call_args.kwargs
    assert kwargs.get("local_files_only") == True

def test_real_local_model_loading():
    """Test loading the real local model (requires no mock)."""
    # This uses the real setting which should point to the valid backend/app/3r models/models/all-MiniLM-L6-v2/
    generator = SemanticEmbeddingGenerator()
    assert generator.model is not None
    
    # Generate embedding
    texts = ["Test text"]
    embeddings = generator.generate_embeddings(texts)
    
    # Embedding generation
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    # Embedding dimension = 384
    assert embeddings.shape[1] == 384

def test_generate_embeddings_cancellation():
    """Test existing cancellation behavior remains unchanged."""
    generator = SemanticEmbeddingGenerator()
    texts = ["One", "Two", "Three", "Four"]
    
    # Batch size artificially small
    with patch("app.config.settings.settings.EMBEDDING_BATCH_SIZE", 2):
        state = {"calls": 0}
        def check_cancellation():
            state["calls"] += 1
            if state["calls"] >= 2:
                from app.core.exceptions import JobCancelledException
                raise JobCancelledException("Cancelled")
                
        with pytest.raises(Exception) as exc_info:
            generator.generate_embeddings(texts, check_cancellation=check_cancellation)
            
        assert "Cancelled" in str(exc_info.value)
