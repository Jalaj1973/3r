import json
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_model():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    MODEL_PATH = base_dir / "models" / "all-MiniLM-L6-v2"
    MANIFEST_PATH = base_dir / "MODEL_MANIFEST.json"
    
    print(f"Verifying model at: {MODEL_PATH}")

    # 1. Resolve & Validate path
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Model path is not a directory: {MODEL_PATH}")

    # 2. Validate required files
    required_files = ["config.json", "model.safetensors"]
    for req in required_files:
        req_path = MODEL_PATH / req
        if not req_path.exists():
            raise RuntimeError(f"Required file missing: {req} in {MODEL_PATH}")

    # 3. Validate SHA256 hashes
    if MANIFEST_PATH.exists():
        print("Manifest found. Validating hashes...")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        expected_hashes = manifest.get("hashes", {})
        for filename, expected_hash in expected_hashes.items():
            filepath = MODEL_PATH / filename
            if filepath.exists():
                actual_hash = calculate_sha256(filepath)
                if actual_hash != expected_hash:
                    raise RuntimeError(f"Hash mismatch for {filename}. Expected {expected_hash}, got {actual_hash}")
            else:
                print(f"Warning: File {filename} in manifest is missing from model directory.")
        print("Hashes verified.")
    else:
        print("Warning: No MODEL_MANIFEST.json found, skipping hash validation.")

    # 4. Load model offline
    print("Loading model (local_files_only=True)...")
    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )
    print("Model loaded successfully.")

    # 5 & 6. Generate embeddings & Verify dimension
    texts = [
        "Database connection timeout",
        "Database connectivity failure"
    ]
    print("Generating test embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    if dim != 384:
        raise RuntimeError(f"Invalid embedding dimension. Expected 384, got {dim}")
        
    print(f"Embedding generation successful. Dimension: {dim}")

    # 7. Print success
    print("\nModel verification completed successfully! The local model is ready for offline use.")

if __name__ == "__main__":
    verify_model()