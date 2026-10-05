import hashlib
import json
from pathlib import Path

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_manifest():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parent / "models"
    model_dir = models_dir / "all-MiniLM-L6-v2"
    manifest_path = script_dir.parent / "MODEL_MANIFEST.json"

    if not model_dir.exists():
        print(f"Error: Model directory not found at {model_dir}")
        return

    print(f"Calculating hashes for model files in {model_dir}...")
    hashes = {}
    for filepath in model_dir.iterdir():
        if filepath.is_file():
            hashes[filepath.name] = calculate_sha256(filepath)

    manifest = {
        "model_name": "all-MiniLM-L6-v2",
        "source_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "framework": "SentenceTransformers / PyTorch",
        "local_model_path": "models/all-MiniLM-L6-v2",
        "verification_status": "verified",
        "hashes": hashes
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    generate_manifest()
