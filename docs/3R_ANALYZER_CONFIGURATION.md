# Configuration

Configuration is exclusively environment-driven via `pydantic-settings`.
No passwords or secrets are hardcoded in the source code.

## Environment Variables
* `DATABASE_URL`: Connection string for PostgreSQL (e.g., `postgresql://user:pass@localhost:5432/threer`).
* `CORS_ORIGINS`: Comma-separated list of allowed UI domains.
* `SIMILARITY_THRESHOLD`: Cosine similarity cutoff for clustering (Default: 0.85).
* `BATCH_SIZE`: AI processing batch size (Default: 500).
* `EMBEDDING_MODEL_PATH`: Path to the local SentenceTransformers model directory. (Default: `backend/app/3r models/models/all-MiniLM-L6-v2`)
* `EMBEDDING_MODEL_LOCAL_ONLY`: If `True`, strictly enforces that no external downloads from Hugging Face occur at runtime. (Default: `True`)

A `.env.example` file is provided in the repository.
