import os
from pathlib import Path


try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"
DATA_DIR = BASE / "data"
MANIFEST = DATA_DIR / "weaviate_manifest.json"


CHAT_MODEL =os.getenv("CHAT_MODEL", "gemma4:12b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
NUM_CTX = 8192


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COLLECTION = "DocChunks"


CHUNK_SIZE = 800
OVERLAP = 190
EMBED_BATCH = 64
TOP_K = 9
ALPHA = 0.9
