import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.environ.get("OMNIFIND_DATA_DIR", str(BASE_DIR / "data"))
DEFAULT_EMBEDDING = os.environ.get("OMNIFIND_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
