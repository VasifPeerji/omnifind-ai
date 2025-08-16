import json
from pathlib import Path
from ..config import DATA_DIR
from ..logger import get_logger

logger = get_logger("data.loader")


def load_sample_catalog(filename='sample_products.json'):
    p = Path(DATA_DIR) / filename
    if not p.exists():
        logger.warning(f"Sample catalog not found at {p}")
        return []
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} products from {p}")
    return data
