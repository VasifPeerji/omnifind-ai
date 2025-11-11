import json
import pandas as pd
from pathlib import Path
from ..config import DATA_DIR
from ..logger import get_logger

logger = get_logger("data.loader")


def load_sample_catalog(filename='sample_products.json'):
    """Old sample loader (JSON)"""
    p = Path(DATA_DIR) / filename
    if not p.exists():
        logger.warning(f"Sample catalog not found at {p}")
        return []
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} products from {p}")
    return data


def load_fashion_catalog(filename="processed/fashion_products.pkl"):
    """New real dataset loader (from PKL or CSV)"""
    p = Path(DATA_DIR) / filename
    if not p.exists():
        logger.error(f"Fashion dataset not found at {p}")
        return pd.DataFrame()

    # Load dataframe
    df = pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_csv(p)
    logger.info(f"✅ Loaded fashion catalog → {len(df)} rows from {p}")
    return df
