import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../../../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRODUCTS_FILE = RAW_DIR / "amazon_products.csv"
CATEGORIES_FILE = RAW_DIR / "amazon_categories.csv"
OUTPUT_CSV = PROCESSED_DIR / "fashion_products.csv"
OUTPUT_PKL = PROCESSED_DIR / "fashion_products.pkl"

# -----------------------------
# Fashion-related categories (whitelist)
# -----------------------------
FASHION_CATEGORIES = {
    "Baby Boys' Clothing & Shoes",
    "Baby Girls' Clothing & Shoes",
    "Boys' Clothing",
    "Girls' Clothing",
    "Men's Clothing",
    "Women's Clothing",
    "Boys' School Uniforms",
    "Girls' School Uniforms",
    "Men's Shoes",
    "Women's Shoes",
    "Boys' Shoes",
    "Girls' Shoes",
    "Men's Accessories",
    "Women's Accessories",
    "Boys' Accessories",
    "Girls' Accessories",
    "Women's Handbags",
    "Travel Duffel Bags",
    "Messenger Bags",
    "Travel Tote Bags",
    "Garment Bags",
    "Luggage Sets",
    "Suitcases",
    "Travel Accessories",
    "Backpacks",
    "Luggage",
    "Laptop Bags",
    "Men's Watches",
    "Women's Watches",
    "Boys' Watches",
    "Girls' Watches",
    "Men's Jewelry",
    "Women's Jewelry",
    "Boys' Jewelry",
    "Girls' Jewelry",
}


def main():
    # -----------------------------
    # Load raw products + categories
    # -----------------------------
    products = pd.read_csv(PRODUCTS_FILE)
    categories = pd.read_csv(CATEGORIES_FILE)

    print(f"✅ Loaded products: {len(products)} rows")
    print(f"✅ Loaded categories: {len(categories)} rows")
    print(f"📦 Product columns: {list(products.columns)}")
    print(f"📦 Category columns: {list(categories.columns)}")

    # -----------------------------
    # Merge on category_id
    # -----------------------------
    merged = products.merge(
        categories,
        left_on="category_id",
        right_on="id",
        how="left"
    )
    print(f"🔗 Merged dataset → {len(merged)} rows")

    # -----------------------------
    # Filter for fashion-related categories (whitelist)
    # -----------------------------
    before_count = len(merged)
    fashion = merged[merged["category_name"].isin(FASHION_CATEGORIES)].copy()
    after_count = len(fashion)

    print(f"🎯 Filtering step: {before_count} → {after_count} rows kept")
    print("📌 Unique categories after filtering:")
    print(fashion["category_name"].value_counts())

    # -----------------------------
    # Save processed dataset
    # -----------------------------
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    fashion.to_csv(OUTPUT_CSV, index=False)
    fashion.to_pickle(OUTPUT_PKL)

    print("\n✅ Saved processed dataset:")
    print(f"   • CSV → {OUTPUT_CSV}")
    print(f"   • PKL → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
