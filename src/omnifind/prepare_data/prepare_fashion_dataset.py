import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file if present
load_dotenv()
print("✅ Loaded environment variables from .env")
# Currency conversion rate and inflation factor
CONVERSION_RATE = 88.67 # 1 USD = 88.67 INR as of October 2025
INFLATION_FACTOR = 0.45  # Decrease price by 55% to account for inflation


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
    # "Boys' School Uniforms",
    # "Girls' School Uniforms",
    "Men's Shoes",
    "Women's Shoes",
    "Boys' Shoes",
    "Girls' Shoes",
    "Men's Accessories",
    "Women's Accessories",
    "Boys' Accessories",
    "Girls' Accessories",
    "Women's Handbags",
    # "Travel Duffel Bags",
    # "Messenger Bags",
    # "Travel Tote Bags",
    # "Garment Bags",
    # "Luggage Sets",
    # "Suitcases",
    # "Travel Accessories",
    # "Backpacks",
    # "Luggage",
    # "Laptop Bags",
    "Men's Watches",
    "Women's Watches",
    "Boys' Watches",
    "Girls' Watches",
    "Men's Jewelry",
    "Women's Jewelry",
    "Boys' Jewelry",
    "Girls' Jewelry",
}

# -----------------------------
# Utility: Clean price column
# -----------------------------
def clean_price(val):
    """Convert price strings like '$19.99' or '₹1,299' to float."""
    if pd.isna(val):
        return None
    val = str(val)
    val = val.replace("$", "").replace("₹", "").replace(",", "").strip()
    try:
        # We'll convert usd price to inr assuming 1 USD = 88.67 INR and decrease the price by 55% to account for inflation
        rupee_val_with_inflation = (float(val) * CONVERSION_RATE) * INFLATION_FACTOR
        return round(rupee_val_with_inflation, 2)
    except ValueError:
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Load raw data
    # -----------------------------
    products = pd.read_csv(PRODUCTS_FILE)
    categories = pd.read_csv(CATEGORIES_FILE)

    print(f"✅ Loaded products: {len(products)} rows")
    print(f"✅ Loaded categories: {len(categories)} rows")

    # -----------------------------
    # Merge to bring category_name into products
    # -----------------------------
    merged = products.merge(
        categories[["id", "category_name"]],
        left_on="category_id",
        right_on="id",
        how="left"
    )
    print(f"🔗 Merged dataset → {len(merged)} rows")

    # -----------------------------
    # Filter for fashion categories
    # -----------------------------
    before_count = len(merged)
    fashion = merged[merged["category_name"].isin(FASHION_CATEGORIES)].copy()
    after_count = len(fashion)

    print(f"🎯 Filtered: {before_count} → {after_count} rows kept")

    # -----------------------------
    # Drop redundant columns
    # -----------------------------
    drop_cols = [col for col in ["id", "category_id"] if col in fashion.columns]
    fashion.drop(columns=drop_cols, inplace=True, errors="ignore")

    # -----------------------------
    # Clean and convert data types
    # -----------------------------
    numeric_cols = ["price", "listPrice"]
    for col in numeric_cols:
        if col in fashion.columns:
            fashion[col] = fashion[col].apply(clean_price)

    if "stars" in fashion.columns:
        fashion["stars"] = pd.to_numeric(fashion["stars"], errors="coerce")

    if "reviews" in fashion.columns:
        fashion["reviews"] = pd.to_numeric(fashion["reviews"], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # Handle missing category_name
    # -----------------------------
    missing_categories = fashion["category_name"].isna().sum()
    if missing_categories:
        print(f"⚠️ Found {missing_categories} rows without category_name, dropping them.")
        fashion = fashion.dropna(subset=["category_name"])

    # -----------------------------
    # Save processed dataset
    # -----------------------------
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    fashion.to_csv(OUTPUT_CSV, index=False)
    fashion.to_pickle(OUTPUT_PKL)

    print("\n✅ Saved cleaned dataset:")
    print(f"   • CSV → {OUTPUT_CSV}")
    print(f"   • PKL → {OUTPUT_PKL}")
    print(f"🧾 Final columns: {list(fashion.columns)}")
    print(f"📊 Final shape: {fashion.shape}")


if __name__ == "__main__":
    main()
