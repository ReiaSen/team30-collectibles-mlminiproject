from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW = Path("data/raw/funko_stockx.csv")
PROCESSED = Path("data/processed/cleaned_collectibles.csv")

ALIASES = {
    "product_name": ["product_name", "name", "title", "product", "item_name"],
    "category": ["category", "topic", "franchise", "product_category"],
    "sale_price": ["sale_price", "price", "average_sale_price", "avg_sale_price", "average_price", "last_sale"],
    "release_date": ["release_date", "release date", "released"],
    "material": ["material"],
    "special_version": ["special_version", "special version", "edition", "variant"],
    "search_volume": ["search_volume", "search volume", "google_trends", "trend_volume"],
    "product_url": ["product_url", "url", "link"],
}

def _normalise_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    rename = {}
    for target, candidates in ALIASES.items():
        for c in candidates:
            key = c.replace(" ", "_").lower()
            if key in df.columns:
                rename[key] = target
                break
    return df.rename(columns=rename)

def load_and_clean(path=RAW):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found: {path}. Obtain authorized real transaction data and save it here. "
            "This project intentionally does not create synthetic data."
        )
    df = _normalise_columns(pd.read_csv(path))
    required = {"product_name", "category", "sale_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. CSV must include product name, category, and real sale price. "
            f"Available columns: {list(df.columns)}"
        )
    df["sale_price"] = pd.to_numeric(
        df["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df = df.dropna(subset=["product_name", "category", "sale_price"])
    df = df[(df["product_name"] != "") & (df["category"] != "") & (df["sale_price"] > 0)].copy()
    # If multiple transactions exist for a product, aggregate to product-level median price.
    # Median is robust to outlier sales; keep optional metadata as the first non-null value.
    aggregations = {"sale_price": "median", "category": "first"}
    for c in ["release_date", "material", "special_version", "search_volume", "product_url"]:
        if c in df.columns:
            aggregations[c] = "first"
    df = df.groupby("product_name", as_index=False).agg(aggregations)
    if len(df) < 25:
        raise ValueError(f"Only {len(df)} valid unique products found. At least 25 are required for a preliminary model; more is strongly recommended.")
    df["product_name"] = df["product_name"].astype(str)
    df["category"] = df["category"].astype(str)
    df["name_length"] = df["product_name"].str.len()
    df["name_word_count"] = df["product_name"].str.split().str.len()
    name_lower = df["product_name"].str.lower()
    for token in ["metallic", "glow", "flocked", "chase", "exclusive", "sdcc", "limited", "glitter", "chrome"]:
        df[f"name_has_{token}"] = name_lower.str.contains(rf"\b{token}\b", regex=True).astype(int)
    if "release_date" in df.columns:
        dt = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = dt.dt.year
    if "search_volume" in df.columns:
        df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce")
        df["log_search_volume"] = np.log1p(df["search_volume"].clip(lower=0))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    return df
