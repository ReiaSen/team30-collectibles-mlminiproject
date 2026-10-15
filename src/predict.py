import argparse, json
import pandas as pd
import joblib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Predict a Funko Pop price tier using the trained real-data model.")
    parser.add_argument("--name", required=True, help="Product name")
    parser.add_argument("--category", required=True, help="Category such as Anime, Marvel, Disney")
    parser.add_argument("--material", default="Unknown")
    parser.add_argument("--special-version", dest="special_version", default="Unknown")
    parser.add_argument("--search-volume", dest="search_volume", type=float, default=float("nan"))
    parser.add_argument("--release-year", dest="release_year", type=float, default=float("nan"))
    args = parser.parse_args()
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        raise SystemExit("No trained model found. Add real data and run: python -m src.train")
    name = args.name
    low = name.lower()
    row = {
        "product_name": name, "category": args.category, "material": args.material,
        "special_version": args.special_version, "name_length": len(name),
        "name_word_count": len(name.split()), "release_year": args.release_year,
        "search_volume": args.search_volume,
        "log_search_volume": __import__("math").log1p(max(0,args.search_volume)) if args.search_volume == args.search_volume else float("nan")
    }
    for token in ["metallic", "glow", "flocked", "chase", "exclusive", "sdcc", "limited", "glitter", "chrome"]:
        row[f"name_has_{token}"] = int(token in low)
    model = joblib.load(model_path)
    pred = int(model.predict(pd.DataFrame([row]))[0]) + 1
    print(f"Predicted relative price tier: {pred}/5")
    print("Tier 1 = lower-priced group; Tier 5 = higher-priced group.")
    print("This is a relative category, not an exact price estimate.")
if __name__ == "__main__":
    main()
