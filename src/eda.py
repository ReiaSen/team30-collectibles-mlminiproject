from pathlib import Path
import matplotlib.pyplot as plt
from src.data import load_and_clean
Path("results").mkdir(exist_ok=True)
df = load_and_clean()
print("Rows/products:", len(df))
print("\nCategory counts:\n", df["category"].value_counts())
print("\nPrice summary:\n", df["sale_price"].describe())
plt.figure(figsize=(8,5))
df["sale_price"].plot(kind="hist", bins=30)
plt.xlabel("Product-level median sale price")
plt.title("Distribution of real collectible sale prices")
plt.tight_layout()
plt.savefig("results/price_distribution.png", dpi=180)
plt.close()
