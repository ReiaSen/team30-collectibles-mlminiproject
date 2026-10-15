from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from src.data import load_and_clean

R = Path("results"); M = Path("models")
R.mkdir(exist_ok=True); M.mkdir(exist_ok=True)

def main():
    df = load_and_clean()
    # Quantile thresholds are learned from training prices only to avoid leakage.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=42
    )
    train_prices = df.iloc[train_idx]["sale_price"]
    # Duplicate quantiles can occur with repeated prices; fail transparently instead of silently mislabeling.
    cuts = np.unique(train_prices.quantile([.2, .4, .6, .8]).to_numpy())
    if len(cuts) != 4:
        raise ValueError("Cannot create five distinct price bins: training prices have too many tied values. "
                         "Use a larger/more varied real dataset or document a justified alternative binning.")
    edges = [-np.inf, *cuts.tolist(), np.inf]
    y = pd.cut(df["sale_price"], bins=edges, labels=False, include_lowest=True).astype(int)
    X = df.drop(columns=["sale_price", "product_url"], errors="ignore")
    # Price bin target is ordinal; features must not include the target price.
    categorical = [c for c in ["category", "product_name", "material", "special_version"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical and c not in ["release_date"]]
    # Avoid a very high-dimensional raw product-name one-hot representation; use name-derived features only.
    if "product_name" in categorical:
        categorical.remove("product_name")
    numeric = [c for c in numeric if c != "product_name"]
    # Product name text is converted to interpretable features in data.py.
    pre = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), numeric)
    ], remainder="drop")
    base_models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=20000),
        "RBF SVM": SVC(kernel="rbf", class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42, min_samples_leaf=2),
        "Bagged Trees": BaggingClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=2, class_weight="balanced"),
                                           n_estimators=100, random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    }
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Ensure stratification is viable for the observed class counts.
    counts = y_train.value_counts()
    if counts.min() < 2:
        raise ValueError("At least one price tier has fewer than 2 training examples. More real observations are needed.")
    rows, fitted = [], {}
    for name, model in base_models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0)
        })
        fitted[name] = (pipe, pred)
    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(R/"metrics.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    (R/"classification_report.txt").write_text(
        f"Best model by macro-F1: {best_name}\n\n" +
        classification_report(y_test, best_pred, zero_division=0, digits=3) +
        "\nPrice-bin edges learned from training data:\n" + repr(edges) +
        "\n\nClass counts (full dataset):\n" + y.value_counts().sort_index().to_string()
    )
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred, labels=range(5)),
                                 display_labels=[f"Tier {i+1}" for i in range(5)])
    disp.plot(values_format="d", xticks_rotation=30)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout(); plt.savefig(R/"confusion_matrix.png", dpi=180); plt.close()
    plt.figure(figsize=(9, 5))
    plotdf = metrics.sort_values("macro_f1")
    plt.barh(plotdf["model"], plotdf["macro_f1"])
    plt.xlabel("Macro-F1 (higher is better)"); plt.title("Model Comparison on Held-Out Real Data")
    plt.tight_layout(); plt.savefig(R/"model_comparison.png", dpi=180); plt.close()
    joblib.dump(best_pipe, M/"best_model.joblib")
    metadata = {
        "best_model": best_name, "n_unique_products": int(len(df)),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "price_bin_edges": [None if not np.isfinite(v) else float(v) for v in edges],
        "target": "five price tiers derived from training split price quantiles",
        "dataset_path": "data/raw/funko_stockx.csv",
        "features": list(X.columns)
    }
    (M/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Training complete on supplied data.")
    print(metrics.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved outputs in {R.resolve()} and {M.resolve()}")
    print("Important: report the dataset source and sample size from data/SOURCE_NOTES.md.")

if __name__ == "__main__":
    main()
