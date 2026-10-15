# Application of Machine Learning Methods to Evaluate Collectibles
**Team 30 | Problem ID 50 | UE24CS352A — Machine Learning Mini-Project**

This project follows the approach in Xiaotian Lu's Stanford CS229 report, *Application of Machine Learning Methods to Evaluate Collectibles*:
https://cs229.stanford.edu/proj2021spr/report2/81980743.pdf

The reference project studies real Funko Pop! secondary-market data from StockX and classifies items into five price bins using SVM and tree-based models. It describes 358 products across Ad Icons, Anime, Disney, Game of Thrones, and Marvel after filtering products without transaction data. This repository is a fresh implementation of that methodology; it does **not** contain fabricated observations or claim to reproduce the paper's results.

## Data integrity (read this first)
The paper describes the data and collection process, but the PDF does not embed the original row-level dataset. This project deliberately does not generate a synthetic replacement. Put the real row-level data you are authorized to use in `data/raw/funko_stockx.csv`.

Expected columns (aliases are supported in many cases):
- `product_name` (or `name` / `title`)
- `category` (or `topic` / `franchise`)
- `sale_price` (or `price` / `average_sale_price`)
- optional: `release_date`, `material`, `special_version`, `search_volume`, `product_url`

The paper says it used StockX transaction data. Its original project repository is linked in the paper:
https://github.com/timothyxlu/CS229/
Check that repository for any publicly available data/code and follow the source's license/terms. If no downloadable CSV is available, assemble a CSV from a lawful, permitted source of real completed sales and record the source/date in `data/SOURCE_NOTES.md`. Public product pages/listing prices are **not** the same as completed transaction prices. Do not bypass access controls or website terms.

## Research question
Can product metadata and market-interest signals classify real Funko Pop! collectibles into relative price tiers?

## Methodology
1. Validate and clean real product/transaction data.
2. Derive a five-class target using price quintiles computed from the training split only.
3. Parse useful terms from product names (e.g., metallic, exclusive, SDCC, character/topic).
4. Encode categorical fields; log-transform and scale search volume where available.
5. Train and compare:
   - Majority-class baseline
   - Linear SVM
   - RBF SVM
   - Decision Tree
   - Bagged Decision Trees
   - Random Forest
6. Use stratified train/test split and stratified cross-validation when sample sizes permit.
7. Report accuracy, macro-F1, balanced accuracy, confusion matrix and per-class metrics. Macro-F1 is emphasized because the paper identifies class imbalance as a central challenge.
8. Save metrics, plots, trained model and a transparent data audit.

## Setup (macOS)
```bash
cd team30_collectibles_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Add data and run
```bash
# Place your real CSV here:
# data/raw/funko_stockx.csv
python -m src.train
python -m src.predict --name "Funko Pop Animation Dragon Ball Z Beerus Metallic SDCC Figure 120" --category Anime
```
If no valid real dataset exists, training stops with a clear error rather than creating fake data.

## Outputs
- `data/processed/cleaned_collectibles.csv`
- `results/metrics.csv`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/model_comparison.png`
- `models/best_model.joblib`
- `models/metadata.json`

## Limitations
- The paper notes that release-date information was available for only 108 products and that high-price classes were sparse.
- Price bins and scores depend on the collected dataset and collection date.
- This is a research/educational classifier, not financial advice or a guaranteed price estimator.
- Do not report performance numbers until `python -m src.train` has successfully evaluated real data.

## Reference
Lu, X. (2021). *Application of Machine Learning Methods to Evaluate Collectibles*. Stanford CS229 project report.
https://cs229.stanford.edu/proj2021spr/report2/81980743.pdf
