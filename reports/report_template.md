# Application of Machine Learning Methods to Evaluate Collectibles
**Team 30 | Project ID 50 | UE24CS352A**

## 1. Problem statement
Collectibles can have substantially different secondary-market values even within a single product line. This project investigates whether product metadata and public-interest signals can classify Funko Pop! collectibles into five relative price tiers.

## 2. Dataset
The project follows the dataset description in Xiaotian Lu's Stanford CS229 report (2021). The paper describes real StockX secondary-market data for Funko Pop! products, with categories Ad Icons, Anime, Disney, Game of Thrones and Marvel. It reports 358 products after excluding products without transaction data. Product features include product name, category/topic, material, special version, limited release-date information and search volume. The paper reports release-date information for 108 items. The PDF does not itself provide the raw row-level dataset; therefore, this report must be completed only after a real, permitted dataset is obtained and audited.

**Actual dataset used:** [fill from data/SOURCE_NOTES.md]  
**Rows after cleaning:** [populate from run]  
**Target price type:** [completed transaction / product-level median completed transaction / other—state accurately]

## 3. Methodology
Data cleaning removes missing product names/categories/prices and non-positive prices. Multiple transactions per product are aggregated using median sale price. The five price tiers are derived from quantiles in the training split, avoiding use of held-out prices to define target boundaries. Product-name features capture edition terms such as metallic, glow, chase, exclusive and SDCC. Categorical features are one-hot encoded and numeric features imputed and scaled.

Models compared: majority baseline, Linear SVM, RBF SVM, Decision Tree, Bagged Decision Trees and Random Forest. The split is held out for final evaluation. Metrics include accuracy, balanced accuracy, macro-F1 and weighted-F1. Macro-F1 is emphasized due to class imbalance.

## 4. Results
**Do not fill this section until the training pipeline has run on real data.**  
Best model: [from results/metrics.csv]  
Accuracy: [value]  
Balanced accuracy: [value]  
Macro-F1: [value]  
Insert `results/model_comparison.png` and `results/confusion_matrix.png`.

## 5. Conclusion
Summarize which model generalized best, which price tiers were hardest to classify, and whether metadata/search volume helped. Discuss data imbalance, limited release-date coverage, possible marketplace bias, and the fact that price tiers are relative to the observed dataset. Do not claim exact-price accuracy from a classifier.

## Reference
Lu, X. (2021). *Application of Machine Learning Methods to Evaluate Collectibles*. Stanford CS229 project report. https://cs229.stanford.edu/proj2021spr/report2/81980743.pdf
