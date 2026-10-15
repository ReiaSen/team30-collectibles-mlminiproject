# Dataset Source and Provenance

## Source
Original Stanford CS229 project:
Application of Machine Learning Methods to Evaluate Collectibles.

Original repository:
https://github.com/timothyxlu/CS229/

Direct CSV:
https://raw.githubusercontent.com/timothyxlu/CS229/master/project/src/data_sourcing/data_curated.csv

## Dataset Details
Download date: 2026-10-15
Unique products after cleaning: 359
Price field: average_sale_price
Prediction target: Five relative price tiers.

## Category Counts
- Anime: 198
- Marvel: 69
- Ad Icons: 44
- Disney: 37
- Game of Thrones: 11

## Preprocessing
- Removed records with missing product names, categories or prices.
- Removed non-positive prices.
- Aggregated records with the same product name using median price.
- Derived product-name features and numerical features.
- Created five price tiers using training-set price quantiles.

## Important Limitations
The Stanford paper reports 358 products, while our cleaned dataset contains 359.
The difference must be investigated before claiming exact reproduction.
The source's data licensing and usage terms have not yet been independently verified.
The price field is average_sale_price; it should not be described as an individual
completed transaction price without verifying the source's definition.

## Experimental Results
Best model by macro-F1: RBF SVM
Test samples: 90
Accuracy: 0.3778
Balanced accuracy: 0.3481
Macro-F1: 0.3448
Weighted-F1: 0.3742

These results are from the current experiment and are not guaranteed to generalize
to other collectible products or future market prices.
