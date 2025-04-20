## How to use these files
Generally, all relevant content is contained in `main.ipynb`. This is the file where we combine all our individual packages to get our final comparisons and evaluations done. In the `individual` folder, we've kept a record of the individual work each member did for their various models before being integrated into the main pipeline. These ipynbs are no longer used as everything has been packaged into the `models` package, but are kept for record keeping and acknowledgement of everyone's work.

The model accuracies may be inconsistent between the main file and the individual files as different versions of the dataset, with different feature engineering choices were used during development. After doing model evaluation and testing for a optimal feature set, a consistent dataset (in output/encoded_data.csv) was used to test all models in `main.ipynb`, which resulted in different (more accurate) metrics.

EDA was done in a seperate ipynb which can be run seperately in `eda.ipynb`.
## File Structure

```
├── data/
│   ├── 2012.csv          # Raw data files
│   ├── 2015.csv
│   ├── 2017.csv
│   ├── geocoded.csv      # Cached geocoded data
│   ├── mall.csv          # Mall station location
│   └── mrt.csv           # MRT station locations
├── models/
│   ├── comparison.py     # Gets models from following model files and runs comparison metrics
│   ├── graph.py          # Utilities to graph model evaluation results
│   ├── validate.py       # Utilities to evaluate model accuracies
│   ├── knn.py            # Individual model implementations
│   ├── linear_regression.py     
│   ├── random_forest.py  
│   └── regression_tree.py   
├── output/
│   ├── data.csv          # Merged raw data
│   ├── clean_data.csv    # Cleaned data
│   └── encoded_data.csv  # Fully processed data
├── preprocessing/
│   ├── clean.py          # Data cleaning functions
│   └── encode.py         # Feature encoding functions
├── utils/
│   ├── calc_distance.py  # Distance calculation utilities for spatial features
│   ├── file_utils.py     # File handling utilities
│   └── geocode.py        # Geocoding utilities
├── main.ipynb            # Compiled report on module evaluation & comparison
└── eda.ipynb             # EDA notebook 

```

## Feature Engineering Choices & Justifications


## Model Evaluation Analysis


## References

https://www.homeanddecor.com.sg/property/property-will-home-prices-increase-as-train-network-grows-in-singapore



