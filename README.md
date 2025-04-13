# Housing Price Prediction System

This repository contains an integrated system for processing housing data, training predictive models, and evaluating model performance for Singapore's resale flat prices.

## System Overview

The system is organized into several components:

1. **Data Processing Pipeline** (`main.py`): Processes raw data through cleaning, geocoding, and feature engineering
2. **Model Training Framework** (`models/model_trainer.py`): Unified interface for training and evaluating models
3. **Model Implementations** (`models/model_implementations.py`): Implementations of various regression models
4. **Model Comparison** (`train_models.py`): Command-line tool for training and comparing multiple models
5. **Results Visualization** (`compare_results.py`): Tools for visualizing model comparison results

## Directory Structure

```
├── data/
│   ├── 2012.csv          # Raw data files
│   ├── 2015.csv
│   ├── 2017.csv
│   ├── geocoded.csv      # Cached geocoded data
│   └── mrt.csv           # MRT station locations
├── models/
│   ├── model_trainer.py  # Unified model training framework
│   └── model_implementations.py  # Model implementations
├── output/
│   ├── data.csv          # Merged raw data
│   ├── clean_data.csv    # Cleaned data
│   ├── encoded_data.csv  # Fully processed data
│   └── model_results/    # Model evaluation results
├── preprocessing/
│   ├── clean.py          # Data cleaning functions
│   └── encode.py         # Feature encoding functions
├── utils/
│   ├── calc_distance.py  # Distance calculation utilities
│   ├── file_utils.py     # File handling utilities
│   └── geocode.py        # Geocoding utilities
├── main.py               # Main data processing script
├── train_models.py       # Model training script
└── compare_results.py    # Results visualization script
```

## Usage

### 1. Data Processing

To process raw data into the format required for modeling:

```bash
python main.py
```

This will:
- Merge CSV files from different years
- Use cached geocoded data (or perform geocoding if `--skip-geocode` is not set)
- Add nearest MRT information
- Clean and format the data
- Encode categorical features and normalize numerical features
- Save the processed data to `output/encoded_data.csv`

### 2. Model Training

#### Compare Multiple Models

To train and compare all available models:

```bash
python train_models.py compare --features all_features
```

Options:
- `--features`: Choose feature set (`all_features`, `no_town`, `no_coordinates`, `minimal`)
- `--optimize`: Enable hyperparameter optimization
- `--data`: Specify path to data file (default: `output/encoded_data.csv`)
- `--output`: Specify directory for saving results (default: `output/model_results`)

#### Train a Specific Model

To train and evaluate a single model:

```bash
python train_models.py single --model rf --features no_town --optimize
```

Available models:
- `linear`: Linear Regression
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `rf`: Random Forest
- `knn`: K-Nearest Neighbors

### 3. All-in-One Processing and Training

You can also process data and train models in a single command:

```bash
python main.py --train-models --model-mode compare --features all_features --optimize
```

Or for a specific model:

```bash
python main.py --train-models --model-mode single --model knn --features no_coordinates
```

### 4. Comparing Results

To compare results from multiple training runs:

```bash
python compare_results.py --dir output/model_results
```

Or specify specific result files:

```bash
python compare_results.py --files output/model_results/all_features_comparison.json output/model_results/no_town_comparison.json
```

Options:
- `--metrics`: Specify metrics to compare (default: `r2 rmse mae train_time`)
- `--output`: Save the comparison plot to a file

## Feature Sets

The system supports different feature set configurations for model comparison:

1. `all_features`: All available features
2. `no_town`: Uses geographical coordinates instead of town categorical features
3. `no_coordinates`: Uses town categorical features instead of coordinates
4. `minimal`: Only essential features (area, floor, lease, MRT distance, time)

## Adding New Models

To add a new model:

1. Implement the model in `models/model_implementations.py`
2. Update `train_models.py` to include the new model in the model registry

## Adding New Feature Sets

To test new feature combinations:

1. Update the `get_feature_sets()` function in `models/model_implementations.py`
2. Use the new feature set with the `--features` flag

## Best Practices

1. **Reproducibility**: Use `--random-state` for consistent results
2. **Model Comparison**: Use the same feature set and test/train split for fair comparisons
3. **Hyperparameter Tuning**: Use the `--optimize` flag for better model performance
4. **Visualization**: Use `compare_results.py` to visualize and compare model performance