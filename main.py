from utils.file_utils import merge_csv, read_csv
from utils.calc_distance import add_nearest_mrt
from utils.geocode import geocode
from preprocessing.clean import clean_data
from preprocessing.encode import encode_data
import argparse
import os

def process_data(skip_geocode=True):
    """
    Process the data pipeline from raw data to encoded data.
    
    Args:
        skip_geocode (bool): Whether to skip geocoding and use cached data
    """
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    csv_files = ['data/2012.csv', 'data/2015.csv', 'data/2017.csv']
    print("Merging CSV files...")

    merged_data = merge_csv(csv_files)
    merged_data.to_csv('output/data.csv', index=False)
    print("Merged data saved to 'data.csv'")

    print("Geocoding data...")
    if skip_geocode:
        print("Skipping geocoding step (using cached data)")
        # Use pre-geocoded data
        geocoded_data = read_csv('data/geocoded.csv')
    else:
        # Perform geocoding
        geocoded_data = geocode(read_csv('output/data.csv'))
        geocoded_data.to_csv('data/geocoded.csv')

    print("Adding nearest MRT information...")
    enhanced_data = add_nearest_mrt(geocoded_data)
    
    print("Cleaning data...")
    cleaned_data = clean_data(enhanced_data)

    # Extract useful fields
    column_order = [
        'year', 'month', 
        'town', 'longitude', 'latitude', 'distance_to_nearest_mrt', 
        'avg_floor', 'flat_type', 'flat_model', 'floor_area_sqm', 'remaining_lease', 
        'resale_price'
    ]

    # Filter columns that exist in the dataframe
    existing_columns = [col for col in column_order if col in cleaned_data.columns]
    cleaned_data = cleaned_data[existing_columns]

    cleaned_data.to_csv('output/clean_data.csv', index=False)
    print("Cleaned data saved to 'clean_data.csv'")

    print("Encoding data...")
    encoding_options = {
        'encoding_method': 'one_hot',
        'handle_outliers': True,
        'moving_window': True,
        'cyclic_month': True,
        'normal_year': True,
        'normal_price': True
    }

    encoded_data = encode_data(cleaned_data, **encoding_options)
    encoded_data.to_csv('output/encoded_data.csv', index=False)
    print("Encoded data saved to 'encoded_data.csv'")
    
    print("Data preprocessing complete!")
    
    return encoded_data

def main():
    parser = argparse.ArgumentParser(description='Process housing data and train models')
    parser.add_argument('--skip-processing', action='store_true', 
                        help='Skip data processing and use existing encoded data')
    parser.add_argument('--train-models', action='store_true',
                        help='Train models after data processing')
    parser.add_argument('--model-mode', choices=['compare', 'single'], default='compare',
                        help='Model training mode (if --train-models is specified)')
    parser.add_argument('--model', choices=['linear', 'ridge', 'lasso', 'rf', 'knn'],
                        help='Specific model to train (if model-mode is "single")')
    parser.add_argument('--features', default='all_features',
                        help='Feature set to use for modeling')
    parser.add_argument('--optimize', action='store_true',
                        help='Use hyperparameter optimization')
    
    args = parser.parse_args()
    
    if not args.skip_processing:
        process_data(skip_geocode=True)  # Default to using cached geocoded data
    else:
        print("Skipping data processing (using existing encoded data)")
    
    if args.train_models:
        from train_models import run_full_comparison, evaluate_specific_model
        
        if args.model_mode == 'compare':
            print("\nRunning model comparison...")
            run_full_comparison('output/encoded_data.csv', args.features, args.optimize)
        else:  # single mode
            if not args.model:
                parser.error("--model is required when model-mode is 'single'")
            print(f"\nEvaluating {args.model} model...")
            evaluate_specific_model('output/encoded_data.csv', args.model, args.features, args.optimize)
    
if __name__ == "__main__":
    main()