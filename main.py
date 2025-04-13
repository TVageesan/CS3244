from utils.file_utils import merge_csv, read_csv
from utils.calc_distance import add_nearest_mrt
from utils.geocode import geocode
from preprocessing.clean import clean_data
from preprocessing.encode import encode_data
from models.validate import prepare_split
from models.knn import get_knn, test_k_values


def process_data(skip_geocode=True):
    """
    Process the data pipeline from raw data to encoded data.
    
    Args:
        skip_geocode (bool): Whether to skip geocoding and use cached data
    """
    # Ensure output directory exists    
    csv_files = ['data/2012.csv', 'data/2015.csv', 'data/2017.csv']
    print("Merging CSV files...")

    merged_data = merge_csv(csv_files)
    merged_data.to_csv('output/data.csv', index=False)
    print("Merged data saved to 'data.csv'")

    print("Geocoding data...")
    if skip_geocode:
        print("Skipping geocoding step (using cached data)")
        geocoded_data = read_csv('data/geocoded.csv')
    else:
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
        'handle_outliers': False,
        'moving_window': False,
        'cyclic_month': True,
        'normal_year': True,
        'normal_price': False
    }

    encoded_data = encode_data(cleaned_data, **encoding_options)
    encoded_data.to_csv('output/encoded_data.csv', index=False)    
    print("Data preprocessing complete!")
    
    return encoded_data

def main():
    skipProcess = False

    data = process_data() if skipProcess else read_csv('output/encoded_data.csv') 
    train_X, test_X, train_y, test_y = prepare_split(data)
    test_k_values(train_X, train_y)

if __name__ == "__main__":
    main()