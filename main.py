from utils.file_utils import merge_csv, read_csv
from utils.calc_distance import add_nearest_mrt
from preprocessing.clean import clean_data
from preprocessing.encode import encode_data

def main():
    csv_files = ['data/2012.csv', 'data/2015.csv', 'data/2017.csv']

    print("Merging CSV files...")

    merged_data = merge_csv(csv_files)
    
    merged_data.to_csv('output/data.csv', index=False)
    print("Merged data saved to 'data.csv'")

    print("Geocoding data...")
    ## Since there's API Limits, try to use the generated geocoded_data set instead of creating yourself
    geocoded_data = read_csv('output/geocoded_data.csv')
    enhanced_data = add_nearest_mrt(geocoded_data)
    cleaned_data = clean_data(enhanced_data)

    # Extract useful fields
    # Note: Might want to get rid of 'town' field for linear regression,
    # Or Get rid of 'longitude', 'latitude' for decision tree
    column_order = [
        'year', 'month', 
        'town', 'longitude', 'latitude', 'distance_to_nearest_mrt', 
        'avg_floor', 'flat_type', 'flat_model', 'floor_area_sqm', 'remaining_lease', 
        'resale_price'
    ]

    cleaned_data = cleaned_data[column_order]

    cleaned_data.to_csv('output/clean_data.csv', index=False)
    
    print("Cleaned data saved to 'clean_data.csv'")

    print("Encoding data...")

    encoded_data = encode_data(cleaned_data, encoding_method='one_hot', handle_outliers=True, moving_window = True, cyclic_month = True, normal_year = True, normal_price = True)

    encoded_data.to_csv('output/encoded_data.csv', index=False)
    print("Encoded data saved to 'encoded_data.csv'")
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()