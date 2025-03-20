from utils.file_utils import merge_csv
from preprocessing.clean import clean_data
from preprocessing.encode import encode_data

def main():
    csv_files = ['data/2012.csv', 'data/2015.csv', 'data/2017.csv']

    print("Merging CSV files...")

    merged_data = merge_csv(csv_files)
    
    merged_data.to_csv('output/data.csv', index=False)
    print("Merged data saved to 'data.csv'")

    print("Cleaning data...")
    
    cleaned_data = clean_data(merged_data, drop_fields = ['block', 'street_name', 'storey_range', 'lease_commence_date'])
    
    cleaned_data.to_csv('output/clean_data.csv', index=False)
    
    print("Cleaned data saved to 'clean_data.csv'")

    print("Encoding data...")

    encoded_data = encode_data(cleaned_data, encoding_method='one_hot', handle_outliers=True, moving_window = True, cyclic_month = True, normal_year = True, normal_price = True)

    encoded_data.to_csv('output/encoded_data.csv', index=False)
    print("Encoded data saved to 'encoded_data.csv'")
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()