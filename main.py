from utils.file_utils import merge_csv
from preprocessing.clean import clean_data
from preprocessing.encode import encode_data

def main():
    """
    Main entry point for the data preprocessing pipeline.
    
    Steps:
    1. Merge CSV files
    2. Clean data
    3. Encode data
    4. Save all results
    """
    csv_files = ['data/2012.csv', 'data/2015.csv', 'data/2017.csv']

    print("Merging CSV files...")
    merged_data = merge_csv(csv_files)
    merged_data.to_csv('output/data.csv', index=False)
    print("Merged data saved to 'data.csv'")

    print("Cleaning data...")
    cleaned_data = clean_data(merged_data)
    cleaned_data.to_csv('output/clean_data.csv', index=False)
    print("Cleaned data saved to 'clean_data.csv'")

    print("Encoding data...")
    encoded_data = encode_data(cleaned_data)
    encoded_data.to_csv('output/encoded_data.csv', index=False)
    print("Encoded data saved to 'encoded_data.csv'")
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()