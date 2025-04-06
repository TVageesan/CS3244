import pandas as pd
import re

def format_storey_field(dataframe):
    """
    Convert 'storey_range' field to numerical 'avg_floor'.
    
    Extracts average floor from ranges like '1 TO 5' -> 3.0
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with 'storey_range' column
        
    Returns:
        None: Modifies dataframe in-place
    """
    def extract_avg_floor(storey_range):
        match = re.search(r'(\d+)\s+TO\s+(\d+)', storey_range)
        start, end = int(match.group(1)), int(match.group(2))
        avg_floor = (start + end) / 2
        return avg_floor
    dataframe['avg_floor'] = dataframe['storey_range'].apply(extract_avg_floor)


def format_month_field(dataframe):
    """
    Separate 'month' field (e.g., '2012-3') into 'year' and 'month' fields.
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with 'month' column
        
    Returns:
        None: Modifies dataframe in-place
    """
    def extract_year_month(month_string):
        parts = month_string.split('-')
        year = int(parts[0])
        month = int(parts[1])
        return year, month
    dataframe['year'], dataframe['month'] = zip(*dataframe['month'].apply(extract_year_month))


def format_lease_field(dataframe):
    """
    Process 'remaining_lease' field and fill in missing values.
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with relevant columns
        
    Returns:
        None: Modifies dataframe in-place
    """
    # Handle missing values by calculating from year and lease_commence_date
    mask_missing = dataframe['remaining_lease'].isna()
    if mask_missing.any():
        dataframe.loc[mask_missing, 'remaining_lease'] = 99 - (dataframe.loc[mask_missing, 'year'] - dataframe.loc[mask_missing, 'lease_commence_date'])
    
    # Process string values to convert to float
    def convert_lease_to_float(lease_str):
        if pd.isna(lease_str) or isinstance(lease_str, (int, float)):
            return lease_str
            
        # Extract years and months using regex
        years_match = re.search(r'(\d+)\s*years?', str(lease_str))
        months_match = re.search(r'(\d+)\s*months?', str(lease_str))
        
        years = float(years_match.group(1)) if years_match else 0
        months = float(months_match.group(1)) if months_match else 0
        
        # Convert months to years (divide by 12) and add to years
        return years + (months / 12)
    
    # Apply the conversion function
    dataframe['remaining_lease'] = dataframe['remaining_lease'].apply(convert_lease_to_float)


def clean_data(dataframe):
    """
    Clean the data according to requirements:
    1) Transform storey-range -> numerical avg_floor feature
    1) Transform storey-range -> numerical avg_floor feature
    2) Process month "2012-3" into separate numerical year and month fields
    3) Process remaining_lease in format "0.4 years 3 months" into numerical format
    4) Fill in missing remaing_lease fields 
    5) re-format and drop unnecessary fields
    
    Args:
        dataframe (pandas.DataFrame): Raw data DataFrame
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    df = dataframe.copy()
    format_storey_field(df)
    format_month_field(df)
    format_lease_field(df)
    return df