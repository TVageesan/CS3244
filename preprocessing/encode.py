import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def cyclic_encode_month(df):
    """
    Apply cyclic encoding to the month field to capture its cyclical nature.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'month' column
        
    Returns:
        None: Modifies dataframe in-place
    """
    # Check for missing month values first
    if df['month'].isna().any():
        # Fill with the median month or use a different strategy
        df['month'] = df['month'].fillna(df['month'].median())
        
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Round to 0 to avoid e-17 values
    df['month_cos'] = df['month_cos'].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    df.drop(columns=['month'], inplace=True)


def normalize_year(df):
    """
    Normalize the year field to capture temporal trends properly.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'year' column
        reference_year (int, optional): Reference year for normalization
        
    Returns:
        None: Modifies dataframe in-place
    """

    reference_year = df['year'].min()
    df['year'] = df['year'] - reference_year


def scale_numerical_features(df, numerical_cols, scaler=None):
    """
    Standardize numerical features.
    
    Args:
        df (pandas.DataFrame): DataFrame with numerical columns
        numerical_cols (list): List of numerical column names
        scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fit scaler for transforms
        
    Returns:
        StandardScaler: The fitted scaler
    """
    # Filter to only include columns that exist in the dataframe
    cols_to_scale = [col for col in numerical_cols if col in df.columns]
    
    if not cols_to_scale:
        return scaler  # No columns to scale
        
    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    return scaler


def cap_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    """
    Cap outliers at specified quantiles.
    
    Args:
        df (pandas.DataFrame): DataFrame with column to cap
        column (str): Column name to cap
        lower_quantile (float): Lower quantile value for capping (default: 0.01)
        upper_quantile (float): Upper quantile value for capping (default: 0.99)
        
    Returns:
        tuple: (Lower limit, Upper limit) for the capped values
    """
    if column not in df.columns:
        return None, None
        
    lower_limit = df[column].quantile(lower_quantile)
    upper_limit = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)
    
    return lower_limit, upper_limit


def one_hot_encode(df, categorical_cols, encoder=None, handle_unknown='ignore'):
    """
    Apply one-hot encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        categorical_cols (list): List of categorical column names
        encoder (OneHotEncoder, optional): Pre-fit encoder for transforms
        handle_unknown (str): Strategy for handling unknown categories
        
    Returns:
        tuple: (DataFrame with one-hot encoded columns, OneHotEncoder)
    """
    # Filter to only include columns that exist in the dataframe
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    if not cols_to_encode:
        return df, encoder  # No columns to encode
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown=handle_unknown)
        encoded = encoder.fit_transform(df[cols_to_encode])
    else:
        encoded = encoder.transform(df[cols_to_encode])
    
    encoded_df = pd.DataFrame(
        encoded, 
        columns=encoder.get_feature_names_out(cols_to_encode),
        index=df.index
    )
    
    # Drop original categorical columns
    df = df.drop(columns=cols_to_encode)
    
    # Concatenate with the original dataframe
    df = pd.concat([df, encoded_df], axis=1)
    
    return df, encoder


def target_encode(df, cols, target_col, encodings=None):
    """
    Apply target encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        cols (list): List of categorical column names
        target_col (str): Target column name for encoding
        encodings (dict, optional): Pre-computed encodings mapping
        
    Returns:
        dict: Mappings of category to target mean for each column
    """
    result_encodings = {}
    
    for categorical_col in cols:
        if categorical_col not in df.columns:
            continue
            
        if encodings is not None and categorical_col in encodings:
            # Use pre-computed encodings
            df[f'{categorical_col}_encoded'] = df[categorical_col].map(encodings[categorical_col])
        else:
            # Compute new encodings
            mean_values = df.groupby(categorical_col)[target_col].mean()
            df[f'{categorical_col}_encoded'] = df[categorical_col].map(mean_values)
            
            # Store encodings for potential reuse
            if encodings is None:
                encodings = {}
            encodings[categorical_col] = mean_values.to_dict()
            
        # Add to result encodings
        result_encodings[categorical_col] = encodings[categorical_col]
        
    # Drop original columns only after all mappings are applied
    for categorical_col in cols:
        if categorical_col in df.columns:
            df.drop(columns=[categorical_col], inplace=True)
            
    return result_encodings


def frequency_encode(df, cols, encodings=None):
    """
    Apply frequency encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        cols (list): List of categorical column names
        encodings (dict, optional): Pre-computed encodings mapping
        
    Returns:
        dict: Mappings of category to frequency for each column
    """
    result_encodings = {}
    
    for categorical_col in cols:
        if categorical_col not in df.columns:
            continue
            
        if encodings is not None and categorical_col in encodings:
            # Use pre-computed encodings
            df[f'{categorical_col}_freq'] = df[categorical_col].map(encodings[categorical_col])
        else:
            # Compute new encodings
            freq_values = df[categorical_col].value_counts(normalize=True)
            df[f'{categorical_col}_freq'] = df[categorical_col].map(freq_values)
            
            # Store encodings for potential reuse
            if encodings is None:
                encodings = {}
            encodings[categorical_col] = freq_values.to_dict()
            
        # Add to result encodings
        result_encodings[categorical_col] = encodings[categorical_col]
        
    # Drop original columns only after all mappings are applied
    for categorical_col in cols:
        if categorical_col in df.columns:
            df.drop(columns=[categorical_col], inplace=True)
            
    return result_encodings

def add_moving_window_features(result_df, windows=[3, 6, 12]):
    """
    Add moving window features with strategies to handle the beginning values.
    
    Args:
        result_df (pandas.DataFrame): DataFrame with required columns
        windows (list): List of window sizes to calculate
        
    Returns:
        pandas.DataFrame: DataFrame with added window features
    """
    result_df['date'] = pd.to_datetime(result_df['year'].astype(str) + '-' + 
                                      result_df['month'].astype(str).str.zfill(2))
    
    # Sort by date to ensure correct time sequence
    result_df = result_df.sort_values('date')
    
    # Group by relevant property characteristics
    groupby_columns = ['town', 'flat_type', 'flat_model']
    grouped = result_df.groupby(groupby_columns)
    
    # For each window size, calculate moving features
    for window in windows:
        # Moving average (mean)
        result_df[f'price_ma_{window}'] = grouped['resale_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Moving standard deviation with special handling for initial values
        result_df[f'price_std_{window}'] = grouped['resale_price'].transform(
            lambda x: x.rolling(window=window, min_periods=2).std())
        
        # OPTION 1: Forward fill group STD from first valid value
        result_df[f'price_std_{window}'] = grouped[f'price_std_{window}'].transform(
            lambda x: x.fillna(method='bfill'))
        
        # OPTION 2 (alternative): For groups with fewer than 2 values, use global std
        # mask = result_df[f'price_std_{window}'].isna()
        # if mask.any():
        #    result_df.loc[mask, f'price_std_{window}'] = result_df['resale_price'].std()
        
        # OPTION 3 (alternative): Use expanding window for the first few observations
        # std_series = grouped['resale_price'].transform(
        #     lambda x: x.expanding(min_periods=2).std().fillna(x.std() if len(x) > 1 else 0))
        # result_df[f'price_std_{window}'] = std_series
        
        # Exponential moving average
        result_df[f'price_ema_{window}'] = grouped['resale_price'].transform(
            lambda x: x.ewm(span=window, min_periods=1, adjust=False).mean())
    
    # Make sure no NaN values remain in any of the new columns
    # Get all the newly added columns
    window_columns = [col for col in result_df.columns 
                      if any(f'price_{metric}_{w}' in col 
                            for metric in ['ma', 'std', 'ema'] 
                            for w in windows)]
    window_columns += [col for col in result_df.columns if '_pct' in col]
    
    # Check if any NaNs remain and handle them
    for col in window_columns:
        if result_df[col].isna().any():
            # As a last resort, use global statistics if any NaNs still exist
            if 'std' in col:
                result_df[col] = result_df[col].fillna(result_df['resale_price'].std())
            else:
                result_df[col] = result_df[col].fillna(result_df['resale_price'].mean())
    
    return result_df



def encode_data(dataframe, encoding_method='one_hot', handle_outliers=True, moving_window = True, cyclic_month = True, normal_year = True, normal_price = True):
    """
    Encode and transform the cleaned data.
    
    Args:
        dataframe (pandas.DataFrame): Cleaned DataFrame
        encoding_method (str): Method for encoding categorical variables 
                              ('one_hot', 'target', or 'frequency')
        reference_year (int, optional): Reference year for year normalization
        handle_outliers (bool): Whether to cap outliers before scaling
        
    Returns:
        pandas.DataFrame: Encoded DataFrame with target as the last column
    """
    df = dataframe.copy()

    # Define column groups
    categorical_cols = ['town', 'flat_type', 'flat_model']
    numerical_cols = ['floor_area_sqm', 'avg_floor', 'remaining_lease']
    
    if normal_price:
        numerical_cols.append('resale_price')
    
    target_col = 'resale_price'

    # Handle outliers in numerical columns if requested
    if handle_outliers:
        for col in numerical_cols:
            if col in df.columns:
                cap_outliers(df, col)
    
    scale_numerical_features(df, numerical_cols)
    
    if moving_window:
        df = add_moving_window_features(df)
    
    if cyclic_month:
        cyclic_encode_month(df) 

    if normal_year:
        normalize_year(df) 

    # Encode categorical features according to specified method
    if encoding_method == 'one_hot':
        df, _ = one_hot_encode(df, categorical_cols)
    elif encoding_method == 'target':
        target_encode(df, categorical_cols, target_col)
    elif encoding_method == 'frequency':
        frequency_encode(df, categorical_cols)
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    return df