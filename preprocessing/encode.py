import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.calc_distance import *


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

import pandas as pd

import pandas as pd

def add_moving_window_features(result_df, windows=[3, 6, 12]):
    """
    Add moving window features using only past data (no data leakage).
    
    Args:
        result_df (pandas.DataFrame): DataFrame with required columns
        windows (list): List of window sizes to calculate
        
    Returns:
        pandas.DataFrame: DataFrame with added window features
    """
    result_df = result_df.copy()
    
    # Create a datetime column for sorting
    result_df['date'] = pd.to_datetime(result_df['year'].astype(str) + '-' + 
                                       result_df['month'].astype(str).str.zfill(2))
    result_df = result_df.sort_values(['town', 'flat_type', 'flat_model', 'date'])

    groupby_columns = ['town', 'flat_type', 'flat_model']

    for window in windows:
        ma_col = f'price_ma_{window}'
        std_col = f'price_std_{window}'
        ema_col = f'price_ema_{window}'

        # Initialize columns
        result_df[ma_col] = None
        result_df[std_col] = None
        result_df[ema_col] = None

        # Calculate leakage-free moving features per group
        for _, group_idx in result_df.groupby(groupby_columns).groups.items():
            group = result_df.loc[group_idx].copy()

            result_df.loc[group_idx, ma_col] = (
                group['resale_price']
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)  # prevent data leakage
            )

            result_df.loc[group_idx, std_col] = (
                group['resale_price']
                .rolling(window=window, min_periods=2)
                .std()
                .shift(1)
            )

            result_df.loc[group_idx, ema_col] = (
                group['resale_price']
                .ewm(span=window, min_periods=1, adjust=False)
                .mean()
                .shift(1)
            )

    # Fill remaining NaNs with forward fill and safe default
    for col in result_df.columns:
        if any(metric in col for metric in ['price_ma_', 'price_std_', 'price_ema_']):
            result_df[col] = result_df[col].fillna(method='ffill')
            result_df[col] = result_df[col].fillna(0)  # fill initial rows where no data exists

    return result_df.drop(columns=['date'])





def encode_data(dataframe, encoding_method='one_hot', handle_outliers=True, moving_window = True, cyclic_month = True, normal_year = True, normal_price = True, spatial_features = True):
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
    numerical_cols = ['floor_area_sqm', 'avg_floor', 'remaining_lease', 'longitude', 'latitude', 'distance_to_nearest_mrt', 'distance_to_cbd', 'distance_to_nearest_mall',
                      'price_ma_3','price_std_3','price_ema_3','price_ma_6','price_std_6','price_ema_6','price_ma_12','price_std_12','price_ema_12'
                     ]
    
    if normal_price:
        numerical_cols.append('resale_price')
    
    target_col = 'resale_price'

    # Handle outliers in numerical columns if requested
    if handle_outliers:
        for col in numerical_cols:
            if col in df.columns:
                cap_outliers(df, col)
        
    if moving_window:
        df = add_moving_window_features(df)
    
    if cyclic_month:
        cyclic_encode_month(df) 

    if normal_year:
        normalize_year(df) 
        numerical_cols.append('year')
        
    if spatial_features:
        for type in spatial_features:
            if type == "CBD":
                df  = add_cbd(df)
            if type == "MRT":
                df = add_nearest_mrt(df)
            if type == "MALL":    
                df = add_nearest_mall(df)
                
        df = df.drop(columns=df.columns[df.columns.str.startswith('town')])
    else:
        df = df.drop(['longitude', 'latitude'], axis=1)
       
    
    scale_numerical_features(df, numerical_cols)

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