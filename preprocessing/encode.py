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
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Round to 0 to avoid e-17 values
    df['month_cos'] = df['month_cos'].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    df.drop(columns=['month'], inplace=True)


def scale_numerical_features(df, numerical_cols):
    """
    Standardize numerical features.
    
    Args:
        df (pandas.DataFrame): DataFrame with numerical columns
        numerical_cols (list): List of numerical column names
        
    Returns:
        None: Modifies dataframe in-place
    """
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


def cap_outliers(df, column, quantile=0.99):
    """
    Cap outliers at a specified quantile.
    
    Args:
        df (pandas.DataFrame): DataFrame with column to cap
        column (str): Column name to cap
        quantile (float): Quantile value for capping (default: 0.99)
        
    Returns:
        None: Modifies dataframe in-place
    """
    upper_limit = df[column].quantile(quantile)
    df[column] = df[column].clip(upper=upper_limit)


def one_hot_encode(df, categorical_cols):
    """
    Apply one-hot encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        categorical_cols (list): List of categorical column names
        
    Returns:
        pandas.DataFrame: DataFrame with one-hot encoded columns
    """
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def target_encode(df, cols, target_col):
    """
    Apply target encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        cols (list): List of categorical column names
        target_col (str): Target column name for encoding
        
    Returns:
        None: Modifies dataframe in-place
    """
    for categorical_col in cols:
        mean_values = df.groupby(categorical_col)[target_col].mean()
        df[categorical_col] = df[categorical_col].map(mean_values)
        df.drop(columns=[categorical_col], inplace=True)


def frequency_encode(df, cols):
    """
    Apply frequency encoding to categorical columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical columns
        cols (list): List of categorical column names
        
    Returns:
        None: Modifies dataframe in-place
    """
    for categorical_col in cols:
        freq_values = df[categorical_col].value_counts(normalize=True)
        df[categorical_col] = df[categorical_col].map(freq_values)
        df.drop(columns=[categorical_col], inplace=True)


def encode_data(dataframe):
    """
    Encode and transform the cleaned data.
    
    Args:
        dataframe (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        pandas.DataFrame: Encoded DataFrame
    """
    df = dataframe.copy()

    categorical_cols = ['town', 'flat_type', 'flat_model']
    numerical_cols = ['floor_area_sqm', 'avg_floor', 'remaining_lease', 'price_per_sqm']
    
    # Add derived features
    df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

    # Encode categorical cols
    # Note: Only use one of the methods at a once, can tune according to your model performance
    df = one_hot_encode(df, categorical_cols)
    # target_encode(df, categorical_cols, 'resale_price')
    # frequency_encode(df, categorical_cols)
    
    # Normalize numerical cols
    cap_outliers(df, 'floor_area_sqm')
    cap_outliers(df, 'resale_price')

    # Note - following two scaling for datetime are experimental, comment out either if its affecting performance
    cyclic_encode_month(df)
    numerical_cols.append('year')

    scale_numerical_features(df, numerical_cols)
    return df