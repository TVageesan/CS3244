import pandas as pd
import os

def read_csv(path):
    if not os.path.exists(path):
        raise Exception("File path: ", path, " not found.")
    
    return pd.read_csv(path)

def merge_csv(file_paths):
    dataframes = [read_csv(file_path) for file_path in file_paths]
    return pd.concat(dataframes, ignore_index=True)