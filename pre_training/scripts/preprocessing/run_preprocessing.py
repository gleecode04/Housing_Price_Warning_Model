
import pandas as pd
import numpy as np
import warnings
import os
import json
import sys
from pathlib import Path
import importlib.util
PREPROCESSING_DIR = Path(__file__).parent

utils_path = PREPROCESSING_DIR / "utils.py"
utils_spec = importlib.util.spec_from_file_location("preprocessing_utils", utils_path)
preprocessing_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(preprocessing_utils)

# import the preprocessing util functions.
impute_missing_values = preprocessing_utils.impute_missing_values
add_rolling_features = preprocessing_utils.add_rolling_features
apply_pca_to_price_block = preprocessing_utils.apply_pca_to_price_block

warnings.filterwarnings('ignore')

# Get paths (2 levels up from scripts/preprocessing/ to pre_training/)
PRE_TRAINING_DIR = Path(__file__).parent.parent.parent
FINAL_TABLE_DIR = PRE_TRAINING_DIR / 'final_table'
TABLES_TEMP_DIR = Path(__file__).parent.parent / 'eda' / 'tables_temp'

os.makedirs(FINAL_TABLE_DIR, exist_ok=True)

# This function applies preprocessing 
# input: dataframe
# ouptut : preprocessed dataframe, dictionary of info

def apply_preprocessing(df):
   
    #initial stats for tracking purposes?

    row_before,cols_before = df.shape[0], df.shape[1]
    missing_by_col = df.isnull().sum()
    tot = 0
    
    for val in missing_by_col:
        tot += val
    
    info = {

        'original_shape':[row_before, cols_before],
        'original_missing':int(tot),
        'steps_applied':[], 'rolling_cols':[],
        'pca_cols':[], 'pca_explained_variance': None
    }
    
    print("Imputation")


    df = impute_missing_values(df)
    missing_after = df.isnull().sum()
    after = 0
    for val in missing_after:
        after += val
    info['steps_applied'].append('imputation')
    
    print("Rolling statistical features")
    try:
        path = TABLES_TEMP_DIR / 'target_linked_rankings.csv'
        
        if os.path.exists(path):
            rankings = pd.read_csv(path)
        else:
            raise Exception("rankfile not found")
        # get the top 10 features from the rankings.    
        selected_cols = []
        top_10 = rankings['feature'].head(10)

        for col in top_10:
       
            if any(k in col.lower() for k in ["value", "median_sale_price", "mlp", "invt_fs", 
                                               "market_temp", "mean_doz"]):
                if col in df.columns:
                    selected_cols.append(col)
                    if len(selected_cols) >= 5:
                        break
        if not selected_cols:
            selected_cols = [c for c in df.columns if any(k in c.lower() for k in 
                         ["value", "median_sale_price", "mlp", "invt_fs", "market_temp", "mean_doz"])][:5]
        df = add_rolling_features(df, selected_cols, windows=(3, 6, 12))
        info['rolling_cols'] = selected_cols
        info['steps_applied'].append('rolling_features')
        print(f"added rolling features for {len(selected_cols)} columns")
    except Exception as e:
        print(f"Could not apply rolling features: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    print("PCA")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors= 'coerce')
        split_date = pd.to_datetime('2021-09-01')

        train_cond,test_cond = df['Date'] < split_date, df['Date'] >= split_date

        train_df = df[train_cond].copy()
        test_df = df[test_cond].copy()
    else:
        size = len(df)
        train_size = int(0.8*size)
        train_df = df.iloc[0:train_size].copy()
        test_df = df.iloc[train_size:].copy()
    
    price_like = ["Value", "median_sale_price_now_uc_sfrcondo_month", "mlp_uc_sfrcondo_sm_month"]
    # price_cols = 
    price_cols = [c for c in price_like if c in train_df.columns]
    
    train_df, test_df, pca, used_cols = apply_pca_to_price_block(
        train_df, test_df, price_cols, n_components=1
    )
    
    if pca is not None:
        info['pca_cols'] = used_cols
        info['pca_explained_variance'] = float(pca.explained_variance_ratio_[0])
        info['steps_applied'].append('pca')
    df_final = pd.concat([train_df, test_df], ignore_index=True)
    df_final = df_final.sort_values(['RegionID', 'Date'] if 'Date' in df_final.columns else ['RegionID'])
    
    info['final_shape'] = list(df_final.shape)
    info['final_missing'] = int(df_final.isnull().sum().sum())
    
    return df_final, info
# driver code for the preprocessing pipeline.
def main():
    try:
        path = TABLES_TEMP_DIR / 'eda_processed_data.csv'
        if not os.path.exists(path):
            raise FileNotFoundError("EDA processed data not found")

        df = pd.read_csv(path)
        r_count,c_count = df.shape[0], df.shape[1]
        print(f"eda dataset with {r_count} rows and {c_count} columns loaded")
    except Exception as e:
        print("Could not load input data for processing")
        return None, None
    
    df_preprocessed, info = apply_preprocessing(df)
    
    if df_preprocessed is None or info is None:
        print("Preprocessing failed - no data to save")
        return None, None
    
    # Save final , preprocessed results
    df_preprocessed.to_csv(FINAL_TABLE_DIR / 'preprocessed_data.csv', index=False)
    print("saved the final preprocessed data")
    
    return df_preprocessed, info

if __name__ == "__main__":
    df_preprocessed, info = main()
