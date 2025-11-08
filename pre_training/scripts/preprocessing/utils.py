
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# imputes missing values(numeric columns)
# forward fills, back fills,median
# 
def impute_missing_values(df):
    
    imputed = df.copy()
    
    if 'Date' not in imputed.columns or 'RegionID' not in imputed.columns:
        return imputed
    
    imputed = imputed.sort_values(['RegionID', 'Date'])
    
    numeric_cols = imputed.select_dtypes(include =[np.number]).columns
    exclude = ['RegionID', 'YoY_Growth_12m', 'YoY_Growth_6m']
    value_cols = []

    for col in numeric_cols:
        if col not in exclude:
            value_cols.append(col)
    
    for col in value_cols:
        if col not in imputed.columns:
            continue
        
        imputed[col] = imputed.groupby('RegionID')[col].ffill()
        imputed[col] = imputed.groupby('RegionID')[col].bfill()
        if imputed[col].isna().any():
            med = imputed[col].median()
            if pd.notna(med):
                imputed[col] = imputed[col].fillna(med)
            else:
                imputed[col] = imputed[col].fillna(0)    
    return imputed
# rolling mean and std for columns
# these are calculated within each region , sorted by date.
def add_rolling_features(df, cols, windows=(3, 6, 12)):
    
    res = df.copy()
    
    if "RegionID" not in res.columns or "Date" not in res.columns:
        return res
    
    res= res.sort_values(["RegionID", "Date"]).reset_index(drop=True)
    
    for col in cols:
        if col not in res.columns:
            continue
        grp = res.groupby("RegionID")[col]
        shifted = grp.shift(1)
        
        for w in windows:
            min_periods = max(1, w//2)
            mean_col = f"{col}_rollmean_{w}"
            std_col=f"{col}_rollstd_{w}"
            res[mean_col] = shifted.rolling(w, min_periods = min_periods).mean()
            res[std_col] = shifted.rolling(w, min_periods = min_periods).std()
    
    return res

# this will apply pca to a section of price-related features
# we use the k_fold pca for optimized performance
def apply_pca_to_price_block(train_df, test_df, price_cols, n_components=1):
    cols= []
    for col in price_cols:
        if col in train_df.columns:
            cols.append(col)
    
    
    if len(cols) < 2:
        return train_df, test_df, None, []
    
    X_train = train_df[cols].replace([ np.inf , -np.inf], np.nan)
    X_train= X_train.fillna(X_train.median())
    
    pca = PCA(n_components = n_components, random_state=42)
    pca.fit(X_train)
    
    train_res = train_df.copy()
    test_res= test_df.copy()
    
    for df in (train_res, test_res):
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        comps= pca.transform(X)
        df[f"price_pc1"] = comps[:, 0]
        df.drop(columns=cols, errors='ignore', inplace= True)
    
    return train_res, test_res, pca, cols
