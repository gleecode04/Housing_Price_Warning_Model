

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os

def make_target_linked_tables(df, target="YoY_Growth_12m", outdir="tables_temp"):
   
    print("\nGenerating target-linked tables...")
    os.makedirs(outdir, exist_ok=True)
    
    df = df.copy()


    all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    use_cols = []
    
    for col in all_num_cols:
        if col not in ["RegionID", "SizeRank"] and  col!= target:
            use_cols.append(col)

    df_drop = df.dropna(subset=[target]).reset_index(drop = True)
    if len(df_drop) == 0:
        print("no rows with non-null!")
        return
    
    # various rankings to see correlations. this is to see if we need dimensionality reduction.
    pearson = df_drop[use_cols].corrwith(df_drop[target], method="pearson")
    spearman = df_drop[use_cols].corrwith(df_drop[target], method="spearman")
    X = df_drop[use_cols]
    y = df_drop[target]

    X = X.fillna(X.median()).values
    y = y.values

    filter = np.isfinite(X).all(axis = 1) & np.isfinite(y)
    X_clean = X[filter]
    y_clean = y[filter]
    
    if len(X_clean) > 0:
        mi_vals = mutual_info_regression(X_clean, y_clean, random_state = 42)
        mi = pd.Series(mi_vals, index= use_cols)
    else:
        mi = pd.Series(0.0, index= use_cols)
    
    rankings = pd.DataFrame({
        "feature": use_cols,"pearson_r": pearson.values,
        "spearman_rho": spearman.values,"mutual_info": mi.values
    }).sort_values(["mutual_info", "pearson_r"], ascending=[False, False])
    
    rankings.to_csv(f"{outdir}/target_linked_rankings.csv", index=False)
    candidate_level = []

    for col in use_cols:
        if any(k in col.lower() for k in ["value", "median_sale_price", "mlp", "invt_fs", 
                          "market_temp", "mean_doz"]):
            candidate_level.append(col)
        if len(candidate_level) >= 6:
            break
    if 'Date' in df_drop.columns and 'RegionID' in df_drop.columns:
        df_drop = df_drop.sort_values(['RegionID', 'Date'])
        lags = [1,3, 6, 12] # 1, 3 , 6, 12 month intervals 
        rows = []


        for level in candidate_level:
            if level not in df_drop.columns:
                continue


            row = {"feature": level}
            for L in lags:
                lagged = df_drop.groupby('RegionID')[level].shift(L)
                corr_val = lagged.corr(df_drop[target])
                row[f"lag_{L}m"] = corr_val if not pd.isna(corr_val) else 0.0
            rows.append(row)
        # lag_grid = pd.DataFrame(rows)
        # lag_grid.to_csv(f"{outdir}/lag_corr_grid.csv", index=False)
    if "Date" in df_drop.columns:
        df_drop["year"] = pd.to_datetime(df_drop["Date"]).dt.year
        cov_rows = []
        uniq_years = sorted(df_drop["year"].unique())
        for year in uniq_years:
            year_data = df_drop[df_drop["year"] == year]
            row = {"year": year}
            for col in use_cols:
                if col in year_data.columns:
                    coverage = (1 - year_data[col].isna().mean()) * 100
                    row[col] = round(coverage, 1)
                else:
                    row[col] = 0.0
            cov_rows.append(row)
        # cov = pd.DataFrame(cov_rows)
        # cov.to_csv(f"{outdir}/feature_coverage_by_year.csv", index=False)
        
