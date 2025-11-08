
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import make_target_linked_tables
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

PRE_TRAINING_DIR = Path(__file__).parent.parent.parent
IMAGES_DIR = PRE_TRAINING_DIR / 'images'
TABLES_TEMP_DIR = Path(__file__).parent / 'tables_temp'
DATA_DIR = PRE_TRAINING_DIR.parent / 'datasets' / 'zillow_downloads'

os.makedirs(IMAGES_DIR, exist_ok= True)
os.makedirs(TABLES_TEMP_DIR, exist_ok = True)

def merge_datasets(data_dir=None):
    
    data_dir = Path(data_dir)
    
    
    key_files = {
        'home_value_index': ['Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'],
        'sales': [
            'Metro_median_sale_price_now_uc_sfrcondo_month.csv',
            'Metro_mean_sale_to_list_uc_sfrcondo_sm_month.csv',
        ],
        'for_sale_listings': [
            'Metro_mlp_uc_sfrcondo_sm_month.csv',
            'Metro_invt_fs_uc_sfrcondo_sm_month.csv',
        ],
        'days_on_market': ['Metro_mean_doz_pending_uc_sfrcondo_sm_month.csv'],
        'market_heat': ['Metro_market_temp_index_uc_sfrcondo_month.csv']
    }
    
    all_data = {}
    for category, file_list in key_files.items():
        cat_folder = data_dir / category
        if not cat_folder.exists():
            continue # maybe log edge case?
        
        loaded_files = []
        for filename in file_list:
            file_path = cat_folder / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = filename
                    df['category'] = category
                    loaded_files.append(df)
                except Exception as e:
                    print("couldnt read file")
        
        if loaded_files:
            all_data[category] = loaded_files
    
    return all_data

def pivot_to_long(df):
    
    meta_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 
                     'StateName', 'source_file', 'category']
    new_cols = []
    for column in meta_cols:
        if column in df.columns:
            new_cols.append(column)
    meta_cols = new_cols
    date_cols = []
    for col in df.columns:
        if col not in meta_cols:
            try:
                pd.to_datetime(col)
                date_cols.append(col)
            except:
                continue
    
    if not date_cols:
        return pd.DataFrame()
    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=date_cols,
        var_name='Date',
        value_name='Value'
    )
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long = df_long.dropna(subset=['Value'])
    return df_long

def compute_yoy_growth(df):
    df = df.sort_values(by = ['RegionID', 'Date']).copy()


    df['Value_12m_ago'] = df.groupby('RegionID')['Value'].shift(periods = 12)
    df['Value_6m_ago'] = df.groupby('RegionID')['Value'].shift(6)

    def calc_grwoth(current,prev):
        if pd.isna(prev) or prev == 0:
            return np.nan
        return (current-prev)/prev * 100

    df['YoY_Growth_12m'] = df.apply(lambda row: calc_grwoth(row['Value'], row['Value_12m_ago']), axis = 1)
    df['YoY_Growth_6m'] = df.apply(lambda row: calc_grwoth(row['Value'], row['Value_6m_ago']), axis = 1)
    return df

def create_feature_matrix(datasets):
    
    home_value_data = datasets.get('home_value_index', [])
    if len(home_value_data) == 0:
        return pd.DataFrame()
    
    base_df = home_value_data[0]
    reshape = pivot_to_long(base_df)
    final = compute_yoy_growth(reshape)
    
    base_features = final[['RegionID','Date', 'Value', 'YoY_Growth_12m', 
                          'YoY_Growth_6m','RegionName', 'StateName']].copy()
    feature_matrix = base_features.copy()
    
    for category,data_list in datasets.items():
        if category == 'home_value_index':
            continue
        
        for file in data_list:
            df_long = pivot_to_long(file)
            if df_long.empty:
                continue

            criteria = ['RegionID','Date']
            raw_file = file['source_file'].iloc[0]
            feature_col = raw_file.replace('.csv' , '')
            if feature_col.startswith('Metro_'):
                feature_name = feature_col.replace('Metro_', '')

            grouped = df_long.groupby(criteria)['Value'].mean()
            grouped = grouped.rename(feature_name).reset_index()
            feature_matrix = pd.merge(feature_matrix, grouped, how='left', on=criteria)
    return feature_matrix

def generate_visualizations(df):

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'RegionID' in numeric_columns:
        numeric_columns.remove('RegionID')
    if 'SizeRank' in numeric_columns:
        numeric_columns.remove('SizeRank')
    
    sample_cols = numeric_columns[:13]
    corr= df[sample_cols].corr()

    plt.figure()
    sns.heatmap(corr, annot=True, 
    fmt='.2f', cmap='coolwarm', 
    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.title("Feature Correlation Matrix")

    path = IMAGES_DIR / 'correlation_heatmap.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
def main():
    datasets = merge_datasets(DATA_DIR)
        
    df = create_feature_matrix(datasets)
    # missing_dict, corr_pairs = check_data_quality(df)
    make_target_linked_tables(df, target="YoY_Growth_12m", outdir=str(TABLES_TEMP_DIR))
    # graphs for report purposes?
    # generate_visualizations(df)
    
    df.to_csv(TABLES_TEMP_DIR / 'eda_processed_data.csv', index=False)
    
    return df

if __name__ == "__main__":
    df = main()
