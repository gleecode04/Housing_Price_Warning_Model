
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os
from pathlib import Path

def load_data(data_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_features(df):
    
    drop_columns = ['RegionID' , 'Date', 'RegionName', 'StateName', 'YoY_Growth_6m', 'YoY_Growth_12m']
    cols = df.drop(columns=[c for c in drop_columns if c in df.columns])
    cols = cols.select_dtypes(include=[np.number])
    cols= cols.fillna(cols.median())
    y = df['YoY_Growth_12m'].dropna()
    cols= cols.loc[y.index]
    return cols, y, df.loc[y.index]

def temporal_split(X, y, df, test_size=0.2):
    
    df_sorted = df.sort_values('Date')
    split_idx = int(len(df_sorted) * (1 - test_size))
    split_date = df_sorted.iloc[split_idx]['Date']

    train_filter= df_sorted['Date'] < split_date
    train_indices = train_filter.index[train_filter]
    test_indices = train_filter.index[~train_filter]

    X_train,X_test,y_train, y_test  = X.loc[train_indices], X.loc[test_indices], y.loc[train_indices], y.loc[test_indices]
    train_filtered_original = df.index.isin(train_indices)
    test_filtered_original = df.index.isin(test_indices)
    
    return X_train, X_test, y_train, y_test, train_filtered_original, test_filtered_original

def tune_hyperparameters(X_train,y_train, n_iter= 20,cv=3):
    # increased the randomness and complexity to prevent overfitting
    param_grid = {
        'n_estimators': [50, 100,150],  'max_depth': [5, 8, 10, 12],  
        'min_samples_split': [20,30,50,100],'min_samples_leaf': [10,15, 20,30], 
        'max_features': ['log2', 0.3, 0.4, 'sqrt']
    }
    STATE = 42
    JOBS = -1
    base_model = RandomForestRegressor(random_state=STATE, n_jobs=JOBS)
    hyper_tuning = RandomizedSearchCV(
        base_model,param_grid, n_iter=n_iter,cv = cv,
        scoring='r2',n_jobs= -1, random_state= 42,verbose = 0
    )
    hyper_tuning.fit(X_train, y_train)
    
    return hyper_tuning.best_params_,hyper_tuning.best_score_

def train_random_forest(X_train, y_train, X_test, y_test, params=None, tune=False):
    
    # flag to do optional hyperparameter tuning.
    
    print("  Tuning hyperparameters...")
    best_params, best_cv_score = tune_hyperparameters(X_train, y_train, n_iter=20, cv=3)
    params = best_params.copy()
    params['random_state'] = 42
    params['n_jobs'] = -1
    # elif params is None:
    #     params = {
    #         'n_estimators':100,'max_depth': 8,  
    #         'min_samples_split':30,  'min_samples_leaf': 15,  
    #         'max_features': 'log2', 'random_state':42,
    #         'n_jobs': -1
    #     }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_train_res, y_test_res = model.predict(X_train), model.predict(X_test)
    
    r2,rmse,mae = r2_score( y_train, y_train_res), np.sqrt(mean_squared_error(y_train, y_train_res)), mean_absolute_error( y_train,  y_train_res)

    t_r2,t_rmse,t_mae = r2_score(y_test,y_test_res), np.sqrt(mean_squared_error( y_test,y_test_res)), mean_absolute_error(y_test,y_test_res)
    metrics = {
        'train_r2': r2,'test_r2': t_r2,
        'train_rmse': rmse,'test_rmse': t_rmse,
        'train_mae': mae, 'test_mae': t_mae,
        'best_params': params if tune else None
    }
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, metrics, feature_importance, y_train_res, y_test_res

def save_model(model, filepath):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

