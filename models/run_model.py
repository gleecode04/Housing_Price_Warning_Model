# -*- coding: utf-8 -*-
"""
Driver script for Random Forest model training and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from train_model import load_data, prepare_features, temporal_split, train_random_forest, save_model

sns.set_style("whitegrid")
MODELS_DIR = Path(__file__).parent
GRAPHS_DIR = MODELS_DIR / 'graphs'
ARTIFACTS_DIR = MODELS_DIR / 'artifacts'
DATA_PATH = MODELS_DIR.parent / 'pre_training' / 'final_table' / 'preprocessed_data.csv'

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def print_performance_report(metrics, feature_importance, X_train, X_test):
   
    print("trainset metrics:")
    print(f" R² Score, RMSE, MAE :  {metrics['train_r2']}, {metrics['train_rmse']}%, {metrics['train_mae']}% ")
    
    print("Test Set:")
    print(f" R² Score, RMSE, MAE :  {metrics['test_r2']}, {metrics['test_rmse']}%, {metrics['test_mae']}% ")    
    print("rest of the results can be found in the graphs directory. check graphs for results")

def main():
    
    print("Random forest training starts: (Data Loading)")
    df = load_data(DATA_PATH)
    print(" feature engineering start:")
    X, y, df_subset = prepare_features(df)

    print("train/test data split")
    X_train, X_test, y_train, y_test, train_mask, test_mask = temporal_split(X, y, df_subset, test_size=0.2)
    print(" Training Random Forest model...")
    model, metrics, feature_importance, y_train_pred, y_test_pred = train_random_forest(
        X_train, y_train, X_test, y_test, tune=True
    )
    # post model analysis  
    print_performance_report(metrics, feature_importance, X_train, X_test)
    
    save_model(model, ARTIFACTS_DIR / 'random_forest_model.pkl')    
    
    # generate_graphs(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred,
    # can activate this later if needed
    print("relevant Graphs generated : (residuals, feature importance, performance metrics etc)")    
    
    print("training is finished")
    return model, metrics, feature_importance

if __name__ == "__main__":
    model, metrics, feature_importance = main()

