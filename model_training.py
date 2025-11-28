# -*- coding: utf-8 -*-
"""
Model Training for Zillow Housing Price Prediction

This script implements multiple models for both classification and regression tasks:
- Elastic Net (Regression and Classification)
- Random Forest (Classification and Regression)
- XGBoost (Classification and Regression)
- Additional models for comparison

Following standard ML practices:
- Temporal train/test split
- Feature engineering
- Hyperparameter tuning
- Overfitting prevention
- Comprehensive evaluation
"""

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('models/results', exist_ok=True)
os.makedirs('models/artifacts', exist_ok=True)
os.makedirs('models/tables', exist_ok=True)

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = f"Error occurred: {str(error_message)}"
    
    def __str__(self):
        return self.error_message

def save_object(file_path, obj):
    """Save object to pickle file."""
    import pickle
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load object from pickle file."""
    import pickle
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def create_features(df):
    """Create additional features for modeling."""
    df = df.copy()
    
    # Extract temporal features
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    
    # Create lag features (if we have enough history)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['RegionID', 'YoY_Growth_12m', 'YoY_Growth_6m']]
    
    # Sort by RegionID and Date for lag computation
    df = df.sort_values(['RegionID', 'Date'])
    
    # Create 1-month and 3-month lags for key features
    for col in ['Value']:  # Only lag the main price feature to avoid too many features
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('RegionID')[col].shift(1)
            df[f'{col}_lag3'] = df.groupby('RegionID')[col].shift(3)
    
    # Drop rows where lag features are NaN (first few months of each region)
    lag_cols = [f'Value_lag1', f'Value_lag3']
    existing_lag_cols = [col for col in lag_cols if col in df.columns]
    if existing_lag_cols:
        df = df.dropna(subset=existing_lag_cols)
    
    return df

def prepare_data(df, task_type='regression'):
    """Prepare data for modeling."""
    print(f"\nPreparing data for {task_type} task...")
    
    # Create features
    df = create_features(df)
    
    # Drop non-feature columns
    drop_cols = ['RegionID', 'Date', 'RegionName', 'StateName', 'YoY_Growth_6m']
    
    if task_type == 'regression':
        target_col = 'YoY_Growth_12m'
    else:  # classification
        target_col = 'YoY_Growth_12m'
        # Create binary target: top quintile (top 20%)
        df['top_quintile'] = (df['YoY_Growth_12m'] >= df['YoY_Growth_12m'].quantile(0.8)).astype(int)
        target_col = 'top_quintile'
        drop_cols.append('YoY_Growth_12m')  # Keep original for reference but don't use as feature
    
    # Remove rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Separate features and target
    X = df.drop(columns=drop_cols + [target_col], errors='ignore')
    y = df[target_col]
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target distribution:")
    if task_type == 'classification':
        print(f"    Top quintile: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")
        print(f"    Bottom 80%: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
    else:
        print(f"    Mean: {y.mean():.2f}%")
        print(f"    Std: {y.std():.2f}%")
        print(f"    Range: [{y.min():.2f}%, {y.max():.2f}%]")
    
    return X, y, df

def temporal_split(X, y, df, test_size=0.2):
    """Split data temporally (not randomly)."""
    # Use date-based split
    df_with_target = df.copy()
    df_with_target['target'] = y.values
    
    # Sort by date
    df_with_target = df_with_target.sort_values('Date')
    
    # Find split point
    split_idx = int(len(df_with_target) * (1 - test_size))
    split_date = df_with_target.iloc[split_idx]['Date']
    
    train_mask = df_with_target['Date'] < split_date
    test_mask = df_with_target['Date'] >= split_date
    
    X_train = X[train_mask.values]
    X_test = X[test_mask.values]
    y_train = y[train_mask.values]
    y_test = y[test_mask.values]
    
    print(f"\nTemporal Split:")
    print(f"  Train: {X_train.shape[0]} samples ({df_with_target[train_mask]['Date'].min()} to {df_with_target[train_mask]['Date'].max()})")
    print(f"  Test: {X_test.shape[0]} samples ({df_with_target[test_mask]['Date'].min()} to {df_with_target[test_mask]['Date'].max()})")
    
    return X_train, X_test, y_train, y_test

@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str
    task_type: str  # 'regression' or 'classification'
    train_model: bool = False  # Whether to actually train (set True for one model)

class ModelTrainer:
    def __init__(self):
        self.results = {}
    
    def train_elastic_net_regression(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train Elastic Net for regression."""
        print("\n" + "="*70)
        print("Elastic Net Regression")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            # Use ElasticNetCV for automatic hyperparameter tuning
            print("  Training with cross-validation for hyperparameter tuning...")
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                alphas=np.logspace(-4, 1, 50),
                cv=5,
                max_iter=2000,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluation
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            print(f"\n  Results:")
            print(f"    Train R²: {train_r2:.4f}")
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Train RMSE: {train_rmse:.4f}%")
            print(f"    Test RMSE: {test_rmse:.4f}%")
            print(f"    Train MAE: {train_mae:.4f}%")
            print(f"    Test MAE: {test_mae:.4f}%")
            print(f"    Best alpha: {model.alpha_:.6f}")
            print(f"    Best l1_ratio: {model.l1_ratio_:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_r2 - test_r2 > 0.1:
                overfit_warning = "⚠️  Potential overfitting detected (train R² >> test R²)"
            elif test_r2 < 0.3:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model
            save_object('models/artifacts/elastic_net_regression.pkl', model)
            
            return {
                'model': model,
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'best_alpha': float(model.alpha_),
                'best_l1_ratio': float(model.l1_ratio_),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train_elastic_net_classification(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train Elastic Net for classification."""
        print("\n" + "="*70)
        print("Elastic Net Classification")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV
            
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print("  Training with cross-validation for hyperparameter tuning...")
            # Use LogisticRegression with elasticnet (l1_ratio parameter)
            base_model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                max_iter=2000,
                random_state=42,
                n_jobs=-1
            )
            
            # Grid search for best l1_ratio and C
            param_grid = {
                'C': np.logspace(-4, 2, 10),
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
            }
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluation
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            # AUC-ROC
            try:
                train_auc = roc_auc_score(y_train, y_train_proba)
                test_auc = roc_auc_score(y_test, y_test_proba)
            except:
                train_auc = 0.0
                test_auc = 0.0
            
            print(f"\n  Results:")
            print(f"    Train Accuracy: {train_acc:.4f}")
            print(f"    Test Accuracy: {test_acc:.4f}")
            print(f"    Train Precision: {train_precision:.4f}")
            print(f"    Test Precision: {test_precision:.4f}")
            print(f"    Train Recall: {train_recall:.4f}")
            print(f"    Test Recall: {test_recall:.4f}")
            print(f"    Train F1: {train_f1:.4f}")
            print(f"    Test F1: {test_f1:.4f}")
            print(f"    Train AUC-ROC: {train_auc:.4f}")
            print(f"    Test AUC-ROC: {test_auc:.4f}")
            print(f"    Best C: {model.C:.6f}")
            print(f"    Best l1_ratio: {model.l1_ratio:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_acc - test_acc > 0.15:
                overfit_warning = "⚠️  Potential overfitting detected (train acc >> test acc)"
            elif test_acc < 0.6:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model and scaler
            save_object('models/artifacts/elastic_net_classification.pkl', model)
            save_object('models/artifacts/elastic_net_classification_scaler.pkl', scaler)
            
            return {
                'model': model,
                'scaler': scaler,
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'train_precision': float(train_precision),
                'test_precision': float(test_precision),
                'train_recall': float(train_recall),
                'test_recall': float(test_recall),
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_auc': float(train_auc),
                'test_auc': float(test_auc),
                'best_C': float(model.C),
                'best_l1_ratio': float(model.l1_ratio),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train_random_forest_regression(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train Random Forest for regression."""
        print("\n" + "="*70)
        print("Random Forest Regression")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            print("  Training Random Forest...")
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluation
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Results:")
            print(f"    Train R²: {train_r2:.4f}")
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Train RMSE: {train_rmse:.4f}%")
            print(f"    Test RMSE: {test_rmse:.4f}%")
            print(f"    Train MAE: {train_mae:.4f}%")
            print(f"    Test MAE: {test_mae:.4f}%")
            print(f"\n  Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_r2 - test_r2 > 0.15:
                overfit_warning = "⚠️  Potential overfitting detected (train R² >> test R²)"
            elif test_r2 < 0.3:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model
            save_object('models/artifacts/random_forest_regression.pkl', model)
            feature_importance.to_csv('models/tables/random_forest_feature_importance.csv', index=False)
            
            return {
                'model': model,
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'feature_importance': feature_importance.to_dict('records'),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train_random_forest_classification(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train Random Forest for classification."""
        print("\n" + "="*70)
        print("Random Forest Classification")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            print("  Training Random Forest...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluation
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            # AUC-ROC
            try:
                train_auc = roc_auc_score(y_train, y_train_proba)
                test_auc = roc_auc_score(y_test, y_test_proba)
            except:
                train_auc = 0.0
                test_auc = 0.0
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Results:")
            print(f"    Train Accuracy: {train_acc:.4f}")
            print(f"    Test Accuracy: {test_acc:.4f}")
            print(f"    Train Precision: {train_precision:.4f}")
            print(f"    Test Precision: {test_precision:.4f}")
            print(f"    Train Recall: {train_recall:.4f}")
            print(f"    Test Recall: {test_recall:.4f}")
            print(f"    Train F1: {train_f1:.4f}")
            print(f"    Test F1: {test_f1:.4f}")
            print(f"    Train AUC-ROC: {train_auc:.4f}")
            print(f"    Test AUC-ROC: {test_auc:.4f}")
            print(f"\n  Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_acc - test_acc > 0.15:
                overfit_warning = "⚠️  Potential overfitting detected (train acc >> test acc)"
            elif test_acc < 0.6:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model
            save_object('models/artifacts/random_forest_classification.pkl', model)
            feature_importance.to_csv('models/tables/random_forest_classification_feature_importance.csv', index=False)
            
            return {
                'model': model,
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'train_precision': float(train_precision),
                'test_precision': float(test_precision),
                'train_recall': float(train_recall),
                'test_recall': float(test_recall),
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_auc': float(train_auc),
                'test_auc': float(test_auc),
                'feature_importance': feature_importance.to_dict('records'),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train_xgboost_regression(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train XGBoost for regression."""
        print("\n" + "="*70)
        print("XGBoost Regression")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            print("  Training XGBoost...")
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,  # Limit depth to prevent overfitting
                learning_rate=0.1,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluation
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Results:")
            print(f"    Train R²: {train_r2:.4f}")
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Train RMSE: {train_rmse:.4f}%")
            print(f"    Test RMSE: {test_rmse:.4f}%")
            print(f"    Train MAE: {train_mae:.4f}%")
            print(f"    Test MAE: {test_mae:.4f}%")
            print(f"\n  Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_r2 - test_r2 > 0.15:
                overfit_warning = "⚠️  Potential overfitting detected (train R² >> test R²)"
            elif test_r2 < 0.3:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model
            save_object('models/artifacts/xgboost_regression.pkl', model)
            feature_importance.to_csv('models/tables/xgboost_regression_feature_importance.csv', index=False)
            
            return {
                'model': model,
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'feature_importance': feature_importance.to_dict('records'),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def train_xgboost_classification(self, X_train, y_train, X_test, y_test, train_it=False):
        """Train XGBoost for classification."""
        print("\n" + "="*70)
        print("XGBoost Classification")
        print("="*70)
        
        if not train_it:
            print("  (Skipping training - model implemented but not trained)")
            return None
        
        try:
            print("  Training XGBoost...")
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,  # Limit depth to prevent overfitting
                learning_rate=0.1,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=4,  # Handle class imbalance (20% vs 80%)
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluation
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            # AUC-ROC
            try:
                train_auc = roc_auc_score(y_train, y_train_proba)
                test_auc = roc_auc_score(y_test, y_test_proba)
            except:
                train_auc = 0.0
                test_auc = 0.0
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Results:")
            print(f"    Train Accuracy: {train_acc:.4f}")
            print(f"    Test Accuracy: {test_acc:.4f}")
            print(f"    Train Precision: {train_precision:.4f}")
            print(f"    Test Precision: {test_precision:.4f}")
            print(f"    Train Recall: {train_recall:.4f}")
            print(f"    Test Recall: {test_recall:.4f}")
            print(f"    Train F1: {train_f1:.4f}")
            print(f"    Test F1: {test_f1:.4f}")
            print(f"    Train AUC-ROC: {train_auc:.4f}")
            print(f"    Test AUC-ROC: {test_auc:.4f}")
            print(f"\n  Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Check for overfitting
            overfit_warning = ""
            if train_acc - test_acc > 0.15:
                overfit_warning = "⚠️  Potential overfitting detected (train acc >> test acc)"
            elif test_acc < 0.6:
                overfit_warning = "⚠️  Low test performance - model may need more features or tuning"
            
            if overfit_warning:
                print(f"    {overfit_warning}")
            
            # Save model
            save_object('models/artifacts/xgboost_classification.pkl', model)
            feature_importance.to_csv('models/tables/xgboost_classification_feature_importance.csv', index=False)
            
            return {
                'model': model,
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'train_precision': float(train_precision),
                'test_precision': float(test_precision),
                'train_recall': float(train_recall),
                'test_recall': float(test_recall),
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_auc': float(train_auc),
                'test_auc': float(test_auc),
                'feature_importance': feature_importance.to_dict('records'),
                'overfit_warning': overfit_warning
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None

def main():
    """Main training pipeline."""
    print("="*70)
    print("MODEL TRAINING - ZILLOW HOUSING PRICE PREDICTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    try:
        df = pd.read_csv('EDA/preprocessed_data/preprocessed_data.csv')
        print(f"✓ Loaded: {df.shape}")
    except FileNotFoundError:
        print("ERROR: Preprocessed data not found!")
        print("Please run EDA preprocessing first.")
        return
    
    all_results = {}
    
    # REGRESSION TASK
    print("\n" + "="*70)
    print("REGRESSION TASK: Predicting YoY Growth Percentage")
    print("="*70)
    
    X_reg, y_reg, df_reg = prepare_data(df, task_type='regression')
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = temporal_split(X_reg, y_reg, df_reg, test_size=0.2)
    
    trainer = ModelTrainer()
    
    # Implement all models but only train one
    # Train Random Forest Regression (good balance of performance and interpretability)
    result_rf_reg = trainer.train_random_forest_regression(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg, train_it=True
    )
    all_results['random_forest_regression'] = result_rf_reg
    
    # Implement but don't train others
    trainer.train_elastic_net_regression(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg, train_it=False
    )
    trainer.train_xgboost_regression(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg, train_it=False
    )
    
    # CLASSIFICATION TASK
    print("\n" + "="*70)
    print("CLASSIFICATION TASK: Predicting Top Quintile")
    print("="*70)
    
    X_clf, y_clf, df_clf = prepare_data(df, task_type='classification')
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = temporal_split(X_clf, y_clf, df_clf, test_size=0.2)
    
    # Implement all models but don't train (already trained one for regression)
    trainer.train_elastic_net_classification(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf, train_it=False
    )
    trainer.train_random_forest_classification(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf, train_it=False
    )
    trainer.train_xgboost_classification(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf, train_it=False
    )
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create results summary
    if result_rf_reg:
        results_summary = {
            'model_trained': 'Random Forest Regression',
            'task': 'Regression',
            'metrics': {
                'train_r2': result_rf_reg['train_r2'],
                'test_r2': result_rf_reg['test_r2'],
                'train_rmse': result_rf_reg['train_rmse'],
                'test_rmse': result_rf_reg['test_rmse'],
                'train_mae': result_rf_reg['train_mae'],
                'test_mae': result_rf_reg['test_mae']
            },
            'overfit_warning': result_rf_reg.get('overfit_warning', ''),
            'top_features': result_rf_reg.get('feature_importance', [])[:10],
            'models_implemented': [
                'Elastic Net Regression',
                'Random Forest Regression (TRAINED)',
                'XGBoost Regression',
                'Elastic Net Classification',
                'Random Forest Classification',
                'XGBoost Classification'
            ]
        }
        
        with open('models/results/training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Create results table
        results_table = pd.DataFrame([
            ['Train R²', f"{result_rf_reg['train_r2']:.4f}"],
            ['Test R²', f"{result_rf_reg['test_r2']:.4f}"],
            ['Train RMSE', f"{result_rf_reg['train_rmse']:.4f}%"],
            ['Test RMSE', f"{result_rf_reg['test_rmse']:.4f}%"],
            ['Train MAE', f"{result_rf_reg['train_mae']:.4f}%"],
            ['Test MAE', f"{result_rf_reg['test_mae']:.4f}%"],
            ['Overfitting Check', result_rf_reg.get('overfit_warning', 'None detected')]
        ], columns=['Metric', 'Value'])
        
        results_table.to_csv('models/tables/model_results.csv', index=False)
        
        print("✓ Saved: models/results/training_results.json")
        print("✓ Saved: models/tables/model_results.csv")
        print("✓ Saved: models/artifacts/random_forest_regression.pkl")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return all_results

if __name__ == "__main__":
    results = main()
