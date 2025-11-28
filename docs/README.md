# Housing Price Warning Model

A machine learning system for predicting year-over-year housing price growth at the metropolitan level using time-series data from Zillow.

## Problem Statement

Predict which metropolitan areas will experience significant housing price appreciation or depreciation 6-12 months ahead. The model forecasts the **year-over-year (YoY) growth percentage** in home values, enabling policymakers and stakeholders to anticipate market shifts and design effective responses.

## Dataset

Seven Zillow datasets integrated to capture housing market dynamics:
- **Home Value Index**: Primary target variable source (middle-tier market segment)
- **Sales Data**: Transaction prices and sale-to-list ratios
- **For-Sale Listings**: Median listing prices and inventory levels
- **Days on Market**: Market velocity indicator
- **Market Heat Index**: Composite indicator of market activity

**Final Dataset**: 227,289 observations across 895 metropolitan regions (January 2000 - September 2025)

## Methods

### 1. Exploratory Data Analysis
- Data integration and time-series transformation
- Missing value analysis (32.9% missing rate, temporal patterns)
- Multicollinearity detection (correlations >0.95 among price features)
- Target-linked feature analysis with lag correlation

### 2. Preprocessing
- **Temporal-aware imputation**: Forward fill → backward fill → median fallback (reduced missing values by 98.3%)
- **Rolling statistics features**: 3, 6, and 12-month rolling means and standard deviations for momentum/volatility signals
- **PCA on price block**: Dimensionality reduction to eliminate multicollinearity (89.3% variance captured)

### 3. Model Training
- **Algorithm**: Random Forest Regressor
- **Target**: YoY_Growth_12m (12-month percentage change)
- **Train/Test Split**: Temporal (chronological) to prevent data leakage
- **Hyperparameters**: Tuned via grid search
  - n_estimators: 50
  - min_samples_split: 50
  - min_samples_leaf: 15
  - max_features: 0.4
  - max_depth: 12

## Results

### Performance Metrics
- **Test R²**: 0.5865
- **Test RMSE**: 4.54%
- **Test MAE**: 2.45%
- **Overfitting**: Minimal (R² gap = 0.0451)

### Key Insights
Top predictive features are rolling statistics capturing market momentum and volatility:
- Value rolling standard deviation (12-month): 20.1%
- Value rolling mean (12-month): 14.4%
- Market temperature index volatility: 4.1%

## Project Structure

```
├── pre_training/          # EDA and preprocessing pipeline
│   ├── scripts/
│   │   ├── eda/          # Exploratory data analysis
│   │   └── preprocessing/ # Data preprocessing
│   └── final_table/       # Preprocessed dataset
├── models/                # Model training and evaluation
│   ├── train_model.py    # Training script
│   ├── graphs/           # Performance visualizations
│   └── versions/         # Saved model artifacts
└── datasets/             # Raw Zillow datasets
```

## Usage

### Run Preprocessing Pipeline
```bash
cd pre_training
python run_pipeline.py
```

### Train Model
```bash
cd models
python train_model.py
```

## Requirements

See `pre_training/requirements.txt` for dependencies.



## Key Features

- **Temporal modeling**: Chronological train/test split prevents data leakage
- **Feature engineering**: Rolling statistics capture market momentum and volatility
- **Robust preprocessing**: Handles 32.9% missing data with temporal-aware imputation
- **Multicollinearity handling**: PCA reduces redundant price features
- **Comprehensive evaluation**: Multiple metrics and visualizations

## Limitations & Future Work

### Current Limitations
- **Model complexity**: Random Forest captures nonlinear relationships but may miss long-term temporal dependencies inherent in housing markets
- **Feature representation**: Rolling statistics provide momentum signals but don't explicitly model sequential patterns
- **Missing data**: Despite temporal imputation, 32.9% initial missing rate may introduce bias, particularly in early time periods
- **Spatial relationships**: Current model treats metropolitan areas independently, missing potential spatial autocorrelation effects

### Future Implementations
- **LSTM Networks**: Implement Long Short-Term Memory networks to capture long-range temporal dependencies and sequential patterns in housing price dynamics. LSTMs can learn complex time-series relationships that traditional feature engineering may miss.
- **XGBoost**: Explore gradient boosting methods (XGBoost) for improved predictive performance through ensemble learning and better handling of feature interactions. XGBoost's regularization capabilities may reduce overfitting and improve generalization.
- **Hybrid Models**: Combine LSTM for temporal patterns with XGBoost for feature interactions, potentially achieving superior performance by leveraging strengths of both approaches.
- **Spatial Modeling**: Incorporate spatial econometric techniques to capture regional spillover effects and neighborhood-level dependencies.


## Model Artifacts

- Trained model: `models/versions/random_forest_model.pkl`
- Performance visualizations: `models/graphs/`
- Feature importance rankings available in model object
