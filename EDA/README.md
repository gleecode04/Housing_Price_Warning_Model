# EDA & Preprocessing Module

This directory contains the Exploratory Data Analysis (EDA) and preprocessing modules for the Zillow Housing Price Prediction project.

## Files

- **`eda.py`**: EDA functions for data loading, analysis, and visualization
- **`preprocessing.py`**: Preprocessing classes for missing value imputation and feature scaling
- **`run_eda.py`**: Main script to run the complete EDA and preprocessing pipeline
- **`report.md`**: Comprehensive report of findings and preprocessing decisions
- **`requirements.txt`**: Python package dependencies

## Usage

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python run_eda.py
```

This will:
- Load all Zillow datasets
- Perform EDA analysis
- Generate visualizations
- Apply preprocessing
- Save processed data and generate report

### Individual Components

#### Run EDA Only
```python
from eda import run_full_eda

results = run_full_eda(
    data_dir='../datasets/zillow_downloads',
    output_dir='./'
)
```

#### Apply Preprocessing Only
```python
from preprocessing import apply_preprocessing_pipeline
import pandas as pd

df = pd.read_csv('your_data.csv')
df_processed, info = apply_preprocessing_pipeline(
    df,
    impute=True,
    scale=True,
    remove_outliers=False
)
```

## Preprocessing Methods

### 1. Missing Value Imputation
- **Class**: `MissingValueImputer`
- **Strategy**: Temporal-aware (forward/backward fill within regions)
- **Fallback**: KNN imputation

### 2. Feature Scaling
- **Class**: `FeatureScaler`
- **Method**: StandardScaler (Z-score normalization)
- **Scope**: All numeric features except IDs and targets

## Output Files

After running the pipeline, you'll get:
- `processed_data.csv`: Preprocessed dataset
- `preprocessing_info.json`: Preprocessing metadata
- `time_series_trends.png`: Time series visualization
- `yoy_growth_distribution.png`: Growth rate distribution
- `correlation_heatmap.png`: Feature correlation matrix

## Report

See `report.md` for detailed findings, preprocessing rationale, and dataset sufficiency assessment.

