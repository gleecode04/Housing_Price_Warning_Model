# -*- coding: utf-8 -*-
"""
Generate dataset schema JSON file detailing columns and datatypes for all CSV files.
Provides column array with date ranges collapsed, and sample row data.
"""

import pandas as pd
import json
from pathlib import Path
import os
from datetime import datetime

def is_date_column(col_name):
    """Check if a column name is a date (time-series data)."""
    try:
        # Try to parse as date (format: YYYY-MM-DD)
        datetime.strptime(str(col_name), '%Y-%m-%d')
        return True
    except:
        return False

def get_date_interval(date_columns):
    """Determine the time interval between dates."""
    if len(date_columns) < 2:
        return "monthly"
    
    try:
        dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in date_columns])
        # Check first few intervals
        intervals = []
        for i in range(min(5, len(dates) - 1)):
            diff = dates[i+1] - dates[i]
            intervals.append(diff.days)
        
        # Most common interval
        avg_days = sum(intervals) / len(intervals)
        
        if 28 <= avg_days <= 31:
            return "monthly"
        elif 7 <= avg_days <= 7:
            return "weekly"
        elif 90 <= avg_days <= 92:
            return "quarterly"
        elif 365 <= avg_days <= 366:
            return "yearly"
        else:
            return f"approximately {int(avg_days)} days"
    except:
        return "monthly"

def get_csv_schema(csv_path):
    """Get schema information for a CSV file."""
    try:
        # Read first row for sample data
        df = pd.read_csv(csv_path, nrows=1)
        
        # Separate structural columns from date columns
        structural_cols = []
        date_cols = []
        
        for col in df.columns:
            if is_date_column(col):
                date_cols.append(col)
            else:
                structural_cols.append(col)
        
        # Build columns array
        columns_array = []
        
        # Add structural columns
        for col in structural_cols:
            columns_array.append(col)
        
        # Add date range object if we have date columns
        if date_cols:
            date_cols_sorted = sorted(date_cols)
            interval = get_date_interval(date_cols_sorted)
            
            columns_array.append({
                "date_range": f"{date_cols_sorted[0]} to {date_cols_sorted[-1]}",
                "num_columns": len(date_cols),
                "note": f"This denotes a date range and columns are separated by {interval} intervals"
            })
        
        # Get sample row (first row)
        sample_row = {}
        
        # Add all structural columns
        for col in structural_cols:
            value = df[col].iloc[0]
            # Convert to native Python types for JSON serialization
            if pd.isna(value):
                sample_row[col] = None
            elif isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                sample_row[col] = str(value)
            elif isinstance(value, (int, float)):
                # Check if it's a whole number
                if isinstance(value, float) and value.is_integer():
                    sample_row[col] = int(value)
                else:
                    sample_row[col] = float(value)
            else:
                sample_row[col] = str(value)
        
        # Add only first 2 and last 2 date columns to sample row (to keep it manageable)
        if date_cols:
            date_cols_sorted = sorted(date_cols)
            sample_dates = []
            
            if len(date_cols_sorted) <= 4:
                # If 4 or fewer, include all
                sample_dates = date_cols_sorted
            else:
                # Include first 2 and last 2
                sample_dates = date_cols_sorted[:2] + date_cols_sorted[-2:]
            
            for col in sample_dates:
                value = df[col].iloc[0]
                if pd.isna(value):
                    sample_row[col] = None
                elif isinstance(value, (int, float)):
                    if isinstance(value, float) and value.is_integer():
                        sample_row[col] = int(value)
                    else:
                        sample_row[col] = float(value)
                else:
                    sample_row[col] = str(value)
            
            # Add note about excluded dates
            if len(date_cols_sorted) > 4:
                sample_row["_note"] = f"Sample row shows first 2 and last 2 date columns. Total of {len(date_cols_sorted)} date columns exist."
        
        # Get row count
        try:
            with open(csv_path, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
        except:
            row_count = 'unknown'
        
        schema = {
            'file_path': str(csv_path),
            'file_name': csv_path.name,
            'num_rows': row_count,
            'columns': columns_array,
            'sample_row': sample_row
        }
        
        return schema
    
    except Exception as e:
        return {
            'file_path': str(csv_path),
            'file_name': csv_path.name,
            'error': str(e)
        }

def main():
    """Generate schema for all CSV files."""
    print("Scanning datasets folder for CSV files...")
    
    datasets_dir = Path('datasets/zillow_downloads')
    
    if not datasets_dir.exists():
        print(f"ERROR: {datasets_dir} not found!")
        return
    
    all_schemas = {}
    
    # Walk through all subdirectories
    for category_dir in datasets_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        print(f"\nProcessing category: {category_name}")
        
        csv_files = list(category_dir.glob('*.csv'))
        
        if not csv_files:
            continue
        
        all_schemas[category_name] = {}
        
        for csv_file in csv_files:
            print(f"  Processing {csv_file.name}...")
            schema = get_csv_schema(csv_file)
            all_schemas[category_name][csv_file.name] = schema
            
            num_cols = len(schema.get('columns', []))
            if schema.get('columns') and isinstance(schema['columns'][-1], dict):
                num_cols = len(schema['columns']) - 1 + schema['columns'][-1].get('num_columns', 0)
            
            print(f"    ✓ {num_cols} total columns")
    
    # Save to JSON
    output_file = 'datasets/dataset_schema.json'
    os.makedirs('datasets', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_schemas, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SCHEMA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved: {output_file}")
    
    # Print summary
    total_files = sum(len(files) for files in all_schemas.values())
    print(f"\nSummary:")
    print(f"  Categories: {len(all_schemas)}")
    print(f"  Total CSV files: {total_files}")
    
    for category, files in all_schemas.items():
        print(f"    {category}: {len(files)} files")
    
    return all_schemas

if __name__ == "__main__":
    schemas = main()
