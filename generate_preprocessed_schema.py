# -*- coding: utf-8 -*-
"""
Generate dataset schema JSON file for preprocessed data tables.
Provides column array and sample row data.
"""

import pandas as pd
import json
from pathlib import Path
import os

def get_csv_schema(csv_path):
    """Get schema information for a CSV file."""
    try:
        # Read first row for sample data
        df = pd.read_csv(csv_path, nrows=1)
        
        # Build columns array - just list all column names
        columns_array = list(df.columns)
        
        # Get sample row (first row)
        sample_row = {}
        
        for col in df.columns:
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
        
        # Get row count
        try:
            with open(csv_path, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
        except:
            row_count = 'unknown'
        
        # Get column datatypes
        column_info = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            has_nulls = df[col].isna().any()
            column_info[col] = {
                'dtype': dtype,
                'has_nulls': bool(has_nulls)
            }
        
        schema = {
            'file_path': str(csv_path),
            'file_name': csv_path.name,
            'num_rows': row_count,
            'num_columns': len(df.columns),
            'columns': columns_array,
            'column_info': column_info,
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
    """Generate schema for all preprocessed CSV files."""
    print("Scanning preprocessed data folder for CSV files...")
    
    # Look in EDA/results directory
    results_dir = Path('EDA/results')
    
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found!")
        return
    
    all_schemas = {}
    
    # Check preprocessed_data directory
    preprocessed_dir = results_dir / 'preprocessed_data'
    if preprocessed_dir.exists():
        csv_files = list(preprocessed_dir.glob('*.csv'))
        if csv_files:
            all_schemas['preprocessed_data'] = {}
            print(f"\nProcessing preprocessed_data directory...")
            for csv_file in csv_files:
                print(f"  Processing {csv_file.name}...")
                schema = get_csv_schema(csv_file)
                all_schemas['preprocessed_data'][csv_file.name] = schema
                print(f"    ✓ {schema.get('num_columns', 'N/A')} columns, {schema.get('num_rows', 'N/A')} rows")
    
    # Check data directory
    data_dir = results_dir / 'data'
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            all_schemas['data'] = {}
            print(f"\nProcessing data directory...")
            for csv_file in csv_files:
                print(f"  Processing {csv_file.name}...")
                schema = get_csv_schema(csv_file)
                all_schemas['data'][csv_file.name] = schema
                print(f"    ✓ {schema.get('num_columns', 'N/A')} columns, {schema.get('num_rows', 'N/A')} rows")
    
    # Check tables directory (these are summary tables, but include them)
    tables_dir = results_dir / 'tables'
    if tables_dir.exists():
        csv_files = list(tables_dir.glob('*.csv'))
        if csv_files:
            all_schemas['tables'] = {}
            print(f"\nProcessing tables directory...")
            for csv_file in csv_files:
                print(f"  Processing {csv_file.name}...")
                schema = get_csv_schema(csv_file)
                all_schemas['tables'][csv_file.name] = schema
                print(f"    ✓ {schema.get('num_columns', 'N/A')} columns, {schema.get('num_rows', 'N/A')} rows")
    
    # Save to JSON
    output_file = 'EDA/results/preprocessed_data_schema.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_schemas, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SCHEMA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved: {output_file}")
    
    # Print summary
    total_files = sum(len(files) for files in all_schemas.values())
    print(f"\nSummary:")
    print(f"  Directories: {len(all_schemas)}")
    print(f"  Total CSV files: {total_files}")
    
    for directory, files in all_schemas.items():
        print(f"    {directory}: {len(files)} files")
    
    return all_schemas

if __name__ == "__main__":
    schemas = main()

