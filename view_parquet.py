import pandas as pd
import sys
import os

def view_parquet(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"\n--- Data Summary for: {file_path} ---")
        print(f"Total Rows: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        print("\n--- Column Names ---")
        print(df.columns.tolist())
        
        print("\n--- Last 5 Rows ---")
        print(df.tail().to_string())
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    # Default to AAPL features if no argument provided
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/features/AAPL.parquet'
    view_parquet(path)
