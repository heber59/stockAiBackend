import pandas as pd
import sys
import os

def inspect_parquet(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
        print(f"\n--- Inspection for: {file_path} ---")
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Last 5 Rows ---")
        print(df.tail())
        print("\n--- Statistics ---")
        print(df.describe())

        # Check for actual dividend/split events
        if "Dividends" in df.columns:
            div_events = df[df["Dividends"] > 0]
            if not div_events.empty:
                print(f"\n--- Found {len(div_events)} Dividend Events ---")
                print(div_events[["Close", "Dividends"]].tail(10))
            else:
                print("\n--- No non-zero Dividend events found in this file ---")

        if "Stock Splits" in df.columns:
            split_events = df[df["Stock Splits"] > 0]
            if not split_events.empty:
                print(f"\n--- Found {len(split_events)} Stock Split Events ---")
                print(split_events[["Close", "Stock Splits"]].tail(10))
            else:
                print("\n--- No Stock Split events found in this file ---")
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_data.py <path_to_parquet_file>")
        print("Example: python3 inspect_data.py data/raw/AAPL.parquet")
    else:
        inspect_parquet(sys.argv[1])
