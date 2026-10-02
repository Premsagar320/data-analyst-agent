import pandas as pd

def summarize_df(df, n_rows=5):
    lines = []
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append("Columns and types:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}, missing={df[col].isna().sum()}")
    lines.append(f"First {n_rows} rows:")
    lines.append(df.head(n_rows).to_string(index=False))
    return "\n".join(lines)

if __name__ == "__main__":
    df = pd.read_csv("data/sales.csv")
    print(summarize_df(df))
