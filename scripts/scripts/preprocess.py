import pandas as pd

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_columns(df):
    """Standardize column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def fill_missing(df):
    """Handle missing values."""
    for col in df.select_dtypes(include='number'):
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object'):
        df[col].fillna("Unknown", inplace=True)
    return df

def preprocess(filepath):
    """Full preprocessing pipeline."""
    df = load_data(filepath)
    df = clean_columns(df)
    df = fill_missing(df)
    return df

if __name__ == "__main__":
    df = preprocess("data/sample_agent_performance.csv")
    print(df.head())
