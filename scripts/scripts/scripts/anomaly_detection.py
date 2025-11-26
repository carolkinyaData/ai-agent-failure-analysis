import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def isolation_forest_anomaly(df, cols):
    """Apply Isolation Forest to detect anomalies."""
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_iforest"] = model.fit_predict(df[cols])
    return df


def one_class_svm_anomaly(df, cols):
    """Apply One-Class SVM for anomaly detection."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    df["anomaly_ocsvm"] = model.fit_predict(scaled)
    return df


def zscore_anomaly(df, cols, threshold=3):
    """Detect anomalies using Z-score."""
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f"anomaly_z_{col}"] = ((df[col] - mean).abs() > threshold * std).astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/sample_agent_performance.csv")
    numeric_cols = df.select_dtypes(include="number").columns

    df = isolation_forest_anomaly(df, numeric_cols)
    df = one_class_svm_anomaly(df, numeric_cols)
    df = zscore_anomaly(df, numeric_cols)

    print(df.head())
