import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


def compute_feature_importance(df, target_col):
    """Compute feature importances using Random Forest."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle non-numeric columns
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)


def plot_feature_importance(importance, save_path="visuals/feature_importance_chart.png"):
    """Plot and save feature importance chart."""
    plt.figure(figsize=(8, 6))
    importance.head(15).plot(kind="bar")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/sample_agent_performance.csv")

    # Assume the dataset has a "failed" column (1 = failure, 0 = success)
    importance = compute_feature_importance(df, target_col="failed")

    print("Top features:\n", importance.head())

    plot_feature_importance(importance)
