import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
import numpy as np


def get_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv("assets/data.csv")
    features = data.drop(columns=["churn", "customer_id"])
    target = data.churn
    return features, target


def create_features(features: pd.DataFrame) -> pd.DataFrame:
    features["product_mean_balance"] = features.apply(
        lambda row: row["balance"] / row["products_number"], axis=1
    )
    features["balance_to_salary"] = features["balance"] / features["estimated_salary"]
    features["risk_age_rank"] = pd.cut(
        features["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["30대이하", "30대", "40대", "50대", "60대이상"],
        right=True,
        include_lowest=True,
    )
    return features


def get_preprocessor(features: pd.DataFrame, target: pd.Series) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["country"]),
            (
                "ordinal_gender",
                OrdinalEncoder(categories=[["Male", "Female"]]),
                ["gender"],
            ),
            (
                "onehot_age_rank",
                OneHotEncoder(handle_unknown="ignore"),
                ["risk_age_rank"],
            ),
            # ("scale", StandardScaler(), ["estimated_salary", "balance"]),
        ],
        remainder="passthrough",
    )

    return preprocessor


def split_data(encoded_features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features, target, test_size=0.25, random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def fold_split_data(X, y):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    return kf.split(X, y)
