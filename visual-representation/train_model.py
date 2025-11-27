
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data") / "blocket_cars.csv"


def dat(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function based on your assignment notebook.

    - Price: keep in [10_000, 3_000_000]
    - Mileage: remove 0 and > 50_000
    - Horsepowers: remove > 1_000
    - Drop rows with missing Price, Mileage, Transmission, Horsepowers

    Note: the drops are kept inside the loop to match the original logic.
    """
    df_raw = df_raw.copy()
    for col in df_raw.columns:
        if col == "Price":
            df_raw[col] = df_raw[col].apply(
                lambda x: None if x < 10_000 else (None if x > 3_000_000 else x)
            )
        elif col == "Mileage":
            df_raw[col] = df_raw[col].apply(
                lambda x: None if x == 0 else (None if x > 50_000 else x)
            )
        elif col == "Horsepowers":
            df_raw[col] = df_raw[col].apply(
                lambda x: None if x > 1_000 else x
            )
        else:
            pass

        df_raw = df_raw.dropna(subset=["Price"]).reset_index(drop=True)
        df_raw = df_raw.dropna(subset=["Mileage"]).reset_index(drop=True)
        df_raw = df_raw.dropna(subset=["Transmission"]).reset_index(drop=True)
        df_raw = df_raw.dropna(subset=["Horsepowers"]).reset_index(drop=True)

    return df_raw


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put blocket_cars.csv into the data/ folder."
        )

    df_raw = pd.read_csv(DATA_PATH)
    df = dat(df_raw)

    # top 10 manufacturers by count
    if "Manufacturer" in df.columns:
        top10_manufacturers = (
            df["Manufacturer"].value_counts().head(10).index.tolist()
        )
    else:
        top10_manufacturers = []

    # Train/test split
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=10)

    # ------------------------------------------------------------------
    numerical_features = df_train.select_dtypes(include=["number"])
    corr_m = numerical_features.corr(method="pearson")["Price"]
    corr_m_no_price = corr_m.drop("Price")
    most_correlated_feature = corr_m_no_price.abs().idxmax()

    X_train_uni = df_train[[most_correlated_feature]].values
    y_train_uni = df_train["Price"].values
    X_test_uni = df_test[[most_correlated_feature]].values
    y_test_uni = df_test["Price"].values

    scaler_uni = StandardScaler()
    X_train_uni_scaled = scaler_uni.fit_transform(X_train_uni)
    X_test_uni_scaled = scaler_uni.transform(X_test_uni)


    beta_uni = np.sum(
        X_train_uni_scaled.flatten() * (y_train_uni - np.mean(y_train_uni))
    ) / np.sum(X_train_uni_scaled.flatten() ** 2)
    alpha_uni = np.mean(y_train_uni)

    y_train_pred_uni = beta_uni * X_train_uni_scaled.flatten() + alpha_uni
    y_test_pred_uni = beta_uni * X_test_uni_scaled.flatten() + alpha_uni

    # Metrics
    SS_res_uni = np.sum((y_train_uni - y_train_pred_uni) ** 2)
    SS_tot_uni = np.sum((y_train_uni - np.mean(y_train_uni)) ** 2)
    r2_uni_manual = 1 - (SS_res_uni / SS_tot_uni)
    r2_uni_test = r2_score(y_test_uni, y_test_pred_uni)
    rmse_uni = np.sqrt(mean_squared_error(y_test_uni, y_test_pred_uni))
    mae_uni = mean_absolute_error(y_test_uni, y_test_pred_uni)
    mape_uni = mean_absolute_percentage_error(y_test_uni, y_test_pred_uni)

    # ------------------------------------------------------------------
    top_manufacturer = df["Manufacturer"].mode()[0]

    df_train_top = df_train[
        (df_train["Manufacturer"] == top_manufacturer)
        & (df_train["Model year"] >= 2010)
    ]
    df_test_top = df_test[
        (df_test["Manufacturer"] == top_manufacturer)
        & (df_test["Model year"] >= 2010)
    ]

    # Numeric columns for correlation
    numeric_cols_top = df_train_top.select_dtypes(include=[np.number])
    corr_top = numeric_cols_top.corr()
    most_correlated_feature_top = (
        corr_top["Price"].drop("Price").abs().idxmax()
    )
    most_correlated_feature_top_corr = corr_top["Price"][most_correlated_feature_top]

    # Univariate regression for top manufacturer
    X_train_top = df_train_top[[most_correlated_feature_top]].values
    y_train_top = df_train_top["Price"].values
    X_test_top = df_test_top[[most_correlated_feature_top]].values
    y_test_top = df_test_top["Price"].values

    mean_model_year_top = df_train_top["Model year"].mean()
    std_model_year_top = df_train_top["Model year"].std()

    X_train_scaled_top = (X_train_top - mean_model_year_top) / std_model_year_top
    X_test_scaled_top = (X_test_top - mean_model_year_top) / std_model_year_top

    beta_top = np.sum(
        X_train_scaled_top.flatten() * (y_train_top - np.mean(y_train_top))
    ) / np.sum(X_train_scaled_top.flatten() ** 2)
    alpha_top = np.mean(y_train_top)

    y_train_pred_top = beta_top * X_train_scaled_top.flatten() + alpha_top
    y_test_pred_top = beta_top * X_test_scaled_top.flatten() + alpha_top

    SS_res_top = np.sum((y_train_top - y_train_pred_top) ** 2)
    SS_tot_top = np.sum((y_train_top - np.mean(y_train_top)) ** 2)
    r2_top_manual = 1 - (SS_res_top / SS_tot_top)
    rmse_top = np.sqrt(mean_squared_error(y_test_top, y_test_pred_top))
    mae_top = mean_absolute_error(y_test_top, y_test_pred_top)
    mape_top = mean_absolute_percentage_error(y_test_top, y_test_pred_top)

    # ------------------------------------------------------------------
    mv_features = ["Model year", "Horsepowers", "Mileage"]

    mv_per_brand = {}

    for brand in top10_manufacturers:
        df_train_b = df_train[
            (df_train["Manufacturer"] == brand)
            & (df_train["Model year"] >= 2010)
        ]
        df_test_b = df_test[
            (df_test["Manufacturer"] == brand)
            & (df_test["Model year"] >= 2010)
        ]

        if df_train_b.empty or df_test_b.empty:
            # Skip brands without enough data in train or test
            continue

        X_train_mv = df_train_b[mv_features].values
        y_train_mv = df_train_b["Price"].values
        X_test_mv = df_test_b[mv_features].values
        y_test_mv = df_test_b["Price"].values

        mean_mv = df_train_b[mv_features].mean().values
        std_mv = df_train_b[mv_features].std().values
        std_mv[std_mv == 0] = 1

        X_train_scaled_mv = (X_train_mv - mean_mv) / std_mv
        X_test_scaled_mv = (X_test_mv - mean_mv) / std_mv

        # Design matrix with intercept
        X_train_intercept = np.column_stack(
            (np.ones(X_train_scaled_mv.shape[0]), X_train_scaled_mv)
        )
        beta_mv = np.linalg.inv(X_train_intercept.T @ X_train_intercept) @ (
            X_train_intercept.T @ y_train_mv
        )

        X_test_intercept = np.column_stack(
            (np.ones(X_test_scaled_mv.shape[0]), X_test_scaled_mv)
        )
        y_train_pred_mv = X_train_intercept @ beta_mv
        y_test_pred_mv = X_test_intercept @ beta_mv

        SS_res_mv = np.sum((y_train_mv - y_train_pred_mv) ** 2)
        SS_tot_mv = np.sum((y_train_mv - np.mean(y_train_mv)) ** 2)
        r2_mv_manual = 1 - (SS_res_mv / SS_tot_mv)
        r2_mv_test = r2_score(y_test_mv, y_test_pred_mv)
        rmse_mv = np.sqrt(mean_squared_error(y_test_mv, y_test_pred_mv))
        mae_mv = mean_absolute_error(y_test_mv, y_test_pred_mv)
        mape_mv = np.mean(np.abs((y_test_mv - y_test_pred_mv) / y_test_mv))

        train_ranges = {
            "Model year": (
                float(df_train_b["Model year"].min()),
                float(df_train_b["Model year"].max()),
            ),
            "Horsepowers": (
                float(df_train_b["Horsepowers"].min()),
                float(df_train_b["Horsepowers"].max()),
            ),
            "Mileage": (
                float(df_train_b["Mileage"].min()),
                float(df_train_b["Mileage"].max()),
            ),
        }

        mv_per_brand[brand] = {
            "features": mv_features,
            "beta": beta_mv.astype(float),
            "mean": mean_mv.astype(float),
            "std": std_mv.astype(float),
            "metrics": {
                "r2_train_manual": float(r2_mv_manual),
                "r2_test": float(r2_mv_test),
                "rmse": float(rmse_mv),
                "mae": float(mae_mv),
                "mape": float(mape_mv),
            },
            "train_ranges": train_ranges,
            "n_train": int(df_train_b.shape[0]),
            "n_test": int(df_test_b.shape[0]),
        }

    mv_top = mv_per_brand.get(top_manufacturer, None)

    artifact = {
        "most_correlated_feature": most_correlated_feature,
        "uni": {
            "beta": float(beta_uni),
            "alpha": float(alpha_uni),
            "feature": most_correlated_feature,
            "metrics": {
                "r2_train_manual": float(r2_uni_manual),
                "r2_test": float(r2_uni_test),
                "rmse": float(rmse_uni),
                "mae": float(mae_uni),
                "mape": float(mape_uni),
            },
        },
        "top_manufacturer": top_manufacturer,
        "top10_manufacturers": top10_manufacturers,
        "top": {
            "feature": most_correlated_feature_top,
            "feature_corr": float(most_correlated_feature_top_corr),
            "beta": float(beta_top),
            "alpha": float(alpha_top),
            "mean_model_year": float(mean_model_year_top),
            "std_model_year": float(std_model_year_top),
            "metrics": {
                "r2_train_manual": float(r2_top_manual),
                "rmse": float(rmse_top),
                "mae": float(mae_top),
                "mape": float(mape_top),
            },
        },
        "mv_per_brand": mv_per_brand,
        "mv_top": mv_top,
    }

    joblib.dump(artifact, "model.pkl")
    print("Model trained and saved to model.pkl")
    print(f"Top manufacturer used in assignment-style subset: {top_manufacturer}")
    if top10_manufacturers:
        print("Top 10 manufacturers (by count):", ", ".join(top10_manufacturers))
    print("Trained multivariate models for brands:", ", ".join(mv_per_brand.keys()))


if __name__ == "__main__":
    train()
