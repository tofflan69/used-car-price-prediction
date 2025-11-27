import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Base directory for this app file (visual-representation/)
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "blocket_cars.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact


@st.cache_resource
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put blocket_cars.csv into the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    return df



def predict_mv(year, hp, mileage, mv_info):
    """Predict price using a brand-specific multivariate model."""
    features = mv_info["features"]
    beta = np.asarray(mv_info["beta"], dtype=float)  # length 1 + n_features
    mean = np.asarray(mv_info["mean"], dtype=float)
    std = np.asarray(mv_info["std"], dtype=float)
    std[std == 0] = 1.0

    # Build feature vector in the correct order
    x_map = {
        "Model year": float(year),
        "Horsepowers": float(hp),
        "Mileage": float(mileage),
    }
    x_vec = np.array([[x_map[name] for name in features]], dtype=float)

    x_scaled = (x_vec - mean) / std
    x_design = np.column_stack((np.ones(x_scaled.shape[0]), x_scaled))
    y_pred = x_design @ beta
    return float(y_pred[0])


def main():
    st.set_page_config(
        page_title="Used Car Price Prediction from Blocket Data",
        page_icon="",
        layout="centered",
    )

    st.title("Used Car Price Prediction from Blocket Data")
    st.write(
        "This app is an interactive demo of my used car price prediction project.\n\n"
        "It uses classic machine learning techniques: standardized numerical features and "
        "closed-form multivariate linear models trained separately for **each of the top 10 "
        "car manufacturers**."
    )


    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error(
            "model.pkl not found.\n\n"
            "Please run `python train_model.py` locally to train and save the model "
            "before running this app."
        )
        return

    # Unpack artifact
    top_manufacturer = artifact["top_manufacturer"]
    top10_manufacturers = artifact.get("top10_manufacturers", [])
    mv_per_brand = artifact.get("mv_per_brand", {})

    # Filter top10 list to only those brands that actually have a trained model
    available_brands = [b for b in top10_manufacturers if b in mv_per_brand]
    if not available_brands:
        st.error("No per-brand multivariate models were trained. Check your data.")
        return

    uni_info = artifact["uni"]
    top_info = artifact["top"]
    mv_top = artifact.get("mv_top")

    tab_predict, tab_data, tab_model = st.tabs(
        ["Predict", "Data & plots", "Model details"]
    )

    # ------------------------ PREDICT TAB ------------------------
    with tab_predict:
        st.subheader("Brand-specific multivariate model (top 10 manufacturers)")

        brand = st.selectbox(
            "Manufacturer",
            options=available_brands,
        )

        mv_info = mv_per_brand[brand]
        train_ranges = mv_info["train_ranges"]

        st.markdown(
            f"This model is trained on **{brand}** cars with model year ≥ 2010, "
            "using these features:"
        )
        st.markdown(
            "• Model year  \n"
            "• Horsepowers  \n"
            "• Mileage (km)"
        )

        col1, col2 = st.columns(2)
        with col1:
            year_min, year_max = train_ranges["Model year"]
            year = st.slider(
                "Model year",
                min_value=int(year_min),
                max_value=int(year_max),
                value=int((year_min + year_max) / 2),
                step=1,
            )
            mileage_min, mileage_max = train_ranges["Mileage"]
            mileage = st.number_input(
                "Mileage",
                min_value=float(mileage_min),
                max_value=float(mileage_max),
                value=float(mileage_min),
                step=1000.0,
            )
        with col2:
            hp_min, hp_max = train_ranges["Horsepowers"]
            hp = st.number_input(
                "Horsepowers",
                min_value=float(hp_min),
                max_value=float(hp_max),
                value=float(hp_min),
                step=5.0,
            )

        input_df = pd.DataFrame(
            {
                "Manufacturer": [brand],
                "Model year": [year],
                "Horsepowers": [hp],
                "Mileage": [mileage],
            }
        )
        st.markdown("##### Your input")
        st.dataframe(input_df, use_container_width=True)

        # Predict
        if st.button("Predict price"):
            y_pred = predict_mv(year, hp, mileage, mv_info)
            st.success(f"Estimated price: **{y_pred:,.0f} SEK**")

            st.caption(
                f"Prediction from the {brand} model trained on "
                f"{mv_info['n_train']} training cars (year ≥ 2010)."
            )

        # ------------------------ DATA & PLOTS TAB ------------------------
    with tab_data:
        st.subheader("Data & plots")

        try:
            df = load_data()
        except FileNotFoundError as e:
            st.error(str(e))
            df = None

        if df is not None:
            st.markdown(
                "This project is based on used car listings from Blocket.\n\n"
                "Before training the models, I apply a simple cleaning step to remove "
                "obvious errors and extreme outliers:\n"
                "- Keep prices between 10 000 and 3 000 000\n"
                "- Keep mileage above 0 and at most 50 000 km\n"
                "- Keep cars with at most 1 000 horsepower\n"
                "- Drop rows with missing values in key columns "
                "(`Price`, `Mileage`, `Transmission`, `Horsepowers`)"
            )

            st.markdown("**Sample of the raw data**")
            st.dataframe(df.head(), use_container_width=True)

            # Re-run cleaning here for EDA display
            from train_model import dat  # reuse same cleaning
            df_clean = dat(df)

            st.markdown("**Sample of the cleaned data**")
            st.dataframe(df_clean.head(), use_container_width=True)

            st.markdown("**Basic stats for the cleaned dataset**")
            col_a, col_b = st.columns(2)
            col_a.metric("Rows", f"{df_clean.shape[0]:,}")
            col_b.metric("Columns", str(df_clean.shape[1]))

            st.markdown("**Top 10 manufacturers by number of cars**")
            top_counts = (
                df_clean["Manufacturer"]
                .value_counts()
                .head(10)
                .rename_axis("Manufacturer")
                .reset_index(name="Count")
            )
            st.dataframe(top_counts, use_container_width=True)

            # Plot: Price vs Model year for the original "top manufacturer"
            df_top = df_clean[
                (df_clean["Manufacturer"] == top_manufacturer)
                & (df_clean["Model year"] >= 2010)
            ]

            st.markdown(
                f"For the first version of this project, I started by focusing on "
                f"**{top_manufacturer}** and model years ≥ 2010 to get a cleaner view "
                "of how price evolves over time for a single brand."
            )
            st.write(f"Rows in this subset: {df_top.shape[0]:,}")

            if {"Model year", "Price"}.issubset(df_top.columns):
                st.markdown("**Price vs model year for this brand**")
                fig, ax = plt.subplots()
                ax.scatter(
                    df_top["Model year"],
                    df_top["Price"],
                    alpha=0.4,
                    label="Data points",
                )

                mean_my = top_info["mean_model_year"]
                std_my = top_info["std_model_year"]
                beta_top = top_info["beta"]
                alpha_top = top_info["alpha"]

                year_range = np.linspace(
                    df_top["Model year"].min(),
                    df_top["Model year"].max(),
                    100,
                )
                scaled_year = (year_range - mean_my) / std_my
                pred_prices = beta_top * scaled_year + alpha_top

                ax.plot(
                    year_range,
                    pred_prices,
                    linestyle="dotted",
                    label="Fitted linear trend",
                )
                ax.set_xlabel("Model year")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)

    # ------------------------ MODEL DETAILS TAB ------------------------
    with tab_model:
        st.subheader("Model details & metrics")

        # ---- Baseline univariate model ----
        st.markdown("### Baseline univariate model")
        st.markdown(
            f"This is a simple baseline where I use a single numeric feature "
            f"(**{uni_info['feature']}**) to predict the price.\n\n"
            "The feature is standardized and I solve for the best line using a closed-form "
            "least squares solution."
        )
        m_uni = uni_info["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² (train, manual)", f"{m_uni['r2_train_manual']:.3f}")
        col2.metric("R² (test)", f"{m_uni['r2_test']:.3f}")
        col3.metric("RMSE (test)", f"{m_uni['rmse']:,.0f}")
        col4.metric("MAPE (test)", f"{m_uni['mape']*100:.1f}%")

        # ---- Single-brand univariate model ----
        st.markdown("### Single-brand univariate model")
        st.markdown(
            f"Here I narrow the data down to **{top_manufacturer}** (model year ≥ 2010) "
            "and fit another simple linear model using the most correlated numeric feature "
            "for that brand."
        )
        m_top = top_info["metrics"]
        st.markdown(
            f"- Brand: **{top_manufacturer}**  \n"
            f"- Feature: **{top_info['feature']}**  \n"
            f"- Correlation with price: {top_info['feature_corr']:.3f}"
        )
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("R² (train, manual)", f"{m_top['r2_train_manual']:.3f}")
        col6.metric("RMSE (test)", f"{m_top['rmse']:,.0f}")
        col7.metric("MAE (test)", f"{m_top['mae']:,.0f}")
        col8.metric("MAPE (test)", f"{m_top['mape']*100:.1f}%")

        # ---- Original single-brand multivariate model (reference) ----
        if mv_top is not None:
            st.markdown("### Single-brand multivariate model (reference)")
            st.markdown(
                "This is the original multivariate version where I still focus on one brand "
                f"(**{top_manufacturer}**, year ≥ 2010) but now use three features:\n"
                "- Model year\n"
                "- Horsepowers\n"
                "- Mileage\n\n"
                "The features are standardized and the parameters are estimated with the "
                "normal equation `(XᵀX)⁻¹ Xᵀy`."
            )
            m_mv_top = mv_top["metrics"]
            col9, col10, col11, col12 = st.columns(4)
            col9.metric("R² (train, manual)", f"{m_mv_top['r2_train_manual']:.3f}")
            col10.metric("R² (test)", f"{m_mv_top['r2_test']:.3f}")
            col11.metric("RMSE (test)", f"{m_mv_top['rmse']:,.0f}")
            col12.metric("MAPE (test)", f"{m_mv_top['mape']*100:.1f}%")

            st.markdown("#### Coefficients for this model")
            beta_mv_top = np.asarray(mv_top["beta"], dtype=float)
            features_top = ["Intercept"] + mv_top["features"]
            coef_df_top = pd.DataFrame(
                {
                    "Term": features_top,
                    "Coefficient": beta_mv_top,
                }
            )
            st.dataframe(coef_df_top, use_container_width=True)

        # ---- Brand-specific multivariate models (top 10) ----
        st.markdown("### Brand-specific multivariate models (top 10)")
        st.markdown(
            "For the current version of the project, I train a separate multivariate model "
            "for each of the top 10 manufacturers. Every model uses the same three features "
            "(model year, horsepower and mileage), but the parameters are learned per brand.\n\n"
            "The table below gives an overview of how each brand-specific model performs "
            "on the test set."
        )
        rows = []
        for brand_name, info in mv_per_brand.items():
            m = info["metrics"]
            rows.append(
                {
                    "Manufacturer": brand_name,
                    "n_train": info["n_train"],
                    "n_test": info["n_test"],
                    "R² train (manual)": m["r2_train_manual"],
                    "R² test": m["r2_test"],
                    "RMSE (test)": m["rmse"],
                    "MAE (test)": m["mae"],
                    "MAPE (test)": m["mape"],
                }
            )
        mv_df = pd.DataFrame(rows).sort_values("R² test", ascending=False)
        st.dataframe(mv_df, use_container_width=True)


if __name__ == "__main__":
    main()
