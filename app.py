from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Laptop Price Prediction App", layout="wide")


BASE_DIR = Path(__file__).resolve().parent

# Define candidate paths for all expected files
PLOT_CANDIDATES = {
    "Price Distribution": [
        "outputs/figures/price_distribution.png",
        "notebooks/outputs/figures/price_distribution.png",
        "price_distribution.png",
    ],
    "Price by Brand": [
        "outputs/figures/price_by_brand.png",
        "notebooks/outputs/figures/price_by_brand.png",
        "price_by_brand.png",
    ],
    "RAM vs Price": [
        "outputs/figures/ram_vs_price.png",
        "notebooks/outputs/figures/ram_vs_price.png",
        "ram_vs_price.png",
    ],
}

# App will look for these CSV files in the specified order and use the first one it finds. 
MODEL_COMPARISON_CANDIDATES = [
    "outputs/metrics/model_comparison.csv",
    "outputs/metrics/model_comparison_v3.csv",
    "model_comparison.csv",
    "model_comparison_v3.csv",
]

FINAL_MODEL_METRICS_CANDIDATES = [
    "outputs/metrics/final_model_metrics.csv",
    "final_model_metrics.csv",
]

PIPELINE_CANDIDATES = [
    "artifacts/final_price_pipeline.pkl",
    "final_price_pipeline.pkl",
]

CONFIG_CANDIDATES = [
    "artifacts/app_config.json",
    "app_config.json",
]


DERIVED_FEATURES = {
    "total_pixels",
    "ppi",
    "threads_per_core",
    "has_dedicated_gpu",
    "gpu_vram_missing_flag",
    "cpu_p_cores_missing_flag",
    "cpu_e_cores_missing_flag",
}

FLAG_TO_BASE = {
    "gpu_vram_missing_flag": "gpu_vram_gb",
    "cpu_p_cores_missing_flag": "cpu_p_cores",
    "cpu_e_cores_missing_flag": "cpu_e_cores",
}


EXPLANATION_TEXT = (
    "The final deployed model is the refined MLP using Feature Set B. "
    "I selected it because it achieved the strongest test performance across RMSE, "
    "MAE and R². Ridge remained an important benchmark with slightly stronger cross validation "
    "stability, but the MLP gave the best final predictive accuracy on unseen test data."
)



# Helper functions
def resolve_existing_path(candidates: list[str]) -> Path | None:
    search_roots = [BASE_DIR, BASE_DIR.parent, Path.cwd()]
    checked: list[Path] = []

    for root in search_roots:
        for candidate in candidates:
            path = (root / candidate).resolve()
            if path not in checked:
                checked.append(path)
            if path.exists():
                return path
    return None


@st.cache_resource
def load_pipeline() -> Any:
    pipeline_path = resolve_existing_path(PIPELINE_CANDIDATES)
    if pipeline_path is None:
        raise FileNotFoundError(
            "Could not find final_price_pipeline.pkl. Expected it in artifacts/ or the app root directory."
        )
    return joblib.load(pipeline_path)


@st.cache_data
def load_config() -> dict[str, Any]:
    config_path = resolve_existing_path(CONFIG_CANDIDATES)
    if config_path is None:
        raise FileNotFoundError(
            "Could not find app_config.json. Expected it in artifacts/ or the app root directory."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(candidates: list[str]) -> pd.DataFrame | None:
    csv_path = resolve_existing_path(candidates)
    if csv_path is None:
        return None
    return pd.read_csv(csv_path)


def discover_plot_paths() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label, candidates in PLOT_CANDIDATES.items():
        path = resolve_existing_path(candidates)
        if path is not None:
            found[label] = path
    return found


def format_currency(value: float) -> str:
    return f"{value:,.2f}"



def is_binary_feature(feature_name: str, cfg: dict[str, Any]) -> bool:
    feature_range = cfg.get("numeric_ranges", {}).get(feature_name, {})
    return feature_range.get("min") == 0.0 and feature_range.get("max") == 1.0



def safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan



def build_user_input_form(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    display_names = cfg.get("display_names", {})
    categorical_options = cfg.get("categorical_options", {})
    numeric_ranges = cfg.get("numeric_ranges", {})
    selected_features = cfg.get("features", [])

    base_features = [f for f in selected_features if f not in DERIVED_FEATURES]
    categorical_features = [f for f in base_features if f in categorical_options]
    numeric_features = [f for f in base_features if f not in categorical_options]

    values: dict[str, Any] = {}
    unknown_flags: dict[str, bool] = {
        "gpu_vram_gb": False,
        "cpu_p_cores": False,
        "cpu_e_cores": False,
    }

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Categorical inputs")
        for feature in categorical_features:
            options = categorical_options[feature]
            label = display_names.get(feature, feature)
            default_index = 0
            if feature == "brand" and "lenovo" in options:
                default_index = options.index("lenovo")
            elif feature == "device_category" and "General" in options:
                default_index = options.index("General")
            elif feature == "cpu_brand" and "Intel" in options:
                default_index = options.index("Intel")
            elif feature == "cpu_family" and "Core" in options:
                default_index = options.index("Core")
            elif feature == "cpu_series" and "Core I5" in options:
                default_index = options.index("Core I5")
            elif feature == "gpu_brand" and "Intel" in options:
                default_index = options.index("Intel")
            elif feature == "gpu_series" and "Iris Xe" in options:
                default_index = options.index("Iris Xe")
            elif feature == "gpu_type" and "Integrated" in options:
                default_index = options.index("Integrated")
            elif feature == "os_name" and "Windows" in options:
                default_index = options.index("Windows")

            values[feature] = st.selectbox(label, options, index=default_index)

    with right_col:
        st.markdown("### Numeric inputs")
        for feature in numeric_features:
            label = display_names.get(feature, feature)
            feature_range = numeric_ranges.get(feature, {})
            min_value = safe_float(feature_range.get("min"))
            max_value = safe_float(feature_range.get("max"))
            median_value = safe_float(feature_range.get("median"))

            if feature in {"gpu_vram_gb", "cpu_p_cores", "cpu_e_cores"}:
                unknown_flags[feature] = st.checkbox(
                    f"{label} unknown / missing",
                    value=False,
                    help="Tick this if the specification is unknown so the app can pass a missing value to the pipeline.",
                )
                if unknown_flags[feature]:
                    values[feature] = np.nan
                    st.caption(f"{label} will be passed as a missing value.")
                    continue

            if is_binary_feature(feature, cfg):
                values[feature] = float(
                    st.selectbox(label, [0.0, 1.0], index=int(median_value if not np.isnan(median_value) else 0.0))
                )
            elif feature in {"display_size_inch", "ppi", "threads_per_core"}:
                step = 0.1 if feature != "threads_per_core" else 0.01
                values[feature] = float(
                    st.number_input(
                        label,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(median_value),
                        step=step,
                    )
                )
            else:
                values[feature] = float(
                    st.number_input(
                        label,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(median_value),
                        step=1.0,
                    )
                )

    return values, unknown_flags

# Create derived features and ensure all expected features are present in correct order

def enrich_features(raw_inputs: dict[str, Any], unknown_flags: dict[str, bool], cfg: dict[str, Any]) -> pd.DataFrame:
    numeric_ranges = cfg.get("numeric_ranges", {})
    selected_features = cfg.get("features", [])

    data = raw_inputs.copy()

    width = safe_float(data.get("display_width_px"))
    height = safe_float(data.get("display_height_px"))
    display_size = safe_float(data.get("display_size_inch"))
    cpu_cores = safe_float(data.get("cpu_core_count"))
    cpu_threads = safe_float(data.get("cpu_thread_count"))

    data["total_pixels"] = width * height if not np.isnan(width) and not np.isnan(height) else np.nan

    if not np.isnan(width) and not np.isnan(height) and not np.isnan(display_size) and display_size > 0:
        data["ppi"] = float(np.sqrt(width ** 2 + height ** 2) / display_size)
    else:
        data["ppi"] = np.nan

    if not np.isnan(cpu_cores) and cpu_cores > 0 and not np.isnan(cpu_threads):
        data["threads_per_core"] = float(cpu_threads / cpu_cores)
    else:
        data["threads_per_core"] = np.nan

    data["has_dedicated_gpu"] = 1.0 if str(data.get("gpu_type", "")).lower() == "dedicated" else 0.0
    data["gpu_vram_missing_flag"] = 1.0 if unknown_flags.get("gpu_vram_gb", False) else 0.0
    data["cpu_p_cores_missing_flag"] = 1.0 if unknown_flags.get("cpu_p_cores", False) else 0.0
    data["cpu_e_cores_missing_flag"] = 1.0 if unknown_flags.get("cpu_e_cores", False) else 0.0

 # Ensure all selected features exist and are ordered exactly as expected.
    for feature in selected_features:
        if feature not in data:
            if feature in FLAG_TO_BASE:
                data[feature] = 0.0
            elif feature in numeric_ranges:
                data[feature] = numeric_ranges[feature].get("median", np.nan)
            else:
                data[feature] = None

    ordered_input = {feature: data[feature] for feature in selected_features}
    return pd.DataFrame([ordered_input])



def render_prediction_tab(cfg: dict[str, Any]) -> None:
    st.header("Price Prediction")
    st.write(
        "Enter the laptop specifications below. The app will transform the raw inputs using the "
        "saved preprocessing pipeline and then generate a predicted price."
    )

    with st.expander("Model and feature summary", expanded=False):
        st.write(f"Selected model: {cfg.get('selected_model', 'Unknown model')}")
        st.write(f"Target variable: {cfg.get('target', 'Unknown target')}")
        st.write(f"Number of deployed features: {len(cfg.get('features', []))}")

    with st.form("prediction_form"):
        raw_inputs, unknown_flags = build_user_input_form(cfg)
        submitted = st.form_submit_button("Predict laptop price", use_container_width=True)

    if not submitted:
        return

    try:
        pipeline = load_pipeline()
        input_df = enrich_features(raw_inputs, unknown_flags, cfg)
        prediction = float(pipeline.predict(input_df)[0])

        currency_note = cfg.get("notes", {}).get(
            "currency_note",
            "Predicted price is returned in the dataset currency unit.",
        )

        st.success(f"Predicted price: {format_currency(prediction)}")
        st.caption(currency_note)

        with st.expander("Show processed input sent to the model", expanded=False):
            st.dataframe(input_df, use_container_width=True)

    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Export the final pipeline to artifacts/final_price_pipeline.pkl and keep app_config.json in artifacts/.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")



def render_insights_tab() -> None:
    st.header("Dataset Insights")
    st.write(
        "This section shows saved exploratory plots from the notebook workflow. "
        "These visuals help explain how the data behaves before modelling."
    )

    found_plots = discover_plot_paths()
    if not found_plots:
        st.warning(
            "No plot images were found. Expected files such as outputs/figures/price_distribution.png, "
            "outputs/figures/price_by_brand.png and outputs/figures/ram_vs_price.png."
        )
        return

    for label, path in found_plots.items():
        st.subheader(label)
        st.image(str(path), use_container_width=True)
        st.caption(str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path))



def render_performance_tab(cfg: dict[str, Any]) -> None:
    st.header("Model Performance")

    model_comparison = load_csv(MODEL_COMPARISON_CANDIDATES)
    final_metrics = load_csv(FINAL_MODEL_METRICS_CANDIDATES)

    if final_metrics is not None and not final_metrics.empty:
        row = final_metrics.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Final test RMSE", format_currency(float(row["test_rmse"])))
        m2.metric("Final test MAE", format_currency(float(row["test_mae"])))
        m3.metric("Final test R²", f"{float(row['test_r2']):.4f}")

        extra_a, extra_b, extra_c = st.columns(3)
        extra_a.metric("Training rows", int(row.get("n_train", 0)))
        extra_b.metric("Test rows", int(row.get("n_test", 0)))
        extra_c.metric("Feature count", int(row.get("feature_count", len(cfg.get("features", [])))))

    st.subheader("Why this model was selected")
    st.write(EXPLANATION_TEXT)

    if model_comparison is not None and not model_comparison.empty:
        st.subheader("Model comparison table")

        display_df = model_comparison.copy()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if "r2" in col.lower():
                display_df[col] = display_df[col].map(lambda x: round(float(x), 4))
            else:
                display_df[col] = display_df[col].map(lambda x: round(float(x), 2))

        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Model comparison CSV not found. Add outputs/metrics/model_comparison.csv or model_comparison_v3.csv.")


# App layout and main function

def main() -> None:
    st.title("Laptop Price Prediction App")
    st.caption("COM763 Advanced Machine Learning portfolio project")

    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Dataset Insights", "Model Performance"])

    with tab1:
        render_prediction_tab(cfg)
    with tab2:
        render_insights_tab()
    with tab3:
        render_performance_tab(cfg)


if __name__ == "__main__":
    main()
