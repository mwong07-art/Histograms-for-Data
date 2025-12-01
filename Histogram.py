"""
NE111 Project – Histogram App by Morgan Wong
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    gamma,
    weibull_min,
    lognorm,
    beta,
    expon,
    rayleigh,
    uniform,
    chi2,
    triang,
)

# --------- Page Config ---------
st.set_page_config(
    page_title="Histogram Distribution Fitter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Helper Functions ---------
DISTRIBUTIONS = {
    "Normal (norm)": norm,
    "Gamma (gamma)": gamma,
    "Weibull (weibull_min)": weibull_min,
    "Lognormal (lognorm)": lognorm,
    "Beta (beta)": beta,
    "Exponential (expon)": expon,
    "Rayleigh (rayleigh)": rayleigh,
    "Uniform (uniform)": uniform,
    "Chi-squared (chi2)": chi2,
    "Triangular (triang)": triang,
}


def parse_manual_data(text: str) -> np.ndarray:
    if not text.strip():
        return np.array([])
    clean = text.replace(",", " ")
    tokens = [t for t in clean.split() if t.strip() != ""]
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            continue
    return np.array(values, dtype=float)


def get_histogram(data: np.ndarray, bins="auto"):
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, edges


def compute_errors(hist_y: np.ndarray, pdf_y: np.ndarray) -> dict:
    diff = pdf_y - hist_y
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return {"MSE": mse, "MAE": mae, "Max Error": max_err}


def split_params(params):
    params = tuple(params)
    if len(params) <= 2:
        shapes = ()
        loc, scale = params
    else:
        shapes = params[:-2]
        loc, scale = params[-2:]
    return shapes, float(loc), float(scale)


def stringify_params(params):
    shapes, loc, scale = split_params(params)
    result = {}
    for i, s in enumerate(shapes, start=1):
        result[f"shape{i}"] = float(s)
    result["loc"] = loc
    result["scale"] = scale
    return result


# --------- UI: Title & Description ---------
st.title("📊 Histogram Distribution Fitter")
st.markdown(
    """
This app fits probability distributions to your data and overlays them on a histogram.

**Workflow:**
1. Enter or upload your data.
2. Choose a distribution from the sidebar.
3. View automatic fits + error metrics.
4. Use the manual sliders to tweak parameters and see how the curve changes.
"""
)

# --------- Sidebar: Data Input ---------
st.sidebar.header("1. Data Input")

input_mode = st.sidebar.radio(
    "How do you want to provide data?",
    ["Paste numbers", "Upload CSV"],
)

data = np.array([])

if input_mode == "Paste numbers":
    example_text = "1.2 1.5 1.9 2.0 2.1 2.3 2.8 3.0 3.1 3.2"
    text = st.sidebar.text_area(
        "Enter numbers (separated by spaces, commas, or new lines):",
        value=example_text,
        height=150,
    )
    data = parse_manual_data(text)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.sidebar.error("No numeric columns found in the CSV.")
            else:
                st.sidebar.write("Numeric columns detected:", numeric_cols)
                selected_cols = st.sidebar.multiselect(
                    "Choose column(s) to use as data:",
                    numeric_cols,
                    default=numeric_cols[:1],
                )
                if selected_cols:
                    data = df[selected_cols].to_numpy().ravel()
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")


if data.size > 0:
    st.subheader("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Count", int(data.size))
    with col2:
        st.metric("Mean", f"{np.mean(data):.4g}")
    with col3:
        st.metric("Std Dev", f"{np.std(data, ddof=1):.4g}")
    with col4:
        st.metric("Min / Max", f"{np.min(data):.4g} / {np.max(data):.4g}")
else:
    st.warning("No valid numeric data yet. Please enter or upload data to proceed.")


if data.size == 0:
    st.stop()

# --------- Sidebar: Distribution Selection ---------
st.sidebar.header("2. Distribution & Fit Options")

dist_name = st.sidebar.selectbox(
    "Choose a distribution to fit:",
    list(DISTRIBUTIONS.keys()),
    index=0,
)
dist_class = DISTRIBUTIONS[dist_name]

num_bins = st.sidebar.slider("Number of histogram bins", min_value=5, max_value=100, value=30)

show_auto_fit = st.sidebar.checkbox("Show automatic fit", value=True)
enable_manual = st.sidebar.checkbox("Enable manual fitting sliders", value=True)

# --------- Compute Histogram ---------
centers, hist_y, bin_edges = get_histogram(data, bins=num_bins)

data_min, data_max = float(np.min(data)), float(np.max(data))
data_range = max(data_max - data_min, 1e-6)

x_plot = np.linspace(data_min - 0.1 * data_range, data_max + 0.1 * data_range, 400)

# --------- Automatic Fit ---------
auto_fit_params = None
auto_pdf_at_centers = None
auto_errors = None

if show_auto_fit:
    try:
        auto_fit_params = dist_class.fit(data)
        auto_shapes, auto_loc, auto_scale = split_params(auto_fit_params)
        dist_auto = dist_class(*auto_fit_params)
        auto_pdf_at_centers = dist_auto.pdf(centers)
        auto_pdf_for_plot = dist_auto.pdf(x_plot)
        auto_errors = compute_errors(hist_y, auto_pdf_at_centers)
    except Exception as e:
        st.error(f"Automatic fit failed for {dist_name}: {e}")
        auto_fit_params = None
        auto_pdf_for_plot = None

# --------- Manual Fit Sliders ---------
manual_params = None
manual_pdf_for_plot = None
manual_errors = None

if enable_manual:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Parameter Controls")

    if auto_fit_params is None:
        fallback_loc = float(np.mean(data))
        fallback_scale = float(np.std(data, ddof=1) or 1.0)
        n_shapes = 1  # generic
        shapes_default = (1.0,) * n_shapes
    else:
        shapes_default, fallback_loc, fallback_scale = split_params(auto_fit_params)
        n_shapes = len(shapes_default)


    shape_values = []
    for i in range(n_shapes):
        default_val = float(shapes_default[i])
        shape_val = st.sidebar.slider(
            f"shape{i+1}",
            min_value=0.01,
            max_value=20.0,
            value=float(default_val),
            step=0.01,
        )
        shape_values.append(shape_val)


    loc_min = data_min - data_range
    loc_max = data_max + data_range
    loc_val = st.sidebar.slider(
        "loc",
        min_value=float(loc_min),
        max_value=float(loc_max),
        value=float(fallback_loc),
    )

    scale_max = max(data_range * 3.0, 1e-3)
    scale_val = st.sidebar.slider(
        "scale",
        min_value=1e-3,
        max_value=float(scale_max),
        value=float(fallback_scale if fallback_scale > 0 else 1.0),
    )

    manual_params = tuple(shape_values + [loc_val, scale_val])

    try:
        dist_manual = dist_class(*manual_params)
        manual_pdf_for_plot = dist_manual.pdf(x_plot)
        manual_pdf_at_centers = dist_manual.pdf(centers)
        manual_errors = compute_errors(hist_y, manual_pdf_at_centers)
    except Exception as e:
        st.sidebar.error(f"Manual parameters are invalid for {dist_name}: {e}")
        manual_params = None
        manual_pdf_for_plot = None

# --------- Main Plot ---------
st.subheader(f"Histogram & Fitted Distribution – {dist_name}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(data, bins=num_bins, density=True, alpha=0.6, edgecolor="black", label="Data histogram")

if show_auto_fit and auto_fit_params is not None:
    ax.plot(x_plot, auto_pdf_for_plot, linewidth=2, label="Automatic fit")

if enable_manual and manual_pdf_for_plot is not None:
    ax.plot(x_plot, manual_pdf_for_plot, linestyle="--", linewidth=2, label="Manual fit")

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title(dist_name)
ax.legend()

st.pyplot(fig)

# --------- Parameter & Error Display ---------
col_auto, col_manual = st.columns(2)

with col_auto:
    st.markdown("### Automatic Fit Parameters")
    if auto_fit_params is None:
        st.info("Automatic fit not available.")
    else:
        params_dict = stringify_params(auto_fit_params)
        st.json(params_dict)

        if auto_errors is not None:
            st.markdown("**Automatic Fit Error Metrics (vs histogram):**")
            st.write(
                f"- MSE: `{auto_errors['MSE']:.4e}`  \n"
                f"- MAE: `{auto_errors['MAE']:.4e}`  \n"
                f"- Max Error: `{auto_errors['Max Error']:.4e}`"
            )

with col_manual:
    st.markdown("### Manual Fit Parameters")
    if not enable_manual or manual_params is None:
        st.info("Manual parameters not available or sliders disabled.")
    else:
        params_dict_manual = stringify_params(manual_params)
        st.json(params_dict_manual)

        if manual_errors is not None:
            st.markdown("**Manual Fit Error Metrics (vs histogram):**")
            st.write(
                f"- MSE: `{manual_errors['MSE']:.4e}`  \n"
                f"- MAE: `{manual_errors['MAE']:.4e}`  \n"
                f"- Max Error: `{manual_errors['Max Error']:.4e}`"
            )

st.markdown("---")
st.caption(
    "Tip: Try different distributions from the sidebar, compare error metrics, "
    "and use the manual sliders to see how each parameter affects the curve."
)


