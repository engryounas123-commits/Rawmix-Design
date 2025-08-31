# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Raw Mix Optimizer (No Cost)", layout="wide")
st.title("Raw Mix Optimizer — Cost Removed (LSF / SM / AM targets)")

st.markdown("""
This app finds material ratios (sum = 100%) for up to 6 materials to best match target:
- LSF = 100 * CaO / (2.8*SiO2 + 1.18*Al2O3 + 0.65*Fe2O3)
- SM  = SiO2 / (Al2O3 + Fe2O3)
- AM  = Al2O3 / Fe2O3

I use a nonlinear optimizer (SLSQP) to minimize squared deviation from targets.
""")

# --- Default materials (example realistic oxide % for demo) ---
default_materials = {
    "Material 1 (Limestone)": {"CaO": 54.0, "SiO2": 3.0, "Al2O3": 0.5, "Fe2O3": 0.2, "MgO": 1.0, "K2O":0.05, "Na2O":0.03, "SO3":0.02},
    "Material 2 (Shale/Clay)":  {"CaO": 2.0,  "SiO2": 55.0,"Al2O3": 20.0,"Fe2O3": 6.0, "MgO": 1.5, "K2O":0.8, "Na2O":0.1, "SO3":0.5},
    "Material 3 (Iron Ore/Red Mud)": {"CaO": 1.0, "SiO2": 10.0,"Al2O3": 4.0, "Fe2O3": 70.0,"MgO": 0.5, "K2O":0.02, "Na2O":0.01, "SO3":0.1},
    "Material 4 (Silica Sand)": {"CaO": 0.5, "SiO2": 96.0,"Al2O3": 0.5, "Fe2O3": 0.2, "MgO": 0.02, "K2O":0.01, "Na2O":0.01, "SO3":0.01},
    "Material 5 (Slag/Fly ash)": {"CaO": 40.0,"SiO2": 30.0,"Al2O3": 8.0, "Fe2O3": 1.5, "MgO": 8.0, "K2O":0.3, "Na2O":0.1, "SO3":1.0},
    "Material 6 (Dolomite/Other)": {"CaO": 30.0,"SiO2": 10.0,"Al2O3": 2.0, "Fe2O3": 0.5, "MgO": 18.0,"K2O":0.02,"Na2O":0.01,"SO3":0.02},
}

st.sidebar.header("General settings")
n_materials = st.sidebar.slider("Number of materials to use", min_value=2, max_value=6, value=4)

materials = [f"Material {i+1}" for i in range(n_materials)]

# Build input table with prefilled realistic defaults (first N)
oxide_cols = ["CaO","SiO2","Al2O3","Fe2O3","MgO","K2O","Na2O","SO3"]
input_table = pd.DataFrame(index=oxide_cols, columns=materials, dtype=float)

# Pre-fill with defaults for convenience
for i, mat in enumerate(materials):
    key = list(default_materials.keys())[i]
    for ox in oxide_cols:
        input_table.loc[ox, mat] = default_materials[key].get(ox, 0.0)

st.subheader("Oxide compositions for each material (%) — edit as needed")
# <<< FIX: use stable data editor API >>>
edited = st.data_editor(input_table, num_rows="dynamic")

# Validate edited table and replace missing values with 0
try:
    # convert to numeric and fill NaN with 0.0
    edited = edited.apply(pd.to_numeric, errors='coerce').fillna(0.0)
except Exception:
    # If user changed the table structure (like renaming columns), rebuild minimal safe matrix
    st.warning("Input table had unexpected structure. Rebuilding from defaults.")
    edited = input_table.copy()

st.sidebar.header("Target & tolerances")
target_lsf = st.sidebar.number_input("Target LSF", value=98.0, step=0.1, format="%.2f")
target_sm  = st.sidebar.number_input("Target SM", value=2.5, step=0.01, format="%.2f")
target_am  = st.sidebar.number_input("Target AM", value=1.6, step=0.01, format="%.2f")

st.sidebar.markdown("**Optional: tolerances (used only for displaying recommended acceptance)**")
tol_lsf = st.sidebar.number_input("LSF tolerance (±)", value=1.0, step=0.1, format="%.2f")
tol_sm  = st.sidebar.number_input("SM tolerance (±)", value=0.05, step=0.01, format="%.2f")
tol_am  = st.sidebar.number_input("AM tolerance (±)", value=0.05, step=0.01, format="%.2f")

st.markdown("### Initial guess (equal split). You can adjust initial guess for solver below.")
# Initial guess vector (equal split)
x0 = np.array([100.0 / n_materials] * n_materials)

# Convert dataframe to numpy arrays (safe)
try:
    oxide_matrix = np.array([[float(edited.loc[ox, mat]) for mat in materials] for ox in oxide_cols])  # shape (8, n)
except Exception:
    # fallback to input_table if indexing failed
    oxide_matrix = np.array([[float(input_table.loc[ox, mat]) for mat in materials] for ox in oxide_cols])
oxide_matrix = oxide_matrix.astype(float)

# Helper: compute mix oxides (percent)
def mix_oxides(x):
    # x in percentages summing to 100
    w = np.array(x) / 100.0  # fractions
    mix = oxide_matrix.dot(w)  # weighted average composition (%)
    return mix  # array length 8 in same oxide order

def compute_LSF(mix):
    CaO = mix[oxide_cols.index("CaO")]
    SiO2 = mix[oxide_cols.index("SiO2")]
    Al2O3 = mix[oxide_cols.index("Al2O3")]
    Fe2O3 = mix[oxide_cols.index("Fe2O3")]
    denom = (2.8 * SiO2 + 1.18 * Al2O3 + 0.65 * Fe2O3)
    if denom <= 1e-9:
        return 1e9
    return 100.0 * CaO / denom

def compute_SM(mix):
    SiO2 = mix[oxide_cols.index("SiO2")]
    Al2O3 = mix[oxide_cols.index("Al2O3")]
    Fe2O3 = mix[oxide_cols.index("Fe2O3")]
    denom = (Al2O3 + Fe2O3)
    if denom <= 1e-9:
        return 1e9
    return SiO2 / denom

def compute_AM(mix):
    Al2O3 = mix[oxide_cols.index("Al2O3")]
    Fe2O3 = mix[oxide_cols.index("Fe2O3")]
    if Fe2O3 <= 1e-9:
        return 1e9
    return Al2O3 / Fe2O3

# Objective: minimize squared relative errors (normalized)
def objective(x):
    mix = mix_oxides(x)
    lsf = compute_LSF(mix)
    sm  = compute_SM(mix)
    am  = compute_AM(mix)
    # normalized squared error (scale each term so numbers are comparable)
    e1 = ((lsf - target_lsf) / max(1.0, target_lsf))**2
    e2 = ((sm  - target_sm)  / max(0.01, target_sm))**2
    e3 = ((am  - target_am)  / max(0.01, target_am))**2
    # tiny penalty to prefer less extreme splits (optional)
    penalty = 1e-6 * np.sum((np.array(x)/100.0)**2)
    return e1 + e2 + e3 + penalty

# Constraints and bounds
bounds = [(0.0, 100.0)] * n_materials
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 100.0},)

st.write("### Run optimizer")
if st.button("Run Optimization"):
    with st.spinner("Solving..."):
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'ftol':1e-9, 'maxiter':1000})
        except Exception as e:
            st.error(f"Optimizer exception: {e}")
            st.stop()

    if not res.success:
        st.error(f"Optimizer failed: {res.message}")
        # still show best attempt if available
        x_best = np.clip(res.x if hasattr(res, 'x') else x0, 0, 100)
    else:
        x_best = np.clip(res.x, 0, 100)

    # normalize to 100 (numerical)
    if x_best.sum() == 0:
        x_norm = x_best
    else:
        x_norm = 100.0 * x_best / x_best.sum()

    mix = mix_oxides(x_norm)
    lsf = compute_LSF(mix)
    sm = compute_SM(mix)
    am = compute_AM(mix)

    st.subheader("Optimized proportions (%)")
    res_df = pd.DataFrame({
        "Material": materials,
        "Proportion (%)": np.round(x_norm, 4)
    })
    st.dataframe(res_df.set_index("Material"))

    st.subheader("Resulting mix oxide composition (%)")
    mix_df = pd.DataFrame({
        "Oxide": oxide_cols,
        "Mix (%)": np.round(mix, 4)
    })
    st.table(mix_df.set_index("Oxide"))

    st.subheader("Achieved metrics")
    metrics = {
        "LSF": (lsf, target_lsf, tol_lsf),
        "SM" : (sm,  target_sm,  tol_sm),
        "AM" : (am,  target_am,  tol_am)
    }
    metrics_df = pd.DataFrame([
        {"Metric":k, "Achieved":round(v[0],6), "Target":v[1], "Tolerance":v[2],
         "Within Tolerance": abs(v[0]-v[1]) <= v[2]}
        for k,v in metrics.items()
    ]).set_index("Metric")
    st.table(metrics_df)

    # Quick diagnostic
    if all(abs(metrics[k][0]-metrics[k][1]) <= metrics[k][2] for k in metrics):
        st.success("All metrics are within the specified tolerances.")
    else:
        st.warning("Some metrics are outside tolerances. Consider adjusting targets or adding/removing materials.")

st.markdown("---")
st.caption("Note: This is a practical working prototype. For production use you may add data validation, bounds on individual materials, or a multi-objective weighting scheme.")
