import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cement Bogue Calculator", page_icon="🧪", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def bogue_from_oxides(row, method="classic", clip_negatives=True, renormalize=False):
    """
    Compute Bogue phases (mass %) from oxide composition (mass %).
    Expected columns (case-insensitive): SiO2, Al2O3, Fe2O3, CaO, SO3 (optional), Na2O (opt), K2O (opt).
    Methods:
      - "classic": classic Bogue without CaO correction for sulfate/alkali
      - "so3_corrected": CaO' = CaO - 0.7 * SO3
      - "so3_alkali_corrected": CaO' = CaO - 0.7 * SO3 - 0.35 * (Na2O + K2O)
        (simple practical adjustment; yes, purists can fight me later)
    """
    # Pull with safe defaults
    s = lambda k: float(row.get(k, row.get(k.lower(), 0.0)) or 0.0)
    SiO2  = s("SiO2")
    Al2O3 = s("Al2O3")
    Fe2O3 = s("Fe2O3")
    CaO   = s("CaO")
    SO3   = s("SO3")
    Na2O  = s("Na2O")
    K2O   = s("K2O")

    # CaO correction options commonly used in plants
    if method == "classic":
        CaO_corr = CaO
    elif method == "so3_corrected":
        CaO_corr = CaO - 0.70 * SO3
    elif method == "so3_alkali_corrected":
        CaO_corr = CaO - 0.70 * SO3 - 0.35 * (Na2O + K2O)
    else:
        CaO_corr = CaO  # fallback, because humans

    # Classic Bogue equations (OPC assumptions, low SCMs)
    C4AF = 3.043 * Fe2O3
    C3A  = 2.650 * Al2O3 - 1.692 * Fe2O3
    C3S  = 4.071 * CaO_corr - 7.600 * SiO2 - 6.718 * Al2O3 - 1.430 * Fe2O3
    C2S  = 2.867 * SiO2 - 0.7544 * C3S

    phases = pd.Series({"C3S": C3S, "C2S": C2S, "C3A": C3A, "C4AF": C4AF}, dtype=float)

    if clip_negatives:
        phases = phases.clip(lower=0)

    if renormalize:
        total = phases.sum()
        if total > 0:
            phases = phases * (100.0 / total)

    return phases

def validate_total_oxides(total, warn_threshold=100.0, hard_threshold=120.0):
    if total <= 0:
        return "Nope. Your oxide total is zero or negative. Try again."
    if total > hard_threshold:
        return f"Oxides total {total:.1f}% is not realistic. Check your units and decimals."
    if abs(total - warn_threshold) > 5:
        return f"Heads up: oxides total {total:.1f}% is far from ~100%. If there’s LOI/moisture/SCMs, fine; otherwise recheck."
    return None

# -----------------------------
# UI
# -----------------------------
st.title("🧪 Bogue Calculator for Cement Clinker")
st.caption("Give me oxides, get phases. Fair trade.")

with st.sidebar:
    st.header("Input options")
    method = st.selectbox(
        "Calculation mode",
        [
            "classic",
            "so3_corrected",
            "so3_alkali_corrected",
        ],
        index=1,
        help=(
            "classic: no CaO correction\n"
            "so3_corrected: CaO' = CaO - 0.70·SO3\n"
            "so3_alkali_corrected: CaO' = CaO - 0.70·SO3 - 0.35·(Na2O+K2O)"
        ),
    )
    clip_neg = st.toggle("Clip negatives to 0", value=True)
    renorm = st.toggle("Renormalize phases to 100%", value=False)

    st.divider()
    st.write("Batch mode (CSV)")
    csv_file = st.file_uploader(
        "Upload CSV with columns: CaO, SiO2, Al2O3, Fe2O3, SO3, Na2O, K2O",
        type=["csv"],
        accept_multiple_files=False,
    )
    st.caption("I’ll ignore extra columns like LOI, MgO, TiO2. I’m generous like that.")

# -----------------------------
# Single-sample inputs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    CaO = st.number_input("CaO %", value=65.0, min_value=0.0, max_value=100.0, step=0.1)
    Al2O3 = st.number_input("Al₂O₃ %", value=4.8, min_value=0.0, max_value=100.0, step=0.1)
with col2:
    SiO2 = st.number_input("SiO₂ %", value=21.0, min_value=0.0, max_value=100.0, step=0.1)
    Fe2O3 = st.number_input("Fe₂O₃ %", value=3.2, min_value=0.0, max_value=100.0, step=0.1)
with col3:
    SO3 = st.number_input("SO₃ %", value=1.5, min_value=0.0, max_value=10.0, step=0.1)
    Na2O = st.number_input("Na₂O %", value=0.2, min_value=0.0, max_value=5.0, step=0.01)
with col4:
    K2O = st.number_input("K₂O %", value=0.6, min_value=0.0, max_value=5.0, step=0.01)
    LOI = st.number_input("LOI % (optional)", value=0.0, min_value=0.0, max_value=30.0, step=0.1)

inputs = {
    "CaO": CaO,
    "SiO2": SiO2,
    "Al2O3": Al2O3,
    "Fe2O3": Fe2O3,
    "SO3": SO3,
    "Na2O": Na2O,
    "K2O": K2O,
    "LOI": LOI,
}
total_ox = CaO + SiO2 + Al2O3 + Fe2O3 + SO3 + Na2O + K2O
msg = validate_total_oxides(total_ox)
if msg:
    st.info(msg)

# -----------------------------
# Single result
# -----------------------------
phases = bogue_from_oxides(inputs, method=method, clip_negatives=clip_neg, renormalize=renorm)
res_df = pd.DataFrame(phases, columns=["%"]).T.round(2)

left, right = st.columns([1, 1])
with left:
    st.subheader("Single Sample Result")
    st.dataframe(res_df, use_container_width=True)
with right:
    st.subheader("Phase Distribution")
    st.bar_chart(phases.rename("Phase %"))

st.caption(
    "Formulas: C₄AF=3.043·Fe₂O₃; C₃A=2.650·Al₂O₃−1.692·Fe₂O₃; "
    "C₃S=4.071·CaO'−7.600·SiO₂−6.718·Al₂O₃−1.430·Fe₂O₃; "
    "C₂S=2.867·SiO₂−0.7544·C₃S. Assumes OPC with low SCMs."
)

# -----------------------------
# Batch mode (CSV)
# -----------------------------
st.divider()
st.header("Batch Calculator (CSV)")
sample_note = st.expander("Sample CSV format (copy/paste)")
with sample_note:
    st.code(
        "SampleID,CaO,SiO2,Al2O3,Fe2O3,SO3,Na2O,K2O\n"
        "K1,65.0,21.0,4.8,3.2,1.5,0.2,0.6\n"
        "K2,64.5,21.5,5.0,3.0,1.2,0.25,0.55\n",
        language="text",
    )

if csv_file is not None:
    try:
        df_in = pd.read_csv(csv_file)
        # Keep columns of interest case-insensitively
        cols_map = {}
        wanted = ["SampleID","CaO","SiO2","Al2O3","Fe2O3","SO3","Na2O","K2O"]
        for c in df_in.columns:
            cl = c.strip()
            for w in wanted:
                if cl.lower() == w.lower() and w not in cols_map.values():
                    cols_map[c] = w
                    break
        df = df_in.rename(columns=cols_map)

        # Build outputs
        out_rows = []
        for i, row in df.iterrows():
            phases_i = bogue_from_oxides(row, method=method, clip_negatives=clip_neg, renormalize=renorm)
            out_rows.append(phases_i)

        out_df = pd.DataFrame(out_rows)
        # Prepend SampleID if present
        if "SampleID" in df.columns or "SampleID" in df.rename(columns=str).columns:
            sid = df.get("SampleID", df.get("sampleid", pd.Series(np.arange(len(df)))))
            out_df.insert(0, "SampleID", sid)

        out_df = out_df.round(2)
        st.success(f"Processed {len(out_df)} samples.")
        st.dataframe(out_df, use_container_width=True)

        # Download
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download results as CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="bogue_results.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not process CSV: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
**Notes**
- These are theoretical phase estimates. Real clinker laughs at theory when burning conditions, minor oxides, and cooling rates get involved.
- If you’re using high SCMs or special cements, use with caution or give me a different model to babysit.
"""
)
