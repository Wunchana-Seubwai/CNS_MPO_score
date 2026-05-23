"""
CNS MPO Score Calculator — Streamlit App
========================================
Calculates the six-property CNS Multiparameter Optimization score
(Wager et al., ACS Chem. Neurosci. 2010) from SMILES input.

Run locally:
    streamlit run app.py
"""

import io
import base64

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from cns_mpo_calculator import (
    calculate_cns_mpo,
    MPO_BOUNDS,
    MPO_UNITS,
    MPO_IDEAL,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNS MPO Calculator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .score-box {
        border-radius: 12px; padding: 20px; text-align: center;
        font-size: 3rem; font-weight: 800; margin: 10px 0;
    }
    .score-green  { background: #e8f5e9; color: #2e7d32; border: 2px solid #43a047; }
    .score-orange { background: #fff3e0; color: #e65100; border: 2px solid #fb8c00; }
    .score-red    { background: #ffebee; color: #c62828; border: 2px solid #e53935; }
    .prop-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 12px; margin: 4px 0; text-align: center;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 4px; margin: 16px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def mol_to_svg(mol, width=380, height=280):
    """Render RDKit molecule to SVG string."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_png_base64(mol, width=380, height=280):
    """Render RDKit molecule to base64 PNG."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        b64 = base64.b64encode(png).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        # Fallback to SVG
        return None


def radar_chart(scores: dict, title: str = "") -> go.Figure:
    """Plotly radar chart for CNS MPO scores."""
    cats   = list(scores.keys())
    vals   = list(scores.values())
    # Close the polygon
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure()

    # Ideal zone (all = 1)
    fig.add_trace(go.Scatterpolar(
        r=[1] * (len(cats) + 1), theta=cats_c,
        fill="toself",
        fillcolor="rgba(76, 175, 80, 0.08)",
        line=dict(color="rgba(76, 175, 80, 0.4)", dash="dot", width=1),
        name="Ideal (score = 1)",
        showlegend=True,
    ))

    # Compound scores
    total = sum(vals)
    if total >= 4:
        fill_color = "rgba(26, 115, 232, 0.20)"
        line_color = "rgba(26, 115, 232, 0.9)"
    elif total >= 3:
        fill_color = "rgba(251, 140, 0, 0.20)"
        line_color = "rgba(251, 140, 0, 0.9)"
    else:
        fill_color = "rgba(229, 57, 53, 0.20)"
        line_color = "rgba(229, 57, 53, 0.9)"

    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        name="Compound",
        showlegend=True,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10),
            angularaxis=dict(tickfont_size=13),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=380,
        margin=dict(t=30, b=60, l=60, r=60),
        title=dict(text=title, font_size=14) if title else None,
    )
    return fig


def score_bar_chart(scores: dict) -> go.Figure:
    """Horizontal bar chart of individual scores."""
    props = list(scores.keys())
    vals  = list(scores.values())
    colors = ["#43a047" if v >= 0.7 else "#fb8c00" if v >= 0.4 else "#e53935" for v in vals]

    fig = go.Figure(go.Bar(
        y=props, x=vals,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in vals],
        textposition="auto",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], title="Score"),
        yaxis=dict(title=""),
        height=280,
        margin=dict(t=10, b=10, l=60, r=20),
    )
    return fig


def mpo_gauge(total: float) -> go.Figure:
    """Gauge chart for total CNS MPO score."""
    if total >= 4:
        bar_color = "#43a047"
    elif total >= 3:
        bar_color = "#fb8c00"
    else:
        bar_color = "#e53935"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text="CNS MPO Score", font_size=16),
        number=dict(font_size=42, suffix=" / 6"),
        gauge=dict(
            axis=dict(range=[0, 6], tickwidth=1, tickcolor="darkblue"),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor="white",
            steps=[
                dict(range=[0, 3],   color="#ffebee"),
                dict(range=[3, 4],   color="#fff3e0"),
                dict(range=[4, 6],   color="#e8f5e9"),
            ],
            threshold=dict(
                line=dict(color="black", width=3),
                thickness=0.75,
                value=4,
            ),
        ),
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert list of result dicts to a flat DataFrame for download."""
    rows = []
    for r in results:
        if r is None:
            continue
        row = {"SMILES": r["smiles"], "Name": r.get("name", "")}
        row.update({f"prop_{k}": v for k, v in r["result"]["properties"].items()})
        row.update({f"score_{k}": v for k, v in r["result"]["scores"].items()})
        row["CNS_MPO"] = r["result"]["total_mpo"]
        row["Interpretation"] = r["result"]["interpretation"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    show_mol = st.checkbox("Show 2D structure", value=True)
    show_radar = st.checkbox("Show radar chart", value=True)
    show_bars = st.checkbox("Show bar chart", value=True)
    show_gauge = st.checkbox("Show MPO gauge", value=True)

    st.markdown("---")
    st.markdown("### 📖 Scoring Reference")
    ref_df = pd.DataFrame(
        [
            {"Property": k,
             "d=1 (optimal)": MPO_IDEAL[k],
             "d=0 (penalty)": f"{'≥' + str(MPO_BOUNDS[k][1])}{' ' + MPO_UNITS[k] if MPO_UNITS[k] else ''}"}
            for k in MPO_BOUNDS
        ]
    )
    st.dataframe(ref_df, hide_index=True, use_container_width=True)

    st.markdown(
        """
        ---
        **Reference:**  
        Wager *et al.*, *ACS Chem. Neurosci.* **2010**, 1, 435–449  
        
        > CNS MPO ≥ 4 → Favorable  
        > CNS MPO 3–4 → Moderate  
        > CNS MPO < 3  → Poor  
        
        ⚠️ *pKa is estimated via SMARTS patterns.  
        For high-accuracy pKa, use Marvin / ChemAxon.*
        """
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧠 CNS MPO Score Calculator</p>', unsafe_allow_html=True)
st.markdown(
    "Calculate the **six-property CNS Multiparameter Optimization** score "
    "from SMILES — Wager *et al.* (2010)"
)

tab1, tab2 = st.tabs(["🔬 Single Molecule", "📊 Batch Analysis"])

# ── TAB 1: Single Molecule ────────────────────────────────────────────────────
with tab1:
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        smiles_input = st.text_input(
            "Enter SMILES",
            value="CN1CCC[C@H]1c2cccnc2",
            placeholder="e.g. c1ccc2[nH]ccc2c1",
            label_visibility="visible",
        )
    with col_in2:
        name_input = st.text_input("Compound name (optional)", value="Nicotine")

    calc_btn = st.button("🧮 Calculate CNS MPO", type="primary", use_container_width=True)

    if calc_btn or smiles_input:
        result = calculate_cns_mpo(smiles_input)

        if result is None:
            st.error("❌ Invalid SMILES — please check your input.")
        else:
            st.markdown("---")

            # ── Score summary row
            c1, c2, c3 = st.columns([1.2, 1, 1.5])

            with c1:
                css_class = (
                    "score-green"  if result["color"] == "green"  else
                    "score-orange" if result["color"] == "orange" else
                    "score-red"
                )
                label = name_input or "Compound"
                st.markdown(
                    f'<div class="score-box {css_class}">'
                    f'  <div style="font-size:1rem;font-weight:500;">{label}</div>'
                    f'  {result["total_mpo"]:.2f} / 6'
                    f'  <div style="font-size:1rem;">{result["interpretation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with c2:
                if show_mol:
                    svg = mol_to_svg(result["mol"], width=300, height=240)
                    st.image(svg.encode(), width=300)

            with c3:
                if show_gauge:
                    st.plotly_chart(
                        mpo_gauge(result["total_mpo"]),
                        use_container_width=True,
                    )

            st.markdown("---")

            # ── Properties & scores table
            st.markdown('<p class="section-header">📋 Properties & Individual Scores</p>',
                        unsafe_allow_html=True)

            prop_cols = st.columns(6)
            props  = result["properties"]
            scores = result["scores"]

            for i, (prop, val) in enumerate(props.items()):
                sc  = scores[prop]
                unit = MPO_UNITS[prop]
                bg  = "#e8f5e9" if sc >= 0.7 else "#fff3e0" if sc >= 0.4 else "#ffebee"
                emoji = "🟢" if sc >= 0.7 else "🟡" if sc >= 0.4 else "🔴"
                with prop_cols[i]:
                    st.markdown(
                        f'<div class="prop-card" style="background:{bg};">'
                        f'  <div style="font-weight:700;font-size:1rem;">{prop}</div>'
                        f'  <div style="font-size:1.4rem;font-weight:600;">'
                        f'    {val}{" " + unit if unit else ""}'
                        f'  </div>'
                        f'  <div style="font-size:0.85rem;color:#555;">score: {sc:.3f}</div>'
                        f'  {emoji}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("")

            # ── Charts row
            if show_radar or show_bars:
                chart_cols = st.columns(2)
                if show_radar:
                    with chart_cols[0]:
                        st.markdown('<p class="section-header">🕸 Radar Chart</p>',
                                    unsafe_allow_html=True)
                        st.plotly_chart(
                            radar_chart(scores, title=""),
                            use_container_width=True,
                        )
                if show_bars:
                    with chart_cols[1]:
                        st.markdown('<p class="section-header">📊 Score Breakdown</p>',
                                    unsafe_allow_html=True)
                        st.plotly_chart(
                            score_bar_chart(scores),
                            use_container_width=True,
                        )


# ── TAB 2: Batch Analysis ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch CNS MPO Calculation")
    st.markdown(
        "Enter one SMILES per line. Optionally add a name separated by a comma or tab.  \n"
        "Example: `CC(=O)Oc1ccccc1C(=O)O, Aspirin`"
    )

    # CSV upload
    uploaded = st.file_uploader(
        "Upload CSV (columns: smiles, name [optional])",
        type=["csv", "txt"],
    )

    batch_text = st.text_area(
        "Or paste SMILES here (one per line)",
        height=180,
        placeholder="c1ccc(CC2CCCCN2)cc1, Example1\nCN1CCC[C@H]1c2cccnc2, Nicotine",
    )

    run_batch = st.button("🚀 Run Batch Calculation", type="primary")

    if run_batch:
        entries = []

        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            # Handle header
            for i, line in enumerate(lines):
                if i == 0 and not Chem.MolFromSmiles(line.split(",")[0].strip()):
                    continue  # skip header
                parts = [p.strip() for p in line.replace("\t", ",").split(",", 1)]
                smi  = parts[0]
                name = parts[1] if len(parts) > 1 else f"Cpd_{i+1}"
                entries.append((smi, name))

        elif batch_text.strip():
            for i, line in enumerate(batch_text.strip().splitlines()):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.replace("\t", ",").split(",", 1)]
                smi  = parts[0]
                name = parts[1] if len(parts) > 1 else f"Cpd_{i+1}"
                entries.append((smi, name))

        if not entries:
            st.warning("⚠️ No valid entries found.")
        else:
            results_list = []
            progress = st.progress(0, text="Calculating…")

            for idx, (smi, name) in enumerate(entries):
                res = calculate_cns_mpo(smi)
                results_list.append({"smiles": smi, "name": name, "result": res})
                progress.progress((idx + 1) / len(entries),
                                   text=f"Processing {idx+1}/{len(entries)}: {name}")

            progress.empty()

            # ── Results table
            valid   = [r for r in results_list if r["result"] is not None]
            invalid = [r for r in results_list if r["result"] is None]

            if invalid:
                st.warning(
                    f"⚠️ {len(invalid)} invalid SMILES skipped: "
                    + ", ".join(r["name"] for r in invalid)
                )

            if valid:
                df = results_to_dataframe(valid)

                # Rename columns for display
                display_cols = {
                    "name": "Name", "smiles": "SMILES",
                    "prop_MW": "MW", "prop_cLogP": "cLogP", "prop_cLogD": "cLogD",
                    "prop_TPSA": "TPSA", "prop_HBD": "HBD", "prop_pKa": "pKa",
                    "score_MW": "s_MW", "score_cLogP": "s_cLogP", "score_cLogD": "s_cLogD",
                    "score_TPSA": "s_TPSA", "score_HBD": "s_HBD", "score_pKa": "s_pKa",
                    "CNS_MPO": "CNS MPO", "Interpretation": "Assessment",
                }
                df_display = df.rename(columns=display_cols)

                # Color-map the CNS MPO column
                def color_mpo(val):
                    if val >= 4:
                        return "background-color: #c8e6c9"
                    elif val >= 3:
                        return "background-color: #ffe0b2"
                    return "background-color: #ffcdd2"

                st.markdown(
                    f"**{len(valid)} compounds** calculated successfully."
                )
                st.dataframe(
                    df_display.style.applymap(color_mpo, subset=["CNS MPO"]),
                    use_container_width=True,
                    hide_index=True,
                )

                # Score distribution histogram
                st.markdown("#### CNS MPO Score Distribution")
                mpo_vals = [r["result"]["total_mpo"] for r in valid]
                fig_hist = go.Figure(go.Histogram(
                    x=mpo_vals, nbinsx=12,
                    marker_color=[
                        "#43a047" if v >= 4 else "#fb8c00" if v >= 3 else "#e53935"
                        for v in mpo_vals
                    ],
                    xbins=dict(start=0, end=6, size=0.5),
                ))
                fig_hist.add_vline(x=4, line_dash="dash", line_color="green",
                                   annotation_text="CNS MPO = 4 (threshold)",
                                   annotation_position="top right")
                fig_hist.update_layout(
                    xaxis_title="CNS MPO Score",
                    yaxis_title="Count",
                    height=300,
                    bargap=0.05,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # ── Download
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name="cns_mpo_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>CNS MPO Calculator · Based on Wager *et al.* (2010) · "
    "Properties: RDKit · pKa: SMARTS estimation (approximate)</small>",
    unsafe_allow_html=True,
)
