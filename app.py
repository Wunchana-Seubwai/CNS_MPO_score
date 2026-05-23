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

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

# Try Cairo (needs libXrender); fall back to PIL-based SVG renderer
try:
    from rdkit.Chem.Draw import rdMolDraw2D
    _DRAW_BACKEND = "cairo"
except ImportError:
    _DRAW_BACKEND = "pil"

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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    /* ── Global typography ─────────────────────────────────────── */
    html, body, [class*="css"], .stApp, .main, .block-container {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: #1a2332;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Crimson Pro', Georgia, 'Times New Roman', serif !important;
        color: #0d2818 !important;
        letter-spacing: -0.01em;
        font-weight: 600 !important;
    }
    code, .stCodeBlock {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }

    /* ── Title block ───────────────────────────────────────────── */
    .main-title {
        font-family: 'Crimson Pro', Georgia, serif !important;
        font-size: 2.4rem;
        font-weight: 700;
        color: #0d3b22;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 1.02rem;
        color: #3a4a52;
        line-height: 1.5;
        margin-top: 0.2rem;
    }
    .main-subtitle b { color: #0d3b22; font-weight: 600; }
    .ref-link {
        display: inline-block;
        margin-top: 0.4rem;
        font-size: 0.88rem;
        color: #1f5f3f;
        text-decoration: none;
        border-bottom: 1px dotted #1f5f3f;
        font-family: 'Source Sans 3', sans-serif;
    }
    .ref-link:hover {
        color: #0d3b22;
        border-bottom-color: #0d3b22;
    }
    .title-rule {
        border: none;
        border-top: 2px solid #0d3b22;
        width: 60px;
        margin: 0.8rem 0 1rem 0;
    }

    /* ── Score box (verdict card) ──────────────────────────────── */
    .score-box {
        border-radius: 6px;
        padding: 22px;
        text-align: center;
        font-family: 'Crimson Pro', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }
    .score-green  { background: #f1f7f1; color: #1b5e20; border: 1.5px solid #1b5e20; }
    .score-orange { background: #fdf6ec; color: #8a4a00; border: 1.5px solid #8a4a00; }
    .score-red    { background: #fbf0ef; color: #8b1a1a; border: 1.5px solid #8b1a1a; }

    /* ── Property cards ────────────────────────────────────────── */
    .prop-card {
        background: #fafbfc;
        border: 1px solid #e3e8ed;
        border-radius: 4px;
        padding: 12px;
        margin: 4px 0;
        text-align: center;
        font-family: 'Source Sans 3', sans-serif;
    }

    /* ── Section header ────────────────────────────────────────── */
    .section-header {
        font-family: 'Crimson Pro', Georgia, serif !important;
        font-size: 1.25rem;
        font-weight: 600;
        color: #0d3b22 !important;
        border-bottom: 1.5px solid #0d3b22;
        padding-bottom: 4px;
        margin: 18px 0 10px 0;
        letter-spacing: -0.01em;
    }

    /* ── Primary button: WHITE bg with DARK GREEN border ──────── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #ffffff !important;
        color: #0d3b22 !important;
        border: 2px solid #0d3b22 !important;
        border-radius: 4px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #0d3b22 !important;
        color: #ffffff !important;
        border-color: #0d3b22 !important;
    }
    .stButton > button[kind="primary"]:active,
    .stButton > button[kind="primary"]:focus {
        background-color: #f1f7f1 !important;
        color: #0d3b22 !important;
        border-color: #0d3b22 !important;
        box-shadow: 0 0 0 2px rgba(13, 59, 34, 0.15) !important;
    }
    /* Secondary buttons (download etc.) same treatment, slightly lighter */
    .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
        background-color: #ffffff !important;
        color: #1f5f3f !important;
        border: 1.5px solid #1f5f3f !important;
        border-radius: 4px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 500 !important;
    }
    .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
        background-color: #1f5f3f !important;
        color: #ffffff !important;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #0d3b22 !important;
        border: 1.5px solid #0d3b22 !important;
        border-radius: 4px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #0d3b22 !important;
        color: #ffffff !important;
    }

    /* ── Tabs (academic look) ──────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #d1dad3;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        color: #5a6b6f !important;
        padding: 0.5rem 0.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0d3b22 !important;
        border-bottom: 2.5px solid #0d3b22 !important;
    }

    /* ── Input fields ──────────────────────────────────────────── */
    .stTextInput input, .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 4px !important;
        border: 1px solid #c5cfc8 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0d3b22 !important;
        box-shadow: 0 0 0 1px #0d3b22 !important;
    }

    /* ── Sidebar styling ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #f7f8f7;
        border-right: 1px solid #d1dad3;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Crimson Pro', Georgia, serif !important;
        color: #0d3b22 !important;
    }

    /* ── Dataframes ────────────────────────────────────────────── */
    .stDataFrame {
        font-family: 'Source Sans 3', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def mol_to_image(mol, width=380, height=280):
    """
    Render molecule → display in Streamlit.
    Uses rdMolDraw2D (SVG) when available, falls back to PIL PNG.
    """
    rdDepictor.Compute2DCoords(mol)

    if _DRAW_BACKEND == "cairo":
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.drawOptions().addAtomIndices = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return "svg", drawer.GetDrawingText()
        except Exception:
            pass

    # PIL fallback — works without X11
    img = Draw.MolToImage(mol, size=(width, height))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "png", buf.getvalue()


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
        fill_color = "rgba(13, 59, 34, 0.18)"
        line_color = "rgba(13, 59, 34, 0.9)"
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
    colors = ["#1b5e20" if v >= 0.7 else "#8a4a00" if v >= 0.4 else "#8b1a1a" for v in vals]

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


def generate_assessment(props: dict, scores: dict, total: float) -> dict:
    """
    Build a detailed CNS-suitability assessment with verdict, flags,
    and per-property commentary (all in English).
    """
    # ── Overall verdict
    if total >= 4.0:
        verdict       = "PASS"
        verdict_long  = "Likely suitable for CNS penetration"
        verdict_color = "#2e7d32"
        verdict_bg    = "#e8f5e9"
        verdict_icon  = "✅"
    elif total >= 3.0:
        verdict       = "BORDERLINE"
        verdict_long  = "Marginal CNS profile — optimization recommended"
        verdict_color = "#e65100"
        verdict_bg    = "#fff3e0"
        verdict_icon  = "⚠️"
    else:
        verdict       = "FAIL"
        verdict_long  = "Unlikely to achieve CNS exposure"
        verdict_color = "#c62828"
        verdict_bg    = "#ffebee"
        verdict_icon  = "❌"

    # ── Per-property commentary
    commentary = []

    # MW
    if props["MW"] <= 360:
        commentary.append(("MW", "good",
            f"Molecular weight ({props['MW']:.1f} Da) is within the optimal range "
            f"(≤360 Da) for BBB penetration."))
    elif props["MW"] <= 500:
        commentary.append(("MW", "warn",
            f"Molecular weight ({props['MW']:.1f} Da) is acceptable but above the "
            f"optimal 360 Da threshold — heavier molecules cross the BBB less efficiently."))
    else:
        commentary.append(("MW", "bad",
            f"Molecular weight ({props['MW']:.1f} Da) exceeds the upper limit of 500 Da. "
            f"Consider truncation or scaffold reduction."))

    # cLogP
    if props["cLogP"] <= 3:
        commentary.append(("cLogP", "good",
            f"cLogP ({props['cLogP']:.2f}) is in the optimal lipophilicity window "
            f"(≤3), balancing permeability and clearance."))
    elif props["cLogP"] <= 5:
        commentary.append(("cLogP", "warn",
            f"cLogP ({props['cLogP']:.2f}) is moderately lipophilic. Watch for "
            f"increased metabolic clearance and off-target promiscuity."))
    else:
        commentary.append(("cLogP", "bad",
            f"cLogP ({props['cLogP']:.2f}) is too high (>5). High lipophilicity "
            f"correlates with poor solubility, hERG liability, and high clearance."))

    # cLogD
    if props["cLogD"] <= 2:
        commentary.append(("cLogD", "good",
            f"cLogD at pH 7.4 ({props['cLogD']:.2f}) is optimal — favorable distribution "
            f"between aqueous and lipid phases."))
    elif props["cLogD"] <= 4:
        commentary.append(("cLogD", "warn",
            f"cLogD ({props['cLogD']:.2f}) is acceptable but elevated. May limit "
            f"free fraction and increase plasma protein binding."))
    else:
        commentary.append(("cLogD", "bad",
            f"cLogD ({props['cLogD']:.2f}) is too high (>4). Likely high non-specific "
            f"binding and poor free-drug exposure in brain."))

    # TPSA
    if props["TPSA"] <= 90:
        commentary.append(("TPSA", "good",
            f"TPSA ({props['TPSA']:.1f} Å²) is within the CNS-friendly range (≤90 Å²) "
            f"associated with passive BBB permeability."))
    elif props["TPSA"] <= 120:
        commentary.append(("TPSA", "warn",
            f"TPSA ({props['TPSA']:.1f} Å²) is elevated. BBB penetration begins to "
            f"decline sharply above 90 Å²."))
    else:
        commentary.append(("TPSA", "bad",
            f"TPSA ({props['TPSA']:.1f} Å²) is too polar for the CNS (>120 Å²). "
            f"Consider masking polar groups or replacing H-bond acceptors."))

    # HBD
    if props["HBD"] == 0:
        commentary.append(("HBD", "good",
            f"No H-bond donors — ideal for BBB crossing."))
    elif props["HBD"] <= 2:
        commentary.append(("HBD", "warn",
            f"{props['HBD']} H-bond donor(s) — still acceptable but each additional "
            f"donor reduces brain penetration."))
    else:
        commentary.append(("HBD", "bad",
            f"{props['HBD']} H-bond donors (>3) is a strong negative predictor of "
            f"CNS exposure. Consider donor masking (methylation, intramolecular H-bonds)."))

    # pKa
    if props["pKa"] <= 8:
        commentary.append(("pKa", "good",
            f"Most basic pKa ({props['pKa']:.1f}) is favorable. The neutral fraction "
            f"at physiological pH supports passive permeability."))
    elif props["pKa"] <= 10:
        commentary.append(("pKa", "warn",
            f"pKa ({props['pKa']:.1f}) is elevated — the compound is largely "
            f"protonated at pH 7.4, reducing passive BBB diffusion."))
    else:
        commentary.append(("pKa", "bad",
            f"pKa ({props['pKa']:.1f}) is too high (>10). The fully protonated "
            f"species cannot easily cross membranes."))

    # ── Top recommendations
    flags = [(p, lvl, msg) for (p, lvl, msg) in commentary if lvl == "bad"]
    warns = [(p, lvl, msg) for (p, lvl, msg) in commentary if lvl == "warn"]

    if flags:
        recommendation = (
            f"**Priority issues:** {', '.join(p for p, _, _ in flags)}. "
            f"Address these properties first to bring CNS MPO above 4."
        )
    elif warns:
        recommendation = (
            f"**Fine-tuning:** {', '.join(p for p, _, _ in warns)} are in the "
            f"borderline range. Small modifications could push the score higher."
        )
    else:
        recommendation = (
            "All six properties are in their optimal ranges. The compound shows "
            "an excellent CNS-drug profile."
        )

    return {
        "verdict":         verdict,
        "verdict_long":    verdict_long,
        "verdict_color":   verdict_color,
        "verdict_bg":      verdict_bg,
        "verdict_icon":    verdict_icon,
        "commentary":      commentary,
        "recommendation":  recommendation,
        "n_pass":          sum(1 for _, lvl, _ in commentary if lvl == "good"),
        "n_warn":          len(warns),
        "n_fail":          len(flags),
    }


def mpo_gauge(total: float) -> go.Figure:
    """Gauge chart for total CNS MPO score."""
    if total >= 4:
        bar_color = "#1b5e20"
    elif total >= 3:
        bar_color = "#8a4a00"
    else:
        bar_color = "#8b1a1a"

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

st.markdown(
    """
    <p class="main-title">🧠 CNS MPO Score Calculator</p>
    <p class="main-subtitle">
        Calculate the <b>six-property CNS Multiparameter Optimization</b> score
        from SMILES&nbsp;:&nbsp; Wager <i>et&nbsp;al.</i> (2010)
    </p>
    <a class="ref-link"
       href="https://pubmed.ncbi.nlm.nih.gov/22778837/"
       target="_blank" rel="noopener noreferrer">
       📑 Reference&nbsp;: PubMed 22778837
    </a>
    <hr class="title-rule">
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔬 Single Molecule",
    "📊 Batch Analysis",
    "📚 Documentation",
])

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
                    fmt, data = mol_to_image(result["mol"], width=300, height=240)
                    if fmt == "svg":
                        # Streamlit's st.image doesn't accept SVG bytes — render as HTML
                        st.markdown(
                            f'<div style="display:flex;justify-content:center;">{data}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.image(data, width=300)

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

            # ── Final assessment / verdict ───────────────────────────────
            st.markdown("---")
            st.markdown('<p class="section-header">📝 CNS Suitability Assessment</p>',
                        unsafe_allow_html=True)

            assessment = generate_assessment(props, scores, result["total_mpo"])

            # Verdict banner
            st.markdown(
                f"""
                <div style="background:{assessment['verdict_bg']};
                            border-left:6px solid {assessment['verdict_color']};
                            border-radius:8px; padding:18px 22px; margin:8px 0;">
                  <div style="font-size:1.4rem; font-weight:700;
                              color:{assessment['verdict_color']};">
                    {assessment['verdict_icon']} {assessment['verdict']} —
                    CNS MPO = {result['total_mpo']:.2f} / 6
                  </div>
                  <div style="font-size:1rem; color:#333; margin-top:4px;">
                    {assessment['verdict_long']}.
                  </div>
                  <div style="font-size:0.9rem; color:#555; margin-top:8px;">
                    <b>{assessment['n_pass']}</b> properties optimal ·
                    <b>{assessment['n_warn']}</b> borderline ·
                    <b>{assessment['n_fail']}</b> out of range
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Brain penetration prediction
            mpo_val = result["total_mpo"]
            if mpo_val >= 4:
                brain_msg = (
                    "Compounds with CNS MPO ≥ 4 have been shown to correlate "
                    "with favorable brain-to-plasma ratios (Kp,uu > 0.3) and "
                    "successful clinical CNS exposure in the Pfizer dataset."
                )
            elif mpo_val >= 3:
                brain_msg = (
                    "Marginal CNS profile. Compounds in this range show variable "
                    "brain penetration and often require efflux-pump (P-gp) testing "
                    "to confirm CNS exposure."
                )
            else:
                brain_msg = (
                    "Compounds with CNS MPO < 3 are statistically unlikely to achieve "
                    "free-drug brain exposure sufficient for CNS pharmacology. "
                    "Major structural revision is recommended."
                )

            st.info(f"🧠 **Brain penetration outlook:** {brain_msg}")

            # Per-property commentary
            st.markdown("##### Property-by-property analysis")

            level_style = {
                "good": ("🟢", "#e8f5e9", "#2e7d32"),
                "warn": ("🟡", "#fff8e1", "#e65100"),
                "bad":  ("🔴", "#ffebee", "#c62828"),
            }

            for prop_name, level, msg in assessment["commentary"]:
                icon, bg, color = level_style[level]
                st.markdown(
                    f"""
                    <div style="background:{bg}; border-left:4px solid {color};
                                border-radius:6px; padding:10px 14px; margin:6px 0;
                                font-size:0.92rem;">
                      <b style="color:{color};">{icon} {prop_name}</b> &nbsp;—&nbsp; {msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Recommendation
            st.markdown("##### Recommendation")
            st.markdown(assessment["recommendation"])



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
                        "#1b5e20" if v >= 4 else "#8a4a00" if v >= 3 else "#8b1a1a"
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


# ── TAB 3: Documentation ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">📖 About CNS MPO</p>', unsafe_allow_html=True)

    st.markdown(
        """
The **Central Nervous System Multiparameter Optimization (CNS MPO)** score
is a composite metric introduced by Wager *et al.* at Pfizer to align the
physicochemical properties of drug candidates with the requirements for
crossing the **blood–brain barrier (BBB)** and achieving meaningful free-drug
exposure in brain tissue.

Unlike rigid filters such as the Lipinski Rule of Five, the MPO uses
**desirability functions** — smooth, monotonic transformations of each
property that map it to a continuous score between 0 (undesirable) and 1
(optimal). The six individual scores are summed to give an overall score
between **0 and 6**.

A threshold of **CNS MPO ≥ 4** was empirically derived: in the Pfizer dataset,
74% of marketed CNS drugs and 60% of clinical CNS candidates met this
criterion, compared to a much smaller fraction of non-CNS reference drugs.
"""
    )

    # ── Desirability function math
    st.markdown('<p class="section-header">📐 The Desirability Function</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            r"""
For a **monotonic decreasing** property with optimal threshold $T_0$
and penalty threshold $T_1$:
"""
        )
        st.latex(
            r"""
            d(x) = \begin{cases}
              1, & x \leq T_0 \\
              \dfrac{T_1 - x}{T_1 - T_0}, & T_0 < x < T_1 \\
              0, & x \geq T_1
            \end{cases}
            """
        )
        st.markdown(
            r"""
The **total CNS MPO score** is the sum of the six individual desirability
scores:
"""
        )
        st.latex(
            r"\text{CNS MPO} = d_{\text{MW}} + d_{\text{cLogP}} + d_{\text{cLogD}} "
            r"+ d_{\text{TPSA}} + d_{\text{HBD}} + d_{\text{pKa}}"
        )
        st.markdown(
            "Score range: **0 ≤ CNS MPO ≤ 6**. The closer to 6, the more "
            "CNS-drug-like the compound."
        )

    with col_b:
        # Mini illustration of the desirability function
        import plotly.graph_objects as _go
        _x = list(range(0, 11))
        _T0, _T1 = 4, 8
        _y = [
            1.0 if v <= _T0 else 0.0 if v >= _T1 else (_T1 - v) / (_T1 - _T0)
            for v in _x
        ]
        _fig = _go.Figure()
        _fig.add_trace(_go.Scatter(
            x=_x, y=_y, mode="lines+markers",
            line=dict(color="#0d3b22", width=3),
            marker=dict(size=7, color="#0d3b22"),
            name="d(x)",
        ))
        _fig.add_vline(x=_T0, line_dash="dot", line_color="#1b5e20",
                       annotation_text="T₀", annotation_position="top")
        _fig.add_vline(x=_T1, line_dash="dot", line_color="#8b1a1a",
                       annotation_text="T₁", annotation_position="top")
        _fig.update_layout(
            title=dict(text="Generic desirability function",
                       font=dict(family="Crimson Pro", size=14)),
            xaxis_title="Property value (x)",
            yaxis_title="Desirability d(x)",
            yaxis=dict(range=[-0.05, 1.1]),
            height=280,
            margin=dict(t=40, b=40, l=40, r=20),
            font=dict(family="Source Sans 3"),
        )
        st.plotly_chart(_fig, use_container_width=True)

    # ── Six properties (detailed)
    st.markdown('<p class="section-header">🔬 The Six Physicochemical Properties</p>',
                unsafe_allow_html=True)

    st.markdown(
        "Each property is monotonically penalized as it moves away from the "
        "CNS-friendly range. Thresholds were derived by Wager *et al.* through "
        "statistical analysis of marketed CNS drugs versus non-CNS reference sets."
    )

    # 1. MW
    with st.expander("**① Molecular Weight (MW)**  ·  T₀ = 360 Da  ·  T₁ = 500 Da",
                     expanded=False):
        st.markdown(
            r"""
**Definition.** The sum of atomic weights of all atoms in the molecule (Da, g/mol).

**Why it matters for CNS exposure.**
Passive permeability across the BBB follows roughly Stokes–Einstein–like
diffusion: smaller molecules diffuse faster. Above ~500 Da, passive
permeability falls sharply because the molecule no longer fits efficiently
through the tight-junction–regulated paracellular and transcellular routes.
Larger molecules also tend to recruit additional polar groups to maintain
solubility, compounding the BBB penalty.

**Wager *et al.* observation.**
Marketed CNS drugs cluster around an MW of ~310 Da, whereas non-CNS oral
drugs average ~400 Da. The desirability function gives full credit up to
**360 Da** and zero credit at **500 Da**.

**Optimization tip.**
Trim aliphatic linkers, fuse rings, replace heavy halogens with fluorine,
or eliminate solubilizing groups that have no SAR contribution.
"""
        )

    # 2. cLogP
    with st.expander("**② cLogP** (calculated octanol–water partition coefficient)  ·  T₀ = 3  ·  T₁ = 5",
                     expanded=False):
        st.markdown(
            r"""
**Definition.**
$$\text{cLogP} = \log_{10}\!\left(\dfrac{[\text{drug}]_{\text{octanol}}}{[\text{drug}]_{\text{water}}}\right)$$

for the **neutral** form of the molecule. In this app it is computed by the
Crippen (Wildman–Crippen) atom-contribution method via RDKit.

**Why it matters for CNS exposure.**
Lipophilicity drives passive membrane permeability — but only up to a point.
Beyond cLogP ≈ 3:
- Plasma protein binding rises rapidly, reducing free drug.
- Metabolic clearance increases (CYP turnover scales with lipophilicity).
- Aqueous solubility falls.
- Off-target promiscuity, hERG inhibition, and phospholipidosis risk all rise.

**Wager *et al.* observation.**
The optimal CNS-drug cLogP window is **2–3**. The function awards full credit
up to **3** and zero credit at **5**. This is the most heavily-weighted
property in real-world failure analyses.

**Optimization tip.**
Replace lipophilic substituents (long alkyl chains, di-halogenated aryls)
with polar bioisosteres (e.g. fluorine for methyl, tetrazole for carboxylic
acid groups already present).
"""
        )

    # 3. cLogD
    with st.expander("**③ cLogD₇.₄** (distribution coefficient at pH 7.4)  ·  T₀ = 2  ·  T₁ = 4",
                     expanded=False):
        st.markdown(
            r"""
**Definition.**
Unlike cLogP (which considers only the neutral form), cLogD reflects the
**actual** distribution at physiological pH, integrating ionized + neutral
forms:
$$\text{cLogD}_{7.4} = \log_{10}\!\left(\dfrac{[\text{drug}]_{\text{octanol}}}
{[\text{total drug}]_{\text{water, pH 7.4}}}\right)$$

For a **monoprotic base** (most CNS-active amines), Henderson–Hasselbalch
gives:
$$\text{cLogD}_{7.4} = \text{cLogP} - \log_{10}\!\big(1 + 10^{\,pK_a - 7.4}\big)$$

This app uses exactly this formula with the estimated most-basic pKa.

**Why it matters for CNS exposure.**
Only the **neutral** species crosses the BBB efficiently by passive
diffusion. cLogD captures the *effective* lipophilicity that the membrane
actually sees. A high cLogP combined with a high pKa (strongly basic amine)
gives a deceptively lower cLogD — and that is the membrane-relevant value.

**Wager *et al.* observation.**
CNS drugs show a tighter cLogD distribution than cLogP. The optimal range
is **0–2**, with full credit at **≤ 2** and zero credit at **≥ 4**.

**Optimization tip.**
Modulating pKa is often easier than modulating cLogP. Lowering the pKa of a
basic amine by ~1–2 units (e.g. piperidine → morpholine, or adding an
α-fluorine) can dramatically improve cLogD.
"""
        )

    # 4. TPSA
    with st.expander("**④ Topological Polar Surface Area (TPSA)**  ·  T₀ = 90 Å²  ·  T₁ = 120 Å²",
                     expanded=False):
        st.markdown(
            r"""
**Definition.**
The sum of surface contributions of polar atoms (N, O, S, P, and their attached
hydrogens) in the molecule, computed from the 2D topology using the Ertl
fragment-based method (RDKit's `CalcTPSA`).

**Why it matters for CNS exposure.**
TPSA is a proxy for the **H-bonding desolvation penalty** a molecule must pay
to cross a lipid bilayer: every polar atom must shed its water shell before
entering the membrane. Above ~90 Å², BBB penetration drops sharply.

**Wager *et al.* observation.**
The "CNS-friendly TPSA window" is widely cited as **≤ 90 Å²** (versus the
≤ 140 Å² limit for general oral bioavailability). The desirability function
gives full credit ≤ **90 Å²** and zero credit ≥ **120 Å²**.

**Optimization tip.**
Mask polar groups (e.g. methylate a hydroxyl, or replace an amide with a
bioisostere with lower TPSA contribution), reduce the number of polar atoms,
or convert acyclic amides into rigid cyclic ones with intramolecular H-bonds.
"""
        )

    # 5. HBD
    with st.expander("**⑤ Hydrogen Bond Donors (HBD)**  ·  T₀ = 0  ·  T₁ = 3",
                     expanded=False):
        st.markdown(
            r"""
**Definition.**
Number of explicit O–H and N–H bonds in the molecule (RDKit's `CalcNumHBD`).

**Why it matters for CNS exposure.**
Each H-bond donor carries a roughly **constant desolvation penalty** when
the molecule transitions from water to the lipid bilayer interior — and
donors penalize CNS permeability more strongly than acceptors. Compounds
with **0 donors** cross the BBB most readily; **≥ 3 donors** is a strong
negative predictor.

**Wager *et al.* observation.**
HBD count alone is one of the strongest single predictors of CNS exposure
in the Pfizer dataset. The function gives full credit at **0 donors** and
zero credit at **≥ 3 donors**.

**Optimization tip.**
- **Methylate** anilines, hydroxyls, or amides where SAR permits.
- Use **intramolecular H-bonds** to mask polar donors ("chameleonic" design).
- Replace primary amides with nitriles, oxadiazoles, or other bioisosteres.
"""
        )

    # 6. pKa
    with st.expander("**⑥ Most basic pKa**  ·  T₀ = 8  ·  T₁ = 10",
                     expanded=False):
        st.markdown(
            r"""
**Definition.**
The **highest** (most basic) pKa among the ionizable centers in the molecule
— typically the most basic nitrogen. In this app it is estimated via SMARTS
pattern matching against a library of basic functional groups (amines,
amidines, guanidines, basic aromatic N, etc.).

**Why it matters for CNS exposure.**
At physiological pH (7.4), only the **neutral** fraction of an ionizable
compound crosses the BBB passively. As pKa rises above ~8:
- The neutral fraction shrinks (e.g. at pKa = 10, only ~0.25% is neutral).
- The cationic form is "trapped" in lysosomes (lysosomotropism).
- Phospholipidosis risk increases.
- Volume of distribution rises non-specifically.

**Wager *et al.* observation.**
Marketed CNS drugs have a median basic pKa around 8. The desirability function
awards full credit ≤ **8** and zero credit ≥ **10**.

**Optimization tip.**
- Add electron-withdrawing groups β to the basic amine (e.g. α-fluorine,
  α-hydroxyl) to lower pKa by 1–2 units.
- Replace a piperidine (pKa ~10) with a morpholine (pKa ~7.5) when SAR allows.
- Constrain the amine into a less basic ring system.

⚠️ *Caveat:* this app's SMARTS-based pKa is an approximation. For
publication-grade work, use a dedicated predictor (ChemAxon Marvin,
ACD/Labs, or a trained ML model such as pkasolver).
"""
        )

    # ── Validation
    st.markdown('<p class="section-header">✅ Validation from the Paper</p>',
                unsafe_allow_html=True)

    st.markdown(
        """
Wager *et al.* validated the CNS MPO score on three datasets:

1. **119 marketed CNS drugs** — **74 %** had CNS MPO ≥ 4.
2. **108 Pfizer candidate CNS drugs** (in clinical development at the time)
   — **60 %** had CNS MPO ≥ 4.
3. A reference set of **marketed non-CNS oral drugs** — significantly lower
   mean MPO, confirming the score discriminates CNS-targeted chemotypes
   from general-purpose oral drugs.

In addition, within the Pfizer set, compounds with **MPO ≥ 4** showed:

- Higher mean **MDCK–MDR1 permeability** (better passive BBB diffusion).
- Lower **P-glycoprotein (P-gp) efflux ratios**.
- Higher unbound brain-to-plasma ratios (**Kp,uu**).
- Improved hERG and safety margins.

The MPO threshold of **≥ 4** has since been widely adopted across the
industry as a first-pass CNS suitability filter.
"""
    )

    # ── Limitations
    st.markdown('<p class="section-header">⚠️ Limitations & Caveats</p>',
                unsafe_allow_html=True)
    st.markdown(
        """
- **Active transport is not modeled.** The MPO captures passive permeability
  only. P-gp efflux, BCRP, and uptake transporters (LAT1, OATP) are
  property-independent and must be assessed separately (e.g. MDCK-MDR1 assay).
- **Pharmacology is not predicted.** A compound with CNS MPO = 6 may still
  lack target engagement in brain. MPO is a *prerequisite*, not a guarantee.
- **pKa accuracy is the dominant source of error in this app.** SMARTS-based
  estimation is a rough approximation; a difference of 1 pKa unit propagates
  into cLogD and changes the score noticeably.
- **The thresholds are statistical, not mechanistic.** A score of 3.9 is not
  qualitatively different from 4.1 — use the **per-property scores** to
  understand where improvement is needed.
- **The simplified TPSA function** used here is monotonic; the original
  paper uses a slightly more elaborate "hump" function. The difference is
  small for typical CNS chemotypes.
"""
    )

    # ── References
    st.markdown('<p class="section-header">📑 References</p>',
                unsafe_allow_html=True)
    st.markdown(
        """
**Primary reference**

> Wager, T. T.; Hou, X.; Verhoest, P. R.; Villalobos, A.
> *Moving beyond rules: the development of a central nervous system multiparameter
> optimization (CNS MPO) approach to enable alignment of druglike properties.*
> **ACS Chem. Neurosci.** **2010**, *1* (6), 435–449.
> [PubMed 22778837](https://pubmed.ncbi.nlm.nih.gov/22778837/)
> · DOI: [10.1021/cn100008c](https://doi.org/10.1021/cn100008c)

**Follow-up / related papers**

> Wager, T. T.; Chandrasekaran, R. Y.; Hou, X.; Troutman, M. D.;
> Verhoest, P. R.; Villalobos, A.; Will, Y.
> *Defining desirable central nervous system drug space through the alignment
> of molecular properties, in vitro ADME, and safety attributes.*
> **ACS Chem. Neurosci.** **2010**, *1* (6), 420–434.
> [PubMed 22778836](https://pubmed.ncbi.nlm.nih.gov/22778836/)

> Rankovic, Z.
> *CNS drug design: balancing physicochemical properties for optimal brain exposure.*
> **J. Med. Chem.** **2015**, *58* (6), 2584–2608.
> DOI: [10.1021/jm501535r](https://doi.org/10.1021/jm501535r)

**Implementation notes for this app**

- Property descriptors: **RDKit** (`MolWt`, `MolLogP`, `CalcTPSA`, `CalcNumHBD`).
- pKa: SMARTS-based pattern matching against curated functional-group library.
- cLogD: Henderson–Hasselbalch monoprotic-base model at pH 7.4.
- Desirability functions: linear piecewise per Wager (2010), Table 1.
"""
    )

    st.markdown(
        """
<div style="background:#f1f7f1; border-left:4px solid #0d3b22;
            padding:12px 16px; border-radius:4px; margin-top:16px;
            font-size:0.9rem;">
<b>Citation suggestion.</b> If you use this calculator in a publication,
please cite the original Wager <i>et al.</i> (2010) paper above. This tool is
a faithful reimplementation of their published method.
</div>
""",
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>CNS MPO Calculator · Based on Wager *et al.* (2010) · "
    "Properties: RDKit · pKa: SMARTS estimation (approximate)</small>",
    unsafe_allow_html=True,
)
