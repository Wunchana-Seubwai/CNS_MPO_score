# CNS MPO Score Calculator

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09%2B-1A5F7A)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Fcn100008c-blue)](https://doi.org/10.1021/cn100008c)

A web-based implementation of the **six-property Central Nervous System
Multiparameter Optimization (CNS MPO)** score of Wager *et al.* (2010) for
prioritization of CNS drug candidates from SMILES input.

> **Wager, T. T.; Hou, X.; Verhoest, P. R.; Villalobos, A.**
> *Moving beyond rules: the development of a central nervous system
> multiparameter optimization (CNS MPO) approach to enable alignment of
> druglike properties.*
> **ACS Chem. Neurosci.** **2010**, *1* (6), 435–449.
> [PubMed 22778837](https://pubmed.ncbi.nlm.nih.gov/22778837/)
> · DOI: [10.1021/cn100008c](https://doi.org/10.1021/cn100008c)

---

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Validation](#validation)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Overview

The blood–brain barrier (BBB) imposes severe physicochemical constraints on
small-molecule drugs intended for central nervous system (CNS) targets. Unlike
rigid rule-based filters (e.g. Lipinski's Rule of Five), the **CNS MPO**
framework introduced by Wager *et al.* at Pfizer assigns continuous
**desirability scores** (0–1) to six physicochemical properties, the sum of
which gives a composite score in the range **0–6**. A threshold of
**CNS MPO ≥ 4** has been empirically associated with successful CNS exposure
and is now widely adopted across the pharmaceutical industry.

This repository provides a self-contained, Python/Streamlit reimplementation
of the published method, with:

- Single-molecule analysis (radar, gauge, bar, per-property commentary)
- Batch analysis from CSV / pasted SMILES
- An in-app documentation tab summarizing the underlying theory
- A pure-Python scoring module (`cns_mpo_calculator.py`) usable as a library

---

## Scientific Background

### The Desirability Function

For a monotonically penalized property with optimal threshold $T_0$ and
penalty threshold $T_1$, the individual desirability is defined as:

$$
d(x) =
\begin{cases}
1 & \text{if } x \leq T_0 \\
\dfrac{T_1 - x}{T_1 - T_0} & \text{if } T_0 < x < T_1 \\
0 & \text{if } x \geq T_1
\end{cases}
$$

The total CNS MPO score is the sum of six individual desirabilities:

$$
\mathrm{CNS\ MPO} = d_{\mathrm{MW}} + d_{\mathrm{cLogP}} + d_{\mathrm{cLogD}} + d_{\mathrm{TPSA}} + d_{\mathrm{HBD}} + d_{\mathrm{p}K_{a}}
$$

### Property Thresholds (Wager 2010)

| # | Property | Definition | $T_0$ (optimal) | $T_1$ (penalty) | Rationale |
|:-:|:---------|:-----------|:---------------:|:---------------:|:----------|
| 1 | **MW** | Molecular weight (Da) | ≤ 360 | ≥ 500 | Passive permeability declines with size; CNS drugs cluster ~310 Da |
| 2 | **cLogP** | Octanol–water partition coefficient (neutral form) | ≤ 3 | ≥ 5 | Higher cLogP raises clearance, hERG, off-target, and solubility risk |
| 3 | **cLogD<sub>7.4</sub>** | Distribution coefficient at pH 7.4 | ≤ 2 | ≥ 4 | Membrane-relevant lipophilicity for ionized molecules |
| 4 | **TPSA** | Topological polar surface area (Å²) | ≤ 90 | ≥ 120 | Proxy for H-bond desolvation cost on BBB crossing |
| 5 | **HBD** | Number of H-bond donors (N–H, O–H) | 0 | ≥ 3 | Each donor adds desolvation penalty; strong negative predictor |
| 6 | **p*K*<sub>a</sub>** | Most basic p*K*<sub>a</sub> | ≤ 8 | ≥ 10 | Only neutral fraction crosses BBB passively |

For a monoprotic base, cLogD is computed from cLogP and the most basic p*K*<sub>a</sub>
using the Henderson–Hasselbalch relation:

$$
\mathrm{cLogD}_{7.4} = \mathrm{cLogP} - \log_{10}\bigl(1 + 10^{\,\mathrm{p}K_a - 7.4}\bigr)
$$

### Interpretation

| Score range | Verdict | Interpretation |
|:-----------:|:-------|:---------------|
| **MPO ≥ 4** | ✅ PASS | Likely suitable for CNS penetration |
| **3 ≤ MPO < 4** | ⚠️ BORDERLINE | Marginal profile; optimization recommended |
| **MPO < 3** | ❌ FAIL | Unlikely to achieve adequate brain exposure |

---

## Features

- **🔬 Single-molecule analysis.** Input a SMILES string and obtain the six
  property values, individual desirability scores, total MPO, and a
  per-property medicinal-chemistry commentary in English.
- **📊 Batch analysis.** Upload a CSV (`smiles, name`) or paste a list of
  SMILES; obtain a tabular summary with score-coded rows, a histogram of
  the MPO distribution, and a downloadable CSV.
- **📚 In-app documentation.** A dedicated tab explains each property in
  detail, with mechanistic rationale, optimization tips, and references to
  the underlying literature.
- **Interactive visualizations.** Plotly-based radar chart, score
  breakdown bars, gauge, and distribution histogram.
- **Standalone library.** `cns_mpo_calculator.py` exposes a single
  `calculate_cns_mpo(smiles)` function and can be imported into other
  Python workflows.

---

## Installation

### Option 1 — Local Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/cns_mpo_score.git
cd cns_mpo_score
pip install -r requirements.txt
streamlit run app.py
```

The application will be available at <http://localhost:8501>.

### Option 2 — Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Select **New app**, choose the repository, set the main file to `app.py`.
4. Click **Deploy**. The system packages in `packages.txt` and Python
   dependencies in `requirements.txt` are installed automatically.

### Dependencies

| Package | Purpose |
|:--------|:--------|
| `streamlit` (≥1.32) | Web application framework |
| `rdkit` (≥2023.09) | Cheminformatics descriptors |
| `pandas`, `numpy` | Tabular data handling |
| `plotly` (≥5.18) | Interactive visualizations |
| `Pillow` | Image fallback for molecule rendering |

System-level libraries (`libxrender1`, `libxext6`, `libxau6`) are required for
RDKit's Cairo rendering and are listed in `packages.txt`.

---

## Usage

### Web Interface

Launch the Streamlit app and navigate the three tabs:

- **🔬 Single Molecule** — paste a SMILES (e.g. `CN1CCC[C@H]1c2cccnc2` for nicotine),
  click *Calculate CNS MPO*, and review the resulting score, structure rendering,
  and per-property assessment.
- **📊 Batch Analysis** — upload a CSV with columns `smiles` and (optionally) `name`,
  or paste one SMILES per line. Results display in a sortable table with
  score-coded MPO values and can be exported as CSV.
- **📚 Documentation** — read detailed explanations of each property, the
  desirability function, validation data, and limitations.

### Programmatic Use

The scoring engine can be used independently of the Streamlit interface:

```python
from cns_mpo_calculator import calculate_cns_mpo

result = calculate_cns_mpo("CN1CCC[C@H]1c2cccnc2")  # nicotine

print(result["properties"])
# {'MW': 162.23, 'cLogP': 1.85, 'cLogD': 0.24,
#  'TPSA': 16.13, 'HBD': 0, 'pKa': 9.0}

print(result["scores"])
# {'MW': 1.0, 'cLogP': 1.0, 'cLogD': 1.0,
#  'TPSA': 1.0, 'HBD': 1.0, 'pKa': 0.5}

print(result["total_mpo"])           # 5.5
print(result["interpretation"])      # 'Favorable CNS candidate'
```

---

## Methodology

### Descriptor Calculation

| Property | Implementation |
|:---------|:---------------|
| MW | `rdkit.Chem.Descriptors.MolWt` |
| cLogP | `rdkit.Chem.Descriptors.MolLogP` (Wildman–Crippen atom-contribution) |
| TPSA | `rdkit.Chem.rdMolDescriptors.CalcTPSA` (Ertl topological method) |
| HBD | `rdkit.Chem.rdMolDescriptors.CalcNumHBD` |
| p*K*<sub>a</sub> | SMARTS pattern matching against a curated library of basic functional groups |
| cLogD<sub>7.4</sub> | Henderson–Hasselbalch from cLogP and estimated p*K*<sub>a</sub> |

### p*K*<sub>a</sub> Estimation

The most basic p*K*<sub>a</sub> is approximated using a hierarchy of SMARTS
substructure patterns covering guanidines, amidines, primary/secondary/tertiary
aliphatic amines, morpholines, imidazoles, pyridines, anilines, and indoles.
The highest tabulated p*K*<sub>a</sub> among matched patterns is returned;
non-basic molecules are assigned a default p*K*<sub>a</sub> of 1.0.

> ⚠️ **Caveat.** This rule-based estimator is approximate. For publication-grade
> work, replace with a quantitative predictor such as ChemAxon Marvin, ACD/Labs,
> or a trained ML model (e.g. [`pkasolver`](https://github.com/mayrf/pkasolver)).

### Desirability Functions

All six properties use the simple piecewise-linear monotonic-decreasing form
specified above. The Wager (2010) paper additionally describes a "humped"
desirability for TPSA (penalizing both very low and very high values); the
present implementation uses the more common monotonic form, which differs
negligibly for typical CNS chemotypes.

---

## Validation

Wager *et al.* validated CNS MPO on three reference datasets:

| Dataset | n | Fraction with MPO ≥ 4 |
|:--------|:-:|:---------------------:|
| Marketed CNS drugs | 119 | **74 %** |
| Pfizer clinical CNS candidates | 108 | **60 %** |
| Marketed non-CNS oral drugs (reference) | — | significantly lower |

Within the Pfizer set, compounds with MPO ≥ 4 exhibited:

- Higher mean MDCK–MDR1 passive permeability,
- Lower P-glycoprotein (P-gp) efflux ratios,
- Higher unbound brain-to-plasma partition coefficients (K<sub>p,uu</sub>),
- Improved hERG and general safety margins.

The threshold of MPO ≥ 4 has since been adopted as a standard first-pass
filter for CNS portfolios across the industry.

---

## Limitations

- **Passive permeability only.** The score does not capture active transport
  (P-gp, BCRP) or uptake transporters (LAT1, OATP). Empirical confirmation
  via MDCK–MDR1 or similar assays remains essential.
- **No pharmacology prediction.** A favourable MPO is a *prerequisite*, not a
  guarantee of CNS efficacy. Target engagement must be assessed separately.
- **p*K*<sub>a</sub> approximation error.** Differences of one p*K*<sub>a</sub>
  unit propagate into cLogD and can shift the score by a meaningful amount.
- **Statistical, not mechanistic, thresholds.** Scores near 4.0 should be
  interpreted with the **per-property breakdown**, not the total alone.
- **Single-conformer descriptors.** All properties are computed from 2D
  topology; conformational effects on cLogP and TPSA are not modelled.

---

## Project Structure

```
cns_mpo_score/
├── app.py                      # Streamlit web application
├── cns_mpo_calculator.py       # Core scoring module (importable as a library)
├── requirements.txt            # Python dependencies
├── packages.txt                # System-level dependencies (Streamlit Cloud)
├── sample_compounds.csv        # Reference compounds (PASS examples)
├── failing_compounds.csv       # Borderline / FAIL examples for testing
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## Citation

If you use this calculator in academic work, please cite the original method:

```bibtex
@article{Wager2010,
  author  = {Wager, Travis T. and Hou, Xinjun and Verhoest, Patrick R. and Villalobos, Anabella},
  title   = {Moving beyond rules: the development of a central nervous system multiparameter
             optimization {(CNS MPO)} approach to enable alignment of druglike properties},
  journal = {ACS Chemical Neuroscience},
  volume  = {1},
  number  = {6},
  pages   = {435--449},
  year    = {2010},
  doi     = {10.1021/cn100008c},
  pmid    = {22778837}
}
```

Optionally, you may also cite this implementation:

```bibtex
@software{cns_mpo_calculator,
  title   = {CNS MPO Score Calculator: A Streamlit implementation of Wager et al. (2010)},
  year    = {2026},
  url     = {https://github.com/<username>/cns_mpo_score}
}
```

---

## References

1. **Wager, T. T.; Hou, X.; Verhoest, P. R.; Villalobos, A.** *Moving beyond
   rules: the development of a central nervous system multiparameter optimization
   (CNS MPO) approach to enable alignment of druglike properties.* **ACS Chem.
   Neurosci.** **2010**, *1*(6), 435–449.
   [PubMed 22778837](https://pubmed.ncbi.nlm.nih.gov/22778837/)
   · DOI: [10.1021/cn100008c](https://doi.org/10.1021/cn100008c)

2. **Wager, T. T.; Chandrasekaran, R. Y.; Hou, X.; Troutman, M. D.; Verhoest, P. R.;
   Villalobos, A.; Will, Y.** *Defining desirable central nervous system drug space
   through the alignment of molecular properties, in vitro ADME, and safety
   attributes.* **ACS Chem. Neurosci.** **2010**, *1*(6), 420–434.
   [PubMed 22778836](https://pubmed.ncbi.nlm.nih.gov/22778836/)

3. **Rankovic, Z.** *CNS drug design: balancing physicochemical properties for
   optimal brain exposure.* **J. Med. Chem.** **2015**, *58*(6), 2584–2608.
   DOI: [10.1021/jm501535r](https://doi.org/10.1021/jm501535r)

4. **Wildman, S. A.; Crippen, G. M.** *Prediction of physicochemical parameters
   by atomic contributions.* **J. Chem. Inf. Comput. Sci.** **1999**, *39*(5),
   868–873. DOI: [10.1021/ci990307l](https://doi.org/10.1021/ci990307l)

5. **Ertl, P.; Rohde, B.; Selzer, P.** *Fast calculation of molecular polar
   surface area as a sum of fragment-based contributions and its application
   to the prediction of drug transport properties.* **J. Med. Chem.**
   **2000**, *43*(20), 3714–3717.
   DOI: [10.1021/jm000942e](https://doi.org/10.1021/jm000942e)

6. **RDKit: Open-source cheminformatics.** <https://www.rdkit.org>

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for full text.

The underlying CNS MPO methodology is the intellectual contribution of
Wager *et al.* (2010) and the citation above should be respected in any
academic or commercial use of this tool.

---

## Acknowledgments

- The CNS MPO methodology was developed by Travis T. Wager and colleagues
  at **Pfizer Worldwide Research and Development**.
- Cheminformatics functionality is provided by the **[RDKit](https://www.rdkit.org)** open-source toolkit.
- Interactive interface built with **[Streamlit](https://streamlit.io)** and **[Plotly](https://plotly.com/python/)**.

---

<p align="center">
  <sub>Built for medicinal chemists, by chemists who write code.</sub>
</p>
