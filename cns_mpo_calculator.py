"""
CNS MPO (Multiparameter Optimization) Score Calculator
Reference: Wager et al., ACS Chem. Neurosci. 2010, 1, 435–449

Six properties scored from 0–1, total CNS MPO = 0–6
CNS MPO ≥ 4 is considered favorable for CNS drug candidates
"""

import math
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ──────────────────────────────────────────────
# pKa Estimation via SMARTS pattern matching
# Order: highest pKa first (so we pick the most basic group)
# ──────────────────────────────────────────────
BASIC_SMARTS = [
    # Guanidine / amidine (pKa ~12–13)
    ("[NH,NH2,NX3H0]C(=[NH,NH2])[NH,NH2,NX3H0]", 12.5, "Guanidine"),
    ("[NH2,NH]C(=[NH])", 11.5, "Amidine"),

    # Primary aliphatic amine (pKa ~10)
    ("[NH2;!$(NC=O);!$(NC=N);!$(Nc);!$(N=*)]", 10.0, "Primary amine"),

    # Secondary aliphatic amine (pKa ~9.5)
    ("[NH;!$(NC=O);!$(NC=N);!$(Nc);!$(N=*);!$([nH])]", 9.5, "Secondary amine"),

    # Tertiary aliphatic amine including piperidine/pyrrolidine (pKa ~9)
    ("[N;H0;!$(NC=O);!$(NC=N);!$(Nc);!$([n]);!$(N=*);!$(N[OH]);!$(N~[!#6!#1])]",
     9.0, "Tertiary amine"),

    # Morpholine N (pKa ~7.5) — tertiary N flanked by two CH2 and one O
    ("N1CCOCC1", 7.5, "Morpholine"),

    # Imidazole aromatic N (pKa ~7)
    ("c1cnc[nH]1", 7.0, "Imidazole"),
    ("c1c[nH]nc1", 7.0, "Imidazole (alt)"),

    # Benzimidazole (pKa ~5.5)
    ("c1ccc2[nH]cnc2c1", 5.5, "Benzimidazole"),

    # Pyridine-like aromatic N (pKa ~5)
    ("[n;r6;H0;+0;!$([n]~[n]);!$([n]~[o]);!$([n]~[s])]", 5.0, "Pyridine"),

    # Pyrimidine (pKa ~1–3 — less basic)
    ("c1cnccn1", 1.5, "Pyrimidine"),

    # Primary aromatic amine / aniline (pKa ~4.5)
    ("[NH2;$(Nc);!$(NC=O)]", 4.5, "Aniline"),

    # Indole N-H (pKa ~0, non-basic but proton donor)
    ("c1ccc2[nH]ccc2c1", 0.0, "Indole"),
]


def estimate_most_basic_pka(mol: Chem.Mol) -> float:
    """
    Estimate the most basic pKa of a molecule using SMARTS patterns.
    Returns the highest pKa found (most basic nitrogen).
    Returns 1.0 if no basic nitrogen is found (treated as non-basic).
    """
    best_pka = 1.0  # non-basic default → will get score=1 in MPO

    for smarts, pka, name in BASIC_SMARTS:
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                if pka > best_pka:
                    best_pka = pka
        except Exception:
            continue

    return best_pka


def calc_clogd(clogp: float, pka: float, ph: float = 7.4) -> float:
    """
    Estimate cLogD at given pH using Henderson-Hasselbalch for a monoprotic base.
    cLogD = cLogP − log10(1 + 10^(pKa − pH))
    For non-basic molecules (pKa ≪ pH), cLogD ≈ cLogP.
    """
    return clogp - math.log10(1.0 + 10 ** (pka - ph))


# ──────────────────────────────────────────────
# Individual MPO desirability function
# ──────────────────────────────────────────────
def mpo_desirability(value: float, lower: float, upper: float) -> float:
    """
    Piecewise-linear desirability (Wager 2010):
      value ≤ lower  →  1.0
      value ≥ upper  →  0.0
      otherwise      →  linear interpolation
    """
    if value <= lower:
        return 1.0
    elif value >= upper:
        return 0.0
    else:
        return (upper - value) / (upper - lower)


# ──────────────────────────────────────────────
# CNS MPO scoring boundaries (Wager et al. 2010)
# ──────────────────────────────────────────────
MPO_BOUNDS = {
    #  property: (lower_d1, upper_d0)
    "MW":    (360.0, 500.0),
    "cLogP": (3.0,   5.0),
    "cLogD": (2.0,   4.0),
    "TPSA":  (90.0,  120.0),
    "HBD":   (0.0,   3.0),
    "pKa":   (8.0,   10.0),
}

MPO_UNITS = {
    "MW":    "Da",
    "cLogP": "",
    "cLogD": "",
    "TPSA":  "Å²",
    "HBD":   "",
    "pKa":   "",
}

MPO_IDEAL = {
    "MW":    "≤ 360 Da",
    "cLogP": "≤ 3",
    "cLogD": "≤ 2",
    "TPSA":  "≤ 90 Å²",
    "HBD":   "0",
    "pKa":   "≤ 8",
}


# ──────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────
def calculate_cns_mpo(smiles: str) -> dict | None:
    """
    Calculate CNS MPO score from a SMILES string.

    Returns
    -------
    dict with keys:
        - mol         : RDKit Mol object
        - properties  : dict of calculated property values
        - scores      : dict of individual MPO scores (0–1)
        - total_mpo   : float, sum of scores (0–6)
        - interpretation : str, human-readable rating
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    # Calculate six properties
    mw    = Descriptors.MolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hbd   = float(rdMolDescriptors.CalcNumHBD(mol))
    pka   = estimate_most_basic_pka(mol)
    clogd = calc_clogd(clogp, pka)

    props = {
        "MW":    round(mw,    2),
        "cLogP": round(clogp, 2),
        "cLogD": round(clogd, 2),
        "TPSA":  round(tpsa,  2),
        "HBD":   int(hbd),
        "pKa":   round(pka,   1),
    }

    # Score each property
    scores = {
        prop: round(mpo_desirability(props[prop], *MPO_BOUNDS[prop]), 3)
        for prop in MPO_BOUNDS
    }

    total = round(sum(scores.values()), 3)

    # Interpretation
    if total >= 4.0:
        interpretation = "Favorable CNS candidate"
        color = "green"
    elif total >= 3.0:
        interpretation = "Moderate CNS profile"
        color = "orange"
    else:
        interpretation = "Poor CNS profile"
        color = "red"

    return {
        "mol":            mol,
        "properties":     props,
        "scores":         scores,
        "total_mpo":      total,
        "interpretation": interpretation,
        "color":          color,
    }
