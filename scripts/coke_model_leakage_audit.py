"""Leakage audit for the V5 coke-rate model. Run this before trusting its R2.

Run:  python scripts/coke_model_leakage_audit.py

The V5 script reports R2 +0.43 with 14 physics features against a 125-feature
baseline at +0.36, and I initially reported that as physics beating plant data.
This audit shows it is mostly one control variable.

WHAT IT CHECKS.

  A  Does the cruising filter turn PCI into an accounting identity for coke?
     The filter keeps only total fuel in [500,600] and nut coke sits near 70,
     so coke = TotalFuel - 70 - PCI almost by construction.
  B  Shared denominator. Every feature and the target are divided by production,
     which can manufacture correlation out of nothing.
  C  Ablation. How much does PCI alone explain, and what is left without it?

THE ANSWER, on the test period: PCI alone scores R2 0.395 against the full 14
features' 0.408, and PCI plus two blast terms scores 0.576 - better than all
fourteen. Removing the single PCI-bearing feature drops the set to 0.248.

That 0.248 is the number that matters for a blend optimiser: how much of coke
variance the burden explains at FIXED PCI. See docs/coke_rate_model_v5_review.md.
"""

import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv(".env")
import numpy as np, pandas as pd
from train_bmo_fuel_v5 import load_base, add_engineered, num, evaluate

df = load_base()
df, energy = add_engineered(df)
cut = int(len(df) * 0.8); te = df.iloc[cut:]
coke = te["target"]; pci = num(te, "PCI_KG/THM"); tf = num(te, "ACT. FUEL RATEKG/THM.")
prod = num(te, "PRODUCTIONTONNESPERHR")

print("=== A. IS COKE JUST (TOTAL FUEL - NUT - PCI)? ===")
print(f"  corr(coke, PCI)        = {coke.corr(pci):+.3f}")
print(f"  corr(coke, TotalFuel)  = {coke.corr(tf):+.3f}")
implied = tf - 70.0 - pci
print(f"  corr(coke, TF-70-PCI)  = {coke.corr(implied):+.3f}")
print(f"  TotalFuel sd {tf.std():.1f}  PCI sd {pci.std():.1f}  coke sd {coke.std():.1f}")

print("\n=== B. SHARED DENOMINATOR: does production drive both? ===")
print(f"  production sd {prod.std():.2f} t/hr  (mean {prod.mean():.1f})")
print(f"  corr(coke, production)              = {coke.corr(prod):+.3f}")
print(f"  corr(FE_INPUT_KG_THM, production)   = {te['FE_INPUT_KG_THM'].corr(prod):+.3f}")
print(f"  corr(coke, FE_INPUT_KG_THM)         = {coke.corr(te['FE_INPUT_KG_THM']):+.3f}")
print(f"  corr(coke, IMPLIED_COKE_KG_THM)     = {coke.corr(te['IMPLIED_COKE_KG_THM']):+.3f}")

print("\n=== C. HOW MUCH DOES PCI ALONE EXPLAIN? ===")
for name, feats in (
    ("PCI only",                 ["PCI_KG/THM"]),
    ("PCI + blast only",         ["PCI_KG/THM", "BLAST_NM3_THM", "BLAST_SENSIBLE_MJ_THM"]),
    ("energy, NO pci-derived",   [c for c in energy if c not in ("IMPLIED_COKE_KG_THM",)]),
    ("energy, ALL 14",           energy),
    ("energy MINUS pci entirely",[c for c in energy if c not in ("IMPLIED_COKE_KG_THM","IMPLIED_CARBON_KG_THM","ENERGY_DEFICIT_MJ_THM")]),
):
    evaluate(df, feats, chronological=True, label=name)
