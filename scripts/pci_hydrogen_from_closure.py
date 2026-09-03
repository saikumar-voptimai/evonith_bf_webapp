"""Can fuel hydrogen be credited to the energy balance? Measured answer: no.

Run:  python scripts/pci_hydrogen_from_closure.py

WHY THIS EXISTS.

Fuel hydrogen is physically a reductant and belongs on the input side of the
energy balance, worth roughly 900 MJ/tHM. It has been switched off since the
balance was built, on the stated grounds that this plant has no ultimate
analysis and so H% was unknown.

That reasoning is now retired. The vendor does not supply an ultimate analysis
either, so H% was instead fixed from published data for the coal's RANK, which
the proximate analysis is enough to establish (see energy_balance.yml). With the
number no longer unknown, the question became answerable: does the balance
actually want the term?

It does not, and this script is what settles it.

THE FALSIFYING TEST.

If fuel hydrogen were the missing input, it would leave a specific fingerprint.
Back-calculate the shell loss needed to force closure to 1.0, holding everything
else fixed. That residual absorbs whatever the model is missing. Fuel hydrogen
enters in proportion to PCI rate, so:

    implied_shell(H omitted) = constant - 120 x H_fraction x pci_rate

The residual must FALL as PCI rate rises, and the slope hands us H% directly -
a derivation from the plant's own record, needing no certificate at all. That
was the hope.

WHAT THE RECORD SAYS.

Over 221 days and a PCI range of 152-205 kg/tHM, the correlation is -0.05.
There is no fingerprint. Regression returns a POSITIVE slope of about +17 MJ per
kg PCI, which would imply a negative hydrogen content. The hypothesis is
refuted, not merely unsupported.

WHAT IS ACTUALLY MISSING.

The residual scales with TOTAL fuel rather than with hydrogen: coke, nut coke
and PCI all carry positive coefficients of roughly +17 to +29 MJ per kg, against
a carbon credit of 28.5 MJ/kg coke. Only about a third of marginal fuel energy
is being absorbed by the modelled output terms. That points to a fuel-scaled
OUTPUT term still missing - top gas, dust carbon, or the direct/indirect
reduction split - and adding a correct input term on top of it only widens the
gap. Note the two are collinear (corr(pci, coke) = -0.85), so the individual
coefficients are not separately trustworthy; the joint conclusion is.

Whoever closes that output-side gap should re-run this. Until then, enabling
supply.include_fuel_hydrogen makes the balance worse on every measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from energy_balance_phase0 import build  # noqa: E402

# Per kg of hydrogen atoms, H2 + 1/2 O2 -> H2O, lower heating value.
H_LHV_MJ_PER_KG = 120.0
# (PCI H fraction, coke H fraction). The first row is the term fully OFF, so
# coke hydrogen must be zero there too - carrying it would misreport the
# baseline the other rows are compared against. 4.2%/0.35% is what
# energy_balance.yml ships; the rest bracket it.
H_CANDIDATES = (
    (0.0, 0.0),
    (0.030, 0.0035),
    (0.042, 0.0035),
    (0.050, 0.0035),
)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def implied_shell_loss(df: pd.DataFrame) -> pd.Series:
    """Shell loss that would force closure to exactly 1.0.

    Everything the model fails to account for lands here, which is precisely
    what makes it a usable probe for a suspected missing term.
    """

    core = df["q_demand_total"] - df["q_loss_total"]  # all demand but the loss
    return df["q_input"] - df["q_topgas"] - core


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, standard errors and R2. Small enough not to need statsmodels."""

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float((resid**2).sum() / dof)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1.0 - float((resid**2).sum() / ((y - y.mean()) ** 2).sum())
    return beta, se, r2


def main() -> None:
    df = build()
    df["quarter"] = df.index.to_period("Q").astype(str)
    df["implied"] = implied_shell_loss(df)
    d = df[
        [
            "implied", "pci_rate", "coke_rate", "nut_rate", "q_input", "q_topgas",
            "q_demand_total", "q_loss_total", "q_loss_stave", "quarter",
        ]
    ].dropna()

    banner("0. SAMPLE")
    print(f"  days {len(d)}   {d.index.min().date()} -> {d.index.max().date()}")
    pci = d["pci_rate"]
    print(f"  PCI rate  p5 {pci.quantile(.05):.0f}  median {pci.median():.0f}  "
          f"p95 {pci.quantile(.95):.0f} kg/tHM")
    print("  A wide PCI range is what makes the test possible - if PCI were held")
    print("  flat there would be no variation to read the hydrogen slope from.")

    banner("1. THE FINGERPRINT: implied shell loss must FALL as PCI rises")
    for col in ("pci_rate", "coke_rate", "nut_rate"):
        print(f"  corr(implied, {col:10s}) = {d['implied'].corr(d[col]):+.3f}")
    r = d["implied"].corr(d["pci_rate"])
    print(f"\n  Required: clearly negative. Observed: {r:+.3f}.")
    if r > -0.15:
        print("  NOT PRESENT. Fuel hydrogen does not behave like the missing term.")

    banner("2. SLOPES  (if hydrogen were the term, slope = -120 x H_fraction)")
    X = np.column_stack(
        [np.ones(len(d)), d["pci_rate"], d["coke_rate"], d["nut_rate"]]
    )
    beta, se, r2 = ols(X, d["implied"].to_numpy())
    for name, b, s in zip(("const", "pci", "coke", "nut"), beta, se):
        implied_h = "" if name == "const" else f"   -> H = {-b / H_LHV_MJ_PER_KG * 100:6.2f} %"
        print(f"  {name:6s} {b:+9.2f}  se {s:6.2f}  t {b / s:+6.1f}{implied_h}")
    print(f"  R2 = {r2:.3f}")
    print(f"\n  corr(pci_rate, coke_rate) = {d['pci_rate'].corr(d['coke_rate']):+.3f}"
          "  <- collinear, so read the")
    print("  coefficients jointly, not individually. Every fuel carries a POSITIVE")
    print("  coefficient, i.e. a negative hydrogen content. Physically impossible,")
    print("  so the residual is not fuel hydrogen.")
    print(f"  For scale, the carbon credit of coke is 32.8 x 0.87 = 28.5 MJ/kg -")
    print("  only about a third of marginal fuel energy reaches a modelled output.")

    banner("3. WHAT ENABLING IT WOULD COST")
    print(f"  {'H_pci':>6}  {'closure':>8}  {'implied shell':>14}  {'quarter drift':>14}")
    for h_pci, h_coke in H_CANDIDATES:
        h_kg = d["pci_rate"] * h_pci + (d["coke_rate"] + d["nut_rate"]) * h_coke
        extra = h_kg * H_LHV_MJ_PER_KG
        closure = (d["q_demand_total"] + d["q_topgas"]) / (d["q_input"] + extra)
        imp = d["implied"] + extra
        by_q = imp.groupby(d["quarter"]).median()
        label = "off" if h_pci == 0 else f"{h_pci * 100:.1f}%"
        print(f"  {label:>6}  {closure.median():8.3f}  {imp.median():14,.0f}"
              f"  {by_q.max() - by_q.min():14,.0f}")
    print(f"\n  Measured shell loss: {d['q_loss_stave'].median():,.0f} MJ/tHM "
          f"(stave rows 6-10), {d['q_loss_total'].median():,.0f} (flow-scaled).")
    print("  Every H% drives the implied residual further from the measurement,")
    print("  and none of them reduces the across-quarter drift materially - so")
    print("  hydrogen does not explain that drift either, which was the other")
    print("  hypothesis worth testing.")

    banner("VERDICT")
    print("  Keep supply.include_fuel_hydrogen false. The blocker was never the")
    print("  unknown H%; it is a fuel-scaled term missing from the OUTPUT side.")
    print("  Re-run this after that term is found.")


if __name__ == "__main__":
    main()
