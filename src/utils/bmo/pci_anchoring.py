"""Coke rate across the whole PCI range, model inside and anchored outside.

WHY THIS EXISTS.

The optimal-coke-rate model is fitted on the PCI band the plant actually runs.
The record spans 135-208 kg/tHM, with 97 of 187 days between 175 and 190, so
outside roughly 150-210 there is no data and a fitted model is extrapolating.

A UI number box does not care about that. An operator can type 0 or 250, and an
extrapolating tree model will answer with something confident and arbitrary.

So the range is split in three:

    PCI < 150        interpolate from the zero-PCI anchor to the model at 150
    150 <= PCI <= 210  the model, which is where its data is
    PCI > 210        interpolate from the model at 210 to the 250 anchor

Both interpolations END ON THE MODEL'S OWN VALUE at the band edge, so there is
no step at the join - the curve is continuous, and what the operator sees at
149 is what they see at 151.

WHERE THE ANCHORS COME FROM.

They are the plant's expected all-coke and maximum-injection operating points,
supplied by the process team, not fitted:

    PCI    0 kg/tHM  ->  coke 460, nut coke 70   (total fuel 530)
    PCI  250 kg/tHM  ->  coke 275, nut coke 70   (total fuel 595)

Two things make them credible rather than arbitrary. The implied end-to-end
substitution is (460 - 275) / 250 = 0.74 kg coke per kg PCI, which falls inside
the 0.74-0.83 measured from plant history by PCI regime. And the total-fuel
envelope they imply, 530-595, is the range the process team expects.

For reference, the energy balance with its calibration offset applied gives
roughly 477 at zero PCI and 254 at 250 - so the anchors sit about 17 kg/tHM
below and 21 above it. That is a real disagreement, not a rounding difference,
and it is the process team's number that is used here.

WHY LINEAR AND NOT A SPLINE.

Linear interpolation is monotone by construction and cannot overshoot, which
matters when the whole point is to stop a model inventing values. A cubic
matching the model's slope at the join would remove the kink in the derivative,
but it can overshoot between knots, and an overshoot here is exactly the failure
this module exists to prevent. The value is continuous; the slope is not, and
that is the safer trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# The band the model is trusted in. Outside it, interpolate to an anchor.
MODEL_PCI_MIN = 150.0
MODEL_PCI_MAX = 210.0

# Process-team operating points. coke kg/tHM at the stated PCI.
ANCHOR_LOW_PCI = 0.0
ANCHOR_LOW_COKE = 460.0
ANCHOR_HIGH_PCI = 250.0
ANCHOR_HIGH_COKE = 275.0

# Nut coke is a fixed charge at this plant and is not a function of PCI.
NUT_COKE_KG_THM = 70.0


@dataclass(frozen=True)
class AnchoredCokeRate:
    """A coke rate, and an honest account of where it came from."""

    coke_kg_per_thm: float
    nut_coke_kg_per_thm: float
    pci_kg_per_thm: float
    basis: str            # "model" | "interpolated_low" | "interpolated_high" | "clamped"
    note: str = ""

    @property
    def total_fuel_kg_per_thm(self) -> float:
        return self.coke_kg_per_thm + self.nut_coke_kg_per_thm + self.pci_kg_per_thm

    @property
    def is_model(self) -> bool:
        """True only inside the band the model was fitted on."""

        return self.basis == "model"


def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Straight line through two points. x1 == x0 returns y0 rather than dividing."""

    if x1 == x0:
        return float(y0)
    t = (float(x) - x0) / (x1 - x0)
    return float(y0) + t * (float(y1) - float(y0))


def anchored_coke_rate(
    pci_kg_per_thm: float,
    model: Callable[[float], float],
    *,
    model_pci_min: float = MODEL_PCI_MIN,
    model_pci_max: float = MODEL_PCI_MAX,
) -> AnchoredCokeRate:
    """Coke rate at this PCI: the model inside its band, anchored outside it.

    Args:
         - pci_kg_per_thm: float - PCI rate the operator has set.
         - model: Callable[[float], float] - The optimal-coke-rate model. Called
           only at PCI values inside the band, including exactly at its two
           edges, which is what makes the joins continuous.
         - model_pci_min \\ model_pci_max: float - The band the model is trusted
           in. Defaults are the plant's data coverage.

    Returns:
         - return AnchoredCokeRate - Coke rate plus the basis it came from, so a
           caller can tell a modelled number from an interpolated one and say so
           on screen.
    """

    pci = float(pci_kg_per_thm)

    if pci < ANCHOR_LOW_PCI:
        return AnchoredCokeRate(
            coke_kg_per_thm=ANCHOR_LOW_COKE, nut_coke_kg_per_thm=NUT_COKE_KG_THM,
            pci_kg_per_thm=ANCHOR_LOW_PCI, basis="clamped",
            note=f"PCI below {ANCHOR_LOW_PCI:g} is not meaningful; clamped.",
        )
    if pci > ANCHOR_HIGH_PCI:
        return AnchoredCokeRate(
            coke_kg_per_thm=ANCHOR_HIGH_COKE, nut_coke_kg_per_thm=NUT_COKE_KG_THM,
            pci_kg_per_thm=ANCHOR_HIGH_PCI, basis="clamped",
            note=(f"PCI above {ANCHOR_HIGH_PCI:g} kg/tHM is beyond anything this "
                  "furnace has run; clamped to the anchor."),
        )

    if model_pci_min <= pci <= model_pci_max:
        return AnchoredCokeRate(
            coke_kg_per_thm=float(model(pci)),
            nut_coke_kg_per_thm=NUT_COKE_KG_THM, pci_kg_per_thm=pci, basis="model",
        )

    if pci < model_pci_min:
        edge = float(model(model_pci_min))
        return AnchoredCokeRate(
            coke_kg_per_thm=_lerp(pci, ANCHOR_LOW_PCI, ANCHOR_LOW_COKE,
                                  model_pci_min, edge),
            nut_coke_kg_per_thm=NUT_COKE_KG_THM, pci_kg_per_thm=pci,
            basis="interpolated_low",
            note=(f"Below {model_pci_min:g} kg/tHM the model has no data. "
                  f"Interpolated between the {ANCHOR_LOW_COKE:g} kg/tHM all-coke "
                  f"anchor and the model's {edge:,.1f} at {model_pci_min:g}."),
        )

    edge = float(model(model_pci_max))
    return AnchoredCokeRate(
        coke_kg_per_thm=_lerp(pci, model_pci_max, edge,
                              ANCHOR_HIGH_PCI, ANCHOR_HIGH_COKE),
        nut_coke_kg_per_thm=NUT_COKE_KG_THM, pci_kg_per_thm=pci,
        basis="interpolated_high",
        note=(f"Above {model_pci_max:g} kg/tHM the model has no data. "
              f"Interpolated between the model's {edge:,.1f} at {model_pci_max:g} "
              f"and the {ANCHOR_HIGH_COKE:g} kg/tHM anchor at "
              f"{ANCHOR_HIGH_PCI:g}."),
    )


def substitution_ratio(
    pci_kg_per_thm: float,
    model: Callable[[float], float],
    *,
    step: float = 5.0,
    **kwargs,
) -> float:
    """Local d(coke)/d(PCI) at this PCI, by central difference.

    Reported rather than assumed constant. Plant history gives -0.83 at 168
    kg/tHM falling to -0.74 at 197 - the raceway cannot burn the extra coal as
    completely, so each further kg replaces less coke. A single fixed ratio
    cannot express that, and the energy balance's -0.86 carbon equivalence in
    particular cannot: it sees only carbon.
    """

    low = anchored_coke_rate(max(0.0, pci_kg_per_thm - step), model, **kwargs)
    high = anchored_coke_rate(pci_kg_per_thm + step, model, **kwargs)
    span = high.pci_kg_per_thm - low.pci_kg_per_thm
    if span <= 0.0:
        return 0.0
    return (high.coke_kg_per_thm - low.coke_kg_per_thm) / span
