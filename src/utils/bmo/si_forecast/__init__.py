"""Production BF2 hot-metal silicon forecast.

This package is intentionally separate from :mod:`utils.bmo.si_prediction`.
The legacy service predicts Si for a proposed blend and still feeds the BMO
coke correction. This package predicts the furnace's actual operation on a
five-minute issue clock and has its own deployment and retraining lifecycle.
"""

from utils.bmo.si_forecast.model import SiliconExtraTreesPredictor

__all__ = [
    "SiliconExtraTreesPredictor",
]
