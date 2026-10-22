"""STRATA-Learn: continuous learning for STRATA-omega, shadow mode (v1).

The learner never changes the deployed model or the page. It reads the daily evidence (the change of omega against
its 14-day anchor and the model channels of the day), tracks the plant-data scales with a random-walk Kalman filter,
compares the tracked parameters (challenger) with the deployed asset (champion) out of sample, watches for drift, and
writes proposals for the fact register and the asset. Value and uncertainty updates within two standard errors are
marked automatic; status changes, support-range growth and promotion of the challenger need an engineer's approval.
The 14-day calibration anchor of the Coke Energy pipeline is kept: the learner works on day-minus-anchor changes only.

  evidence.py   evidence store (daily rows, blend family, trial flag) and the seed history
  kalman.py     random-walk Kalman filter on (lam_pellet, lam_lump, lam_production, lam_other, a_moisture), AR(1)
  gate.py       champion-challenger on a rolling window (Diebold-Mariano), CUSUM drift alarm
  autofill.py   register proposals (GAS-14, GAS-18 support, VAL-01..12) with approval flags; apply after approval
  explore.py    constrained exploration: Thompson-sampling probability of optimality under a regret budget
  run.py        nightly shadow run: python -m utils.burden_engine.learn.run
"""
