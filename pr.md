# Add governed charged-coke and live HM Si forecasting

## Summary

This change upgrades the BF2 Blend Mix Optimiser with two independent advisory forecasts while preserving the existing optimisation, blend-response and costing behaviour:

- a five-hour charged-coke forecast for the Data-Driven coke-rate option;
- a live HM Si forecast with causal process, charge and laboratory inputs;
- governed retrain, review, approval and rollback flows for both models;
- compact operator-facing trends, diagnostics and snapshot reporting.

## Charged-coke forecast

- Forecasts the four-hour charged coke rate at +1 h through +5 h.
- Uses the shipped `20261007_0400` bundle unless an approved deployment is available at runtime.
- Keeps the +5 h forecast as the optimiser input and shows all five horizons in the trend.
- Adds per-horizon ranges, seven-day forecast-versus-actual monitoring and comparison with holding the current rate.
- Supports background retraining, candidate review, supervisor/admin approval and rollback.
- Adds the Data-Driven `force-predict` control: an out-of-range case can still produce an advisory value while retaining the exact warning and violations.
- Fixes the temporary candidate-blend slag response at `0.10 kg coke/kg slag`; it remains locked for operators and configurable by administrators.
- Removes the obsolete Robust 24 h option from the operator workflow.

## Live HM Si forecast

- Integrates the supplied ExtraTrees model as a live production service rather than a frozen result.
- Uses completed ten-minute Influx bins, causal charge records and real irregular HM Si samples.
- Preserves both sample time and database `created_at`; late laboratory entries never rewrite an earlier issued forecast.
- Does not interpolate artificial HM Si observations. Uneven laboratory spacing is represented through the latest Si, its age, the previous Si and the mean of the latest three samples.
- Supports the current legacy +2 h bundle and a version-2 retraining path for `Now`, `+1 h`, `+2 h` and `+3 h` as one approved deployment.
- Refreshes the advisory panel every five minutes, shows the current Si result with the optimiser output, and plots rolling model inference against irregular laboratory samples.
- Adds a full seven-day IST trend with actual samples, model inference, horizon scoring, persistence comparison, bias, coverage and matched-sample counts.
- Treats missing blast flow and other operating checks as explicit warnings when the regression features remain usable.
- Removes the old proposed-blend Si model and its accuracy panel from the active BMO path.

## Data, snapshots and reporting

- Extends Influx query support needed by the versioned Si channel map and completed-bin aggregation.
- Refreshes the furnace dataset and associated cache metadata used by BMO model workflows.
- Stores the complete charged-coke and HM Si forecast paths, ranges, issue times, model versions and warnings in BMO snapshots and reports.
- Keeps online timestamps in UTC internally and converts all operator plots and labels to IST; offline database timestamps retain their India-time contract.
- Excludes local runtime ledgers, accepted/retrained deployment directories and scratch integration output from version control. Shipped model bundles and reproducible test fixtures remain versioned.

## Validation

- Full BMO regression suite: **525 passed, 1 skipped**.
- Charged-coke tests cover feature parity, frozen replay, policy gates, force-predict behaviour, scenarios, live service, training, approval and rollback.
- HM Si tests cover ten-minute feature construction, irregular/late lab samples, five-minute target alignment, UTC/IST handling, warnings, rolling replay, per-horizon scoring, immutable forecasts, retraining, approval, rollback and legacy bundles.
- Snapshot tests cover persistence and report rendering for both forecast paths.
- Python compilation and `git diff --check` passed.

## UAT checks

1. Confirm Energy Balance still shows model accuracy and the independent HM Si advisory.
2. Confirm Data-Driven shows +1 h to +5 h charged-coke values and the +5 h value reaches the optimiser.
3. Exercise `force-predict` on an out-of-range row and verify the value and exact warning appear together.
4. Confirm HM Si charts show IST timestamps, irregular laboratory points and a populated seven-day inferred trend.
5. Run each retrain flow, review the candidate without changing production, then test approve and rollback with a supervisor/admin account.
6. Save and reopen a BMO snapshot and verify both complete forecast paths and warnings are retained.

## Rollback

- Charged coke falls back to the shipped bundle when no approved runtime deployment exists.
- HM Si format-v1 bundles remain readable, and an approved version can be rolled back from the model panel.
- The HM Si forecast remains advisory and does not alter blend feasibility, optimiser costs or charged-coke inference.
