# BF2 live HM Si forecast

The production HM Si model is an advisory furnace forecast. It does not change
blend feasibility, optimiser cost or the charged-coke forecast. The retired
proposed-blend Si model is no longer used by BMO.

## What the model predicts

A format-v2 deployment issues four values every five minutes: Si if sampled
now, then Si at +1 h, +2 h and +3 h. Each target is a five-minute bin containing
a real raw laboratory sample. Training and accuracy calculations never
interpolate Si between samples.

The database sample clock and availability clock are both preserved. A sample
can be used as an input only when its sample time is earlier than the issue and
its `created_at` is no later than the issue. A late-entered result is plotted at
its real sample time but never changes an earlier saved live forecast.

Uneven sample spacing is preserved rather than regularised. Each issue uses the
latest available Si value, its age in hours, the preceding real Si value and the
mean of the latest three real values. A two-hour gap therefore changes the age
of the latest value; it is not filled with synthetic hourly or ten-minute Si.
The selected 40-feature model does **not** separately encode the older
sample-to-sample gaps, a time-normalised Si slope, cast sequence or time of day.
Those are sensible candidate features for the next retraining study, but adding
them changes the model and needs honest holdout evidence before deployment.

The trend is populated before the new live ledger matures by causally replaying
the active bundle across the last seven days. The first tab displays its latest
24 hours; the Last 7 days tab displays the continuous inferred model series and
raw assays at their irregular timestamps. Replay results are never written to
the immutable live-forecast ledger.

## Inputs

The selected model has 40 engineered features:

- Four prior-Si features: latest value, its age, the previous value and the mean
  of the latest three available samples.
- Thirty-four process and thermal features built from 51 versioned Influx
  channels. They cover top temperature, furnace-body temperature and pressure
  drop, stave Delta-T and heat load, cooling-water flow, charge rate, coal rate
  and production over 0-1 h, 3-6 h and 6-12 h windows plus selected four-hour
  changes. Only completed ten-minute bins are used, with one-bin latency.
- Two raw-charge features from events 1-3 h before issue: total ferrous burden
  divided by coke plus nut coke, and pellet tonnes per hour.

Previous-hour hot blast, PCI, production, CO + CO2 and fuel rate are operating
checks. They are shown as warnings but are not regression inputs. A missing
blast-volume reading therefore does not hide an otherwise computable advisory
prediction.

The selected feature set does not use slag composition, other HM chemistry or
raw-material composition. Those require a separate timestamp/provenance study
before they can be added without leakage.

## Deployment lifecycle

Retraining uses up to 60 days and leaves the final seven days untouched for
testing. Each horizon is a separate ExtraTrees model with its own 90% error
range calibrated within the training period. A candidate is deployed only after
all causal, sample-count, bias and persistence checks pass and a supervisor or
administrator accepts it. The four horizons activate and roll back together.

Format-v1 hourly 2-3 h bundles remain readable for rollback. The UI labels them
as legacy and shows their single point until a format-v2 candidate is accepted.
