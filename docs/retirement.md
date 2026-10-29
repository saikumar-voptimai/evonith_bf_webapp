# Retirement of the Streamlit pages (SV-6.7, #321)

The Streamlit app stays under `/legacy` behind Caddy (single sign-on with the SPA, SV-6.6). A Streamlit page is retired
once its Svelte replacement has passed its gate, 14 days of shadow use and the shift team's sign-off. **No page is
retired yet** (29 Oct 2026): gates G2 and G3 and the sign-offs are pending.

## Gates (epic #262)

| Gate | Condition | Pages |
|---|---|---|
| G1 | Live Twin in shadow use 14 days, ≥ 99 % of scheduled jobs succeed, Δ ≤ 0.1 kg/tHM against page 10 (`scripts/twin_parity.py` on plant data) | Coke Energy Model |
| G2 | Values equal the Streamlit pages (parity tests below, relative difference ≤ 1e-6); shift-team sign-off | V-Board, FurnaceMind, Coke Energy Model, Welcome |
| G3 | 16 Oct 2026 case report and 5 recorded BMO days (30 Sep – 4 Oct 2026) reproduced | Blend Mix Optimiser, Burden & Fuel Optimiser |
| G4 | Admin writes audited; HMT forecast gate result recorded | Welcome (admin tools) |
| G5 | Ported Streamlit pages retired from the navigation (this list) | all ported pages |

## Retirement list

One row per page of `src/config/page_registry.py`; `tests/test_page_registry.py` checks the rows against the registry
(titles, retirement and deletion dates, SPA routes) and that every test named below exists.

| Streamlit page | Svelte replacement | Gate | Shadow use from | Operator feedback | Sign-off | Retired on | Deleted on |
|---|---|---|---|---|---|---|---|
| Welcome | `/`, `/admin/users` | G2, G4 | pending | pending | pending | not retired | — |
| Data Explorer | none (not ported) | — | — | — | — | not retired | — |
| V-Board | `/status` | G2 | pending | pending | pending | not retired | — |
| V-Sense | none (not ported) | — | — | — | — | not retired | — |
| CoPilot | none (not ported) | — | — | — | — | not retired | — |
| Material Balance | none (not ported) | — | — | — | — | not retired | — |
| FurnaceMind | `/furnacemind`, `/reports` | G2 | pending | pending | pending | not retired | — |
| Blend Mix Optimiser | `/bmo` | G3 | pending | pending | pending | not retired | — |
| Coke Energy Model | `/`, `/trace`, `/forge`, `/admin/calibration` | G1, G2 | pending | pending | pending | not retired | — |
| Burden & Fuel Optimiser | `/optimiser`, `/strata`, `/admin/what-if`, `/admin/burnt-vs-charged`, `/admin/validation` | G3 | pending | pending | pending | not retired | — |
| Feedback | none (not ported) | — | — | — | — | not retired | — |

"Shadow use from" is the first IST day the shift team works on the Svelte page with the Streamlit page beside it;
retirement needs 14 days of it. "Operator feedback" links the feedback tickets or notes of that period; "Sign-off" is
the name and date of the shift-team approval. "Deleted on" is retirement + 30 days.

## Per page

### Welcome (`custom_pages/1_Welcome.py`)

- **Replacement:** `/` (Live Twin, the SPA landing; SV-1.9, #289) for the KPI strip and the module tiles;
  `/admin/users` (SV-4.7, #310) for hopper mapping, burden distribution and user management.
- **Parity evidence:** `tests/test_services_plant_admin.py::test_the_twin_api_writes_the_rows_of_the_streamlit_tools`.
- **Before retirement:** Welcome is the default page (`/legacy/`); retired, the next page of the navigation (Data
  Explorer) becomes the default, for every role and in the SPA's `LEGACY_PAGES`. Its tiles link the pages that are
  not ported, which stay in the Streamlit navigation.

### V-Board (`custom_pages/3_Data_Visualisation.py`)

- **Replacement:** `/status` (Furnace status, SV-2.1 #293 / SV-2.2 #294).
- **Parity evidence:**
  `tests/test_services_status.py::test_snapshot_equals_the_streamlit_page_for_the_same_timestamp`,
  `tests/test_services_status.py::test_trend_equals_the_streamlit_trend_view`,
  `tests/test_api_status.py::test_snapshot_route_equals_the_streamlit_page`.
- **Before retirement:** the *Visualisations* section (longitudinal temperature contours over a chosen period) is not
  ported; port it or have the sign-off accept dropping it.

### FurnaceMind (`custom_pages/7_FurnaceMind.py`)

- **Replacement:** `/furnacemind` (SV-5.1–5.3, #312–#314) for AI Co-Operate; `/reports` (SV-2.3 #295, SV-2.4 #296,
  shift record SV-2.5 #297) for the Reports tab.
- **Parity evidence:**
  `tests/test_services_furnacemind.py::test_reference_questions_make_the_same_tool_calls_through_page_7_and_the_api`,
  `tests/test_services_furnacemind.py::test_service_prompt_is_page_7s_prompt_without_the_user_layers`,
  `tests/test_services_furnacemind.py::test_history_window_counts_artifact_entries_like_page_7`,
  `tests/test_api_furnacemind_page.py::test_skills_are_page_7s_built_in_ones_without_a_skill_table`,
  `tests/test_api_furnacemind_page.py::test_a_skill_turn_stores_page_7s_prompt_and_puts_its_context_in_the_system_prompt`,
  `tests/test_api_furnacemind_page.py::test_the_full_dataset_downloads_as_page_7s_csv_while_the_process_holds_it`,
  `tests/test_services_reports.py::test_the_report_data_equal_the_reports_tab`.

### Blend Mix Optimiser (`custom_pages/9_Blend_Optimizer.py`)

- **Replacement:** `/bmo` (SV-3.4–3.6, #301–#303; operator inputs in Neon SV-3.1, #298).
- **Parity evidence (G3, 5 recorded days):** `tests/test_services_bmo_case.py::test_lp_plan_equals_page9`,
  `tests/test_services_bmo.py::test_page_blend_table_is_the_service_builder`,
  `tests/test_api_bmo.py::test_the_snapshots_of_page_9_are_listed_and_read`,
  `tests/test_services_operator_inputs.py::test_the_yaml_document_equals_the_streamlit_reader`.
- **Before retirement:** `/bmo` links three sections of page 9 (`PageSection`: Intensive Optimizer (DE search),
  Snapshot replay, Sandbox). Port them or drop the links; while they exist, retiring page 9 fails the SPA type check.

### Coke Energy Model (`custom_pages/10_Coke_Energy.py`)

- **Replacement:** `/` (Live Twin, SV-1.9 #289), `/trace` (SV-1.10 #290), `/forge` (SV-1.11 #291),
  `/admin/calibration` (SV-4.6 #309).
- **Parity evidence:** `tests/test_twin_window.py::test_windows_on_completed_shifts_equal_the_pipeline`,
  `tests/test_twin_parity_script.py::test_windows_of_the_last_days_equal_the_pipeline`,
  `tests/test_services_coke_energy.py::test_streamlit_path_equals_service_path`,
  `tests/test_services_trace.py::test_shift_values_equal_page_10`,
  `tests/test_services_trace.py::test_latest_shift_inputs_and_feeds_equal_page_10`,
  `tests/test_services_trace.py::test_feeds_aged_at_a_later_request_equal_page_10_at_that_time`,
  `tests/test_services_forge.py::test_statistics_equal_page_10_for_the_same_window`,
  `tests/test_services_forge.py::test_the_record_and_the_rolling_statistics_equal_page_10`,
  `tests/test_services_forge.py::test_the_calibration_of_a_day_equals_the_shift_tables_fixed_calibration`,
  `tests/test_api_calibration.py::test_the_console_equals_live_calibration_on_the_same_window`.
- **Before retirement:** G1 on plant data (`python scripts/twin_parity.py`, 14 days of shadow use).

### Burden & Fuel Optimiser (`custom_pages/11_Burden_Fuel_Optimiser.py`)

- **Replacement:** `/optimiser` (SV-3.2 #299, SV-3.3 #300), `/strata` (SV-1.12 #292), `/admin/what-if` (SV-4.3
  #306), `/admin/burnt-vs-charged` (SV-4.4 #307), `/admin/validation` (SV-4.5 #308).
- **Parity evidence (G3, 16 Oct 2026 case report):**
  `tests/test_api_optimiser.py::test_the_recorded_result_equals_page_11`,
  `tests/test_api_optimiser.py::test_step2_reproduces_cases_1_and_2_of_the_case_report`,
  `tests/test_api_optimiser.py::test_step1_reproduces_case_4_of_the_case_report`,
  `tests/test_api_optimiser.py::test_maximum_production_reproduces_case_3_of_the_case_report`,
  `tests/test_services_optimiser.py::test_memoised_yard_and_streamlit_yard_agree`,
  `tests/test_services_strata.py::test_mod_responses_equal_model_facts_at_the_same_calibration`,
  `tests/test_services_strata.py::test_the_ring_view_equals_the_network_solve_at_the_calibration_state`,
  `tests/test_services_whatif.py::test_strata_forced_blends_reproduce_the_case_report`,
  `tests/test_services_burnt_vs_charged.py::test_window_table_equals_window_agreement`,
  `tests/test_services_burnt_vs_charged.py::test_lag_equals_lag_scan`,
  `tests/test_services_burnt_vs_charged.py::test_closure_equals_transit_closure`,
  `tests/test_api_burnt_vs_charged.py::test_values_equal_transit_on_the_same_data`.
- **Before retirement:** the validation and STRATA-Learn console (`/admin/validation`, SV-4.5 #308) and its parity
  evidence.

### Not ported: Data Explorer, V-Sense, CoPilot, Material Balance, Feedback

No Svelte replacement in programme #262; these pages stay in the Streamlit navigation and are not on the retirement
list.

## Procedure

1. Record shadow use, feedback and sign-off in the row above (PR to `docs/retirement.md`).
2. Retire: give the page's descriptor in `src/config/page_registry.py`
   `retirement=Retirement(date(YYYY, M, D), ("/route", ...))` with the IST date of the change, fill "Retired on" and
   "Deleted on" here, run `python scripts/export_legacy_pages.py` and remove the SPA's links to the page (`legacy:` in
   `frontend/src/lib/nav.ts`, `legacyHref(...)` calls: they fail `pnpm check` once the page leaves `LEGACY_PAGES`).
3. For 30 days the page is out of the navigation for every role (and out of the Welcome tiles and the feedback page
   list); administrators open it at its URL (`/legacy/<URL name>`, e.g. `/legacy/Data_Visualisation`), hidden from
   the menu, with a banner naming the replacement and the deletion date. Other roles are not served it (Streamlit
   falls back to the default page).
4. From the deletion date nobody is served the page, and `tests/test_page_registry.py` fails until the page file, its
   registry entry and the code only it uses are deleted.
