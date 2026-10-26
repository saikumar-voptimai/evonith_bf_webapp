# Design system of the BF-02 operator console

Scope: the SvelteKit frontend (`frontend/`). Issue SV-0.5 (#274); plan section 9. Catalogue of every component in
its states: `/dev/components`.

## 1. Purpose and audience

The console is read by shift operators and process engineers in the control room and the technical office, on
1920×1080 control-room screens, 1366×768 laptops and 768×1024 tablets. Its primary task is to show the present state
of BF-02 and the twin's predictions (coke rate from TRACE, FORGE and STRATA; HMT; implied shaft efficiency) within
seconds, and to run the hourly and per-shift analyses. The design follows from that task: numbers first, status
explicit, data age always visible, no decoration.

## 2. Tokens (`src/lib/theme/tokens.css`)

Palette from www.v-optimai.com (yellow on the brand's warm black). Contrast ratios are WCAG 2.2 values on the panel
surface #2C211C; `contrast.test.ts` checks every text token on all four surfaces (minimum 4.5:1).

| Token                                                                     | Value                                           | Role                                                          | On panel    |
| ------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | ----------- |
| `--vo-bg`                                                                 | #1B0F0A                                         | Canvas                                                        |             |
| `--vo-black`                                                              | #000000                                         | Chrome: navigation rail, top bar, navigation drawer           |             |
| `--vo-surface`                                                            | #2C211C                                         | Panels, table header                                          |             |
| `--vo-surface-raised`                                                     | #362B27                                         | Hover, selected rows, menus, notifications                    |             |
| `--vo-border`                                                             | #3D332F                                         | Rules between regions                                         |             |
| `--vo-border-input`                                                       | #857A73                                         | Input and control boundaries (≥ 3:1 against panel and canvas) | 3.75:1      |
| `--vo-text`                                                               | #F7F4EE                                         | Values, body text                                             | 14.3:1      |
| `--vo-text-muted`                                                         | #B9B1AA                                         | Labels, units, notes                                          | 7.4:1       |
| `--vo-yellow`                                                             | #FFEB3B                                         | Current item, focus ring, shift band, STRATA series           | 12.8:1      |
| `--vo-yellow-strong`                                                      | #FACC15                                         | Primary button fill (black text 13.7:1)                       | 10.2:1      |
| `--vo-amber` / `--vo-amber-text`                                          | #FFA000 / #FDAE0F                               | FORGE series, secondary emphasis                              | 7.7 / 8.4:1 |
| `--vo-ok` / `--vo-warning` / `--vo-critical` / `--vo-stale` / `--vo-info` | #99D966 / #FB923C / #F87171 / #9E9690 / #8FB8DE | Status, always with icon and word                             | ≥ 5.4:1     |
| `--vo-series-trace` / `-forge` / `-strata` / `-charged` / `-hmt`          | #F7F4EE / #FFA000 / #FFEB3B / #8FB8DE / #FF7043 | Chart series and tile swatches                                |             |

Type: Montserrat Variable (self-hosted, OFL-1.1) for all text, weights 400/500/600/700, tabular lining figures
(`tnum`) in every readout, table and input. Montserrat has no Greek and no subscript digits: ω, σ, Δ come from the
vendored Noto Sans Greek subset (`src/lib/theme/fonts/`); chemical formulas are written with Unicode subscripts
(SiO₂, Al₂O₃) and rendered by `Formula.svelte` as `<sub>` with Montserrat digits.

| Size              | px / weight  | Use                                 |
| ----------------- | ------------ | ----------------------------------- |
| `--vo-fs-display` | 36 / 600     | Headline readout of the Live Twin   |
| `--vo-fs-kpi`     | 28 / 600     | Readouts in tiles                   |
| `--vo-fs-heading` | 21 / 600     | Page title (h1, top bar)            |
| `--vo-fs-title`   | 16 / 600     | Panel titles (h2)                   |
| `--vo-fs-body`    | 14 / 400     | Body, table cells, controls         |
| `--vo-fs-label`   | 12 / 500–600 | Labels, units, hints, table headers |

Space: 4 px base (`--vo-space-1..7` = 4, 8, 12, 16, 24, 32, 48). Radius by hierarchy: panel 8, control 6, small 4,
pill 999. Chrome: rail 232 px (collapsed 64 px), top bar 56 px, gutter 24 px (16 px below 768 px), touch target
40 px. Only overlays (dialogs, menus, notifications) carry a shadow.

## 3. Rules

1. Numbers are the content. Tabular figures; the unit is always shown, in the muted colour; a missing value is an em
   dash with "No data"; decimals are fixed per quantity; minus is U+2212; grouping is international (230,000 Nm³/h).
2. Colour carries a role, never decoration. Yellow marks the current item (page, shift), focus, the primary action
   and the STRATA prediction; amber marks FORGE. Status colours appear only on status elements and always with an
   icon and a word (Normal, Warning, Critical, Stale, Info, or a specific word such as "Above band").
3. Data age is visible. Panels and tiles take `updatedAt`; past 20 min the age reads "Delayed", past 75 min "Stale"
   and the readout is greyed.
4. Quiet chrome, dense content. Flat surfaces, 1 px rules, no gradients, no entrance animations. Motion answers an
   action (dialog entry, job spinner, progress) and is removed under `prefers-reduced-motion`.
5. Accessible in the control room: text contrast ≥ 4.5:1 (most ≥ 7:1), 2 px yellow focus ring, 40 px targets,
   WAI-ARIA patterns for tabs, dialogs and the spin button, live regions for notifications and job states.
6. Copy is plain and specific: sentence case; buttons name the action ("Apply plan", "Run optimiser"); errors state
   what happened and how to correct it; empty states give the next step.

Generic defaults rejected in the design review: tracked all-caps labels (sentence-case group labels instead); middot
meta strings (the browser title is "Page | V-OptimAIse BF-02"); separate shadowed KPI cards (tiles share one strip
with 1 px rules, `KpiStrip`); a monospace face for data (Montserrat tabular figures); a single accent (yellow and amber
have separate roles).

## 4. Shell

```
┌ rail 232 ───┬ top bar 56: page title           [C] ▮▮▮▮▮▮▯▯ 22:00–06:00  1 h 37 min left  Connected  27 Oct 04:22 IST  user ┐
│ V-OptimAIse │                                                                                                          │
│ Digital twin│  page (12-column grid, 16 px gaps)                                                                       │
│  Live Twin  │                                                                                                          │
│  TRACE …    │                                                                                                          │
│ Furnace     │                                                                                                          │
│ Optimisers  │                                                                                                          │
│ Admin*      │                                                                                                          │
│ Streamlit ↗ │                                                                                                          │
└─────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The shift band is the one strong element of the chrome: the shift letter on yellow, the 8-hour window with one
segment per hour (hourly analyses run at the segment boundaries) and the time left. Below 1024 px the rail becomes a
navigation drawer opened from the top bar; the shift band drops the window text. The Admin group is shown to
administrators only. Navigation, page titles and planned-page placeholders read `src/lib/nav.ts`.

## 5. Components (`src/lib/components/`)

| Component                                                      | Use                                                                                  |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `shell/AppShell`, `NavRail`, `TopBar`, `ShiftBand`, `UserMenu` | Application frame                                                                    |
| `Panel`                                                        | Titled region with note, data age, actions, footer; `flush` for edge-to-edge content |
| `KpiTile`, `KpiStrip`                                          | Readout with unit, ±σ, change, plant band gauge, status, data age, series swatch     |
| `StatusChip`, `Freshness`                                      | Status and data age                                                                  |
| `DataTable`                                                    | Sortable table, units in the header, sticky header, empty and loading states         |
| `Tabs`                                                         | WAI-ARIA tabs, automatic activation                                                  |
| `Modal`, `Drawer` (`Dialog`)                                   | Native `<dialog>`: focus contained, Escape and backdrop close, focus returned        |
| `Toaster` + `toasts`                                           | Notifications; critical ones stay until closed                                       |
| `JobButton`                                                    | Background job: queued, running with progress and elapsed time, finished, failed     |
| `NumberInput`                                                  | Entry with unit and bounds; refuses, never clamps                                    |
| `DateRange`                                                    | IST calendar days with presets                                                       |
| `EmptyState`, `ErrorState`, `Skeleton`                         | Empty, failed and loading states                                                     |
| `Button`, `Formula`, `PlannedPage`                             | Actions; chemical subscripts; placeholder of a page not built yet                    |

## 6. Verification

- `pnpm test`: token contrast, formatting, parsing, sorting, toast timers, input and period validation.
- `pnpm build && pnpm audit:ui`: axe-core (WCAG 2.2 AA and best practice) on every page at the three viewports,
  horizontal overflow, keyboard and dialog behaviour; screenshots in `.audit/`. Needs Playwright with Chromium.
