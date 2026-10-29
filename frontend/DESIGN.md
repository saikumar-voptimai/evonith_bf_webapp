# Design system of the BF-02 operator console

Scope: the SvelteKit frontend (`frontend/`). Issue SV-0.5 (#274); plan section 9. Catalogue of every component in
its states: `/dev/components`.

Reference: the V-OptimAIse product family as implemented in Optimus (split sign-in page; dark green-charcoal sidebar,
light canvas, white panels, yellow accent, Inter), adopted by the product owner on 27 Oct 2026. It replaces the first,
dark palette of SV-0.5 (commit b90dc40).

## 1. Purpose and audience

The console is read by shift operators and process engineers in the control room and the technical office, on
1920×1080 control-room screens, 1366×768 laptops and 768×1024 tablets. Its primary task is to show the present state
of BF-02 and the twin's predictions (coke rate from TRACE, FORGE and STRATA; HMT; implied shaft efficiency) within
seconds, and to run the hourly and per-shift analyses. The design follows from that task: numbers first, status
explicit, data age always visible, no decoration on the working pages.

## 2. Tokens (`src/lib/theme/tokens.css`)

Contrast ratios are WCAG 2.2 values. `contrast.test.ts` checks every text token on every surface of its side
(minimum 4.5:1), chart lines, input edges and the focus ring (minimum 3:1), and status words on their chip fills.

Navigation chrome (sidebar, navigation drawer, sign-in product panel):

| Token                                         | Value                       | Role                                             | Contrast on chrome |
| --------------------------------------------- | --------------------------- | ------------------------------------------------ | ------------------ |
| `--vo-chrome`                                 | #232823                     | Background                                       |                    |
| `--vo-chrome-hover` / `-active`               | #2C322C / #3B3E29           | Hover; current page (olive)                      |                    |
| `--vo-chrome-text` / `-muted` / `-strong`     | #C6CEBE / #A0AA93 / #F1F3ED | Items; section labels and notes; product name    | 9.3 / 6.2 / 13.4:1 |
| `--vo-chrome-accent`                          | #E9CF5B                     | Text and icon of the current page; focus ring    | 9.7:1              |
| `--vo-chrome-display` / `-lede` / `-eyebrow`  | #E8EEE4 / #B2BCA8 / #C1BF90 | Sign-in headline, introduction and section label | 12.7 / 7.6 / 8.0:1 |
| `--vo-chrome-band` / `-band-strong`           | #262E25 / #2D3627           | Bands of the lining profile on the sign-in panel | (decorative)       |
| `--vo-chrome-line` / `-scrollbar` / `-avatar` | #353B34 / #4B5448 / #384930 | Rules; scrollbar thumb; user initial             |                    |

Light surfaces and text:

| Token                                                     | Value                                                     | Role                                                                                | Contrast                   |
| --------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------- |
| `--vo-bg`                                                 | #F7F8F6                                                   | Canvas                                                                              |                            |
| `--vo-surface` / `-raised`                                | #FFFFFF / #F1F3EF                                         | Panels and inputs; hover, chips, table header band                                  |                            |
| `--vo-cream` / `-raised` / `-line` / `-strong`            | #FAF6E3 / #FFFAF0 / #ECE3BF / #D9C98E                     | Highlighted panel, its cards, its border, card border on hover                      |                            |
| `--vo-border` / `-strong`                                 | #E3E7DF / #D5DAD0                                         | Region rules; table header rule                                                     |                            |
| `--vo-border-input`                                       | #858C80                                                   | Input and control boundaries                                                        | ≥ 3.1:1 on all surfaces    |
| `--vo-text`                                               | #282D29                                                   | Values, body text                                                                   | 14.0:1 on white, ≥ 12.6    |
| `--vo-text-muted`                                         | #676E65                                                   | Labels, units, notes                                                                | 5.3:1 on white, ≥ 4.7      |
| `--vo-text-placeholder`                                   | #70776D                                                   | Example entries in empty inputs                                                     | 4.6:1 on white             |
| `--vo-accent-ink` / `--vo-focus`                          | #6B5A12 / #7A6418                                         | Links and gold text; focus ring on light surfaces                                   | ≥ 6.1 / ≥ 5.1:1            |
| `--vo-accent` / `-hover`                                  | #EFD054 / #E5C443                                         | Primary action, current shift, selected option (fill)                               | text #282D29: 9.2 / 8.2:1  |
| `--vo-accent-strong`                                      | #D4B12F                                                   | Selected outline, progress bar, elapsed shift hours                                 |                            |
| `--vo-accent-soft` / `-icon`                              | #F5EFD2 / #8A7430                                         | Icon tiles                                                                          |                            |
| `--vo-ok` / `-warning` / `-critical` / `-stale` / `-info` | #2F6B2A / #9A4A07 / #B42318 / #5E6570 / #1F5FA8           | Status text, always with icon and word, on tinted fills (`-bg`)                     | ≥ 5.1:1 on the fill        |
| `--vo-cat-1` … `-6`                                       | #2F6DB5 / #C2620A / #16876A / #A87C00 / #A33460 / #6F8F1F | Categorical chart slots, used in this order                                         | 3.7–6.5:1 on white         |
| `--vo-series-context`                                     | #737A70                                                   | Context and reference series (charged coke, identity line, Other)                   | 4.4:1 on white             |
| `--vo-chart-grid` / `-axis` / `-shift`                    | #E9ECE6 / #C4C9BF / #D8DCD3                               | Gridlines; axis line; shift boundaries                                              | (recessive)                |
| `--vo-furnace-*`                                          | body #F3F5F1 → #EDC19B; outline #5F675B; warm #C4571A     | Furnace schematic only: illustrative shading, outline, lining, burden layers, flows | outline 5.9:1, flows 4.4:1 |

Type: Inter Variable (self-hosted, `@fontsource-variable/inter`, OFL-1.1) for all text, including ω, σ and Δ;
tabular lining figures in every readout, table and input. Chemical formulas are written with Unicode subscripts
(SiO₂, Al₂O₃) and rendered by `Formula.svelte` as `<sub>` with Inter digits.

| Size              | px / weight | Use                                         |
| ----------------- | ----------- | ------------------------------------------- |
| `--vo-fs-display` | 40 / 650    | Headline readout of the Live Twin           |
| `--vo-fs-kpi`     | 30 / 650    | Readouts in tiles                           |
| `--vo-fs-heading` | 26 / 650    | Page title (h1 in the page header)          |
| `--vo-fs-title`   | 17 / 650    | Panel titles (h2)                           |
| `--vo-fs-text`    | 15 / 400    | Prose, navigation items, page subtitle      |
| `--vo-fs-body`    | 14 / 400    | Table cells, controls                       |
| `--vo-fs-label`   | 13 / 550    | Labels, units, hints, table headers         |
| `--vo-fs-eyebrow` | 11.5 / 600  | Section labels: upper case, tracked 0.12 em |

Space: 4 px base (`--vo-space-1..7` = 4, 8, 12, 16, 24, 32, 48). Radius by hierarchy: panel 10, control 7, small 5,
pill 999. Layout: sidebar 260 px (collapsed 68 px), page header 64 px minimum, gutter 32 px (24 px up to 1279 px,
16 px up to 767 px), targets 40 px (sidebar items 38 px on the desktop, 40 px in the drawer). Panels are flat with a
1 px rule; only overlays (dialogs, menus, notifications) carry a shadow.

## 3. Rules

1. Numbers are the content. Tabular figures; the unit is always shown, in the muted colour, and joined to the value
   by a no-break space (`formatQuantity`: 9.0 %, 6,250 t/d); a missing value is an em dash with "No data"; decimals
   are fixed per quantity; minus is U+2212; grouping is international (230,000 Nm³/h).
2. Colour carries a role, never decoration. The yellow fill marks the current shift, the primary action and the
   selected option; gold ink marks links; in the sidebar the current page is olive with yellow text. Status colours
   appear only on status elements and always with an icon and a word (Normal, Warning, Critical, Stale, Info, or a
   specific word such as "Above band"). Chart series take the categorical slots in fixed order, and the twin's
   quantities keep their slot on every page: TRACE blue (1), FORGE orange (2), STRATA teal (3) with its ±1σ band, HMT
   gold (4); charged coke and reference lines in the context grey, dashed (section 6).
3. Data age is visible. Panels and tiles take `updatedAt`; past 20 min the age reads "Delayed", past 75 min "Stale"
   and the readout is greyed.
4. Light canvas, white panels, one highlight. At most one cream panel per page, for the item that needs attention or
   the next step (the plan of the shift, a page in preparation). No gradients except the scroll fade of the sidebar
   list; no entrance animations. Motion answers an action (dialog entry, job spinner, progress) and is removed under
   `prefers-reduced-motion`.
5. Accessible in the control room: text contrast ≥ 4.5:1, 2 px focus ring (gold on light surfaces, yellow in the
   chrome), 38–40 px targets, WAI-ARIA patterns for tabs, dialogs and the spin button, live regions for notifications
   and job states.
6. Copy is plain and specific: sentence case; buttons name the action ("Apply plan", "Run optimiser"); errors state
   what happened and how to correct it; empty states give the next step.

Product-family conventions taken from the reference: tracked upper-case section labels (sidebar sections, eyebrows
of highlighted panels and of the sign-in page); page header "Group / Title" with a one-line subtitle; highlighted
panel with icon tile, eyebrow, title, note and link cards; footer "BF-02 Twin by V-OptimAIse"; split sign-in page.
Data panels keep their own conventions: tiles share one strip with 1 px rules (`KpiStrip`) rather than separate
shadowed cards, and readouts use Inter's tabular figures rather than a monospace face.

## 4. Shell

```
┌ sidebar 260 ──────────┬ header: Group / Page title     [C] ▮▮▮▮▮▮▯▯ 22:00–06:00 39 min left   ✓ Connected   05:20 IST ┐
│ ▣ V-OptimAIse    [⇤]  │         one-line subtitle                                                                    │
│   BF-02, Evonith Steel│ ──────────────────────────────────────────────────────────────────────────────────────────── │
│ DIGITAL TWIN          │                                                                                              │
│ ▌Live Twin (olive)    │  page: panels on the canvas, 32 px gutter                                                    │
│  TRACE, FORGE, STRATA │                                                                                              │
│ FURNACE               │                                                                                              │
│ OPTIMISERS            │                                                                                              │
│ ADMIN (administrators)│                                                                                              │
│ ───────────────────── │                                                                                              │
│  Streamlit pages ↗    │ ──────────────────────────────────────────────────────────────────────────────────────────── │
│ (A) user, role     ⌃  │  BF-02 Twin by V-OptimAIse                       Times in IST. Shifts A 06:00, B 14:00, C 22:00. │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘
```

The shift band is the strongest element of the header: the shift letter on yellow, one segment per hour of the
8-hour shift (hourly analyses run at the segment boundaries; elapsed hours in `--vo-accent-strong`), the window and
the time left. Below 1024 px the sidebar becomes a navigation drawer opened from the menu button of the header; the
subtitle and the shift window are dropped. When the sections do not fit the height (1366×768 laptops), the list
scrolls and fades at the edge that has more items; focused items scroll clear of the fade. The collapse control sits
in the product row. The Admin group is shown to administrators only; the section lists its views (`ADMIN_VIEWS`) in
its own sub-navigation (`shell/SectionNav`), one route per view (`routes/admin/<view>/`). Navigation, page titles,
subtitles and planned-page placeholders read `src/lib/nav.ts`.

Sign-in page: product panel in the chrome colour (mark, section label, two-line headline with the second line in
yellow, introduction, the BF-02 lining profile of `setting_ds_dv.yml` drawn as two bands rising from the lower right
edge, footer); form panel on the canvas (shift band and IST clock, section label, "Sign in", user name, password with
show/hide, yellow submit button, administrator note). Below 1024 px the product panel becomes a header band with the
mark and the headline.

## 5. Components (`src/lib/components/`)

| Component                                                          | Use                                                                                                                            |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `shell/AppShell`, `NavRail`, `PageHeader`, `ShiftBand`, `UserMenu` | Application frame                                                                                                              |
| `Panel`                                                            | Titled region: eyebrow, icon tile, note, data age, actions, footer; `tone="cream"` highlight; `flush` for edge-to-edge content |
| `KpiTile`, `KpiStrip`                                              | Readout with unit, ±σ, change, plant band gauge, status, data age, series swatch                                               |
| `StatusChip`, `Freshness`                                          | Status and data age                                                                                                            |
| `DataTable`                                                        | Sortable table, units in the header, sticky header, empty and loading states                                                   |
| `Tabs`                                                             | WAI-ARIA tabs, automatic activation                                                                                            |
| `Modal`, `Drawer` (`Dialog`)                                       | Native `<dialog>`: focus contained, Escape and backdrop close, focus returned                                                  |
| `Toaster` + `toasts`                                               | Notifications; critical ones stay until closed                                                                                 |
| `JobButton`                                                        | Background job: queued, running with progress and elapsed time, finished, failed                                               |
| `NumberInput`                                                      | Entry with unit and bounds; refuses, never clamps                                                                              |
| `DateRange`                                                        | IST calendar days with presets                                                                                                 |
| `EmptyState`, `ErrorState`, `Skeleton`                             | Empty, failed and loading states                                                                                               |
| `Button`, `Formula`, `PlannedPage`                                 | Actions; chemical subscripts; highlighted placeholder of a page in preparation                                                 |

## 6. Charts and schematic (`src/lib/charts/`, `src/lib/furnace/`)

Apache ECharts 6.1 with the SVG renderer, tree-shaken (`charts/echarts.ts`: line, scatter, bar and pie series; grid,
tooltip, mark line, area and point, aria) and loaded on first use, so pages without charts do not download it. The
option builders (`timeseries.ts`, `xy.ts`, `rist.ts`) are pure functions, unit-tested without a browser; the
components read the tokens once (`readTheme`) and hold no colour of their own.

Colour:

- The categorical slots `--vo-cat-1..6` are assigned in fixed order and never cycled; a seventh category folds into
  "Other (n)" in the context grey. `theme/palette.test.ts` validates them on both surfaces (OKLab distances ×100,
  colour-vision deficiency simulated after Machado, Oliveira and Fernandes 2009): OKLCH lightness 0.50–0.61 (band
  0.43–0.77), chroma ≥ 0.105 (floor 0.10), neighbouring slots ΔE ≥ 9.1 under protanopia and deuteranopia (gate 8) and
  ≥ 16.8 for normal vision (gate 15), contrast ≥ 3.5:1 on the canvas (gate 3). Any two of the first three slots
  (scatter series): ΔE ≥ 9.5 (CVD) and ≥ 16.1 (normal).
- Entity colours are aliases of the slots: `--vo-series-trace` = slot 1, `-forge` = 2 (its rolling bias ±1 SD on the
  FORGE page as a 14 % wash), `-strata` = 3 (±1σ band as a 14 % wash), `-hmt` = 4 (forecast interval 14 %),
  `-charged` = context grey. Hiding a series never repaints the others.
- Text in a chart uses the text tokens (labels muted, values in text ink); the mark beside it carries the colour.

Marks and axes:

- Lines 2 px with round joins and no symbols; dashed for charged and forecast values. Scatter dots 8 px with a 2 px
  surface ring. Bars at most 24 px thick, 4 px radius at the data end, from zero, value at the tip. Donut segments
  separated by 2 px surface gaps.
- Value axes on a nice scale (`niceScale`: steps of 1, 2, 2.5 or 5 × 10ⁿ, about five intervals, every tick with the
  decimals of the step), widened 3 % so that no line runs on the frame; the unit is the axis name above the labels.
  One value axis per chart: two quantities of different scale go into two charts.
- Time axes in IST (instants shifted by +05:30 on a UTC axis), labels HH:MM and the date at midnight. Shift starts
  (A 06:00, B 14:00, C 22:00) as hairlines with the letter above the plot; annotations as dashed lines labelled inside
  the top of the plot, on the left of the line beyond 80 % of the range.
- A plant band is shaded in `--vo-track` and named in the legend; the value axis takes in both edges with a margin of
  4 % of the range, so the band never fills the plot.
- The legend sits above the plot from two entries on (series, bands, reference lines); a series entry toggles the
  series, and the last visible series stays. Time series of up to four series carry direct end labels (short name and
  last value) when the chart is at least 520 px wide; overlapping labels are moved apart.

Interaction and access:

- Time series: crosshair listing every visible series at the instant (value, unit, band). Scatter and Rist points:
  24–26 px hit targets around 8–12 px marks. Bars and donut segments: tooltip per mark.
- Every chart has a table view (Table / Chart) with every value and the units in the header, newest row first for
  time series; the drawing carries an `aria-label` summary (time span, last values, band).
- Render budget: `setOption` to `finished` below 100 ms for the 72 h coke chart (four series, a band, shifts and an
  annotation; `data-render-ms`, 7.9 ms in the audit).

| Component                  | Use                                                                                                                                                                                                                                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TimeSeriesChart`          | Quantities over time: uncertainty bands, plant band, shifts, annotations, end labels                                                                                                                                                                                                                              |
| `ScatterChart`             | Two quantities against each other: identity line on a common scale of both axes, limit lines (Bland–Altman)                                                                                                                                                                                                       |
| `BarChart`                 | One quantity across categories; `highlight` shows the bar in question in colour and the others in grey                                                                                                                                                                                                            |
| `DonutChart`               | Part of a whole: six segments at most, total in the centre, shares in the legend                                                                                                                                                                                                                                  |
| `RistDiagram`              | Rist operating diagram of a shift: shift line, model line (ω fixed), ideal line (ω = 1), points A, D, E, W, T, P and the model's E′, P′ (shift letters right of or below their points, model letters left); whole diagram or reserve and upper zone                                                               |
| `ChartHeader`, `EChart`    | Legend and table switch; host of one ECharts instance (lazy load, resize, dispose)                                                                                                                                                                                                                                |
| `furnace/FurnaceSchematic` | BF-02 cross-section from the configured profile: zones, burden layers, flows, readings in callouts with status icon and word; a zone button highlights the zone and its related readings. Below 440 px the callouts become a list under the drawing; `overlay` draws inside the furnace (Live Twin burden column) |

Parity with the Python models: `make fixtures` writes `charts/fixtures/rist_shift.json` (shift 14 Oct A through
`utils.coke_energy.rist_diagram`) and `furnace/fixtures/schematic.json` (`ui.furnace_status.schematic`).
`charts.test.ts` and `geometry.test.ts` check that the Svelte versions mark the same points and draw the same outline,
lining and zone dividers to the decimal; `tests/test_chart_fixtures.py` fails when a committed fixture is out of date.

## 7. Verification

- `pnpm test`: token contrast, palette gates, chart option builders (scales, shifts, annotations, bands, end labels,
  identity line, bars, donut folding), Rist and schematic parity, formatting (`formatQuantity`, ranges, signs),
  parsing, sorting, toast timers, input and period validation.
- `pnpm build && pnpm audit:ui`: axe-core (WCAG 2.2 AA and best practice) on every page at the three viewports and on
  the catalogue at 390×844, horizontal overflow, keyboard and dialog behaviour, chart render budget, legend toggle and
  table view, schematic zone highlight and compact list, sign-in, role guard and sign-out; screenshots in `.audit/`.
  Needs Playwright with Chromium.
