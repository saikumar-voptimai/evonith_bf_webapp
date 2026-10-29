/**
 * A4 print plan of a shift report (SV-2.4, #296). The report view prints as one column of panels in this order:
 * header, KPI tiles (five per row), fuel and burden, parameters, temperatures, twin, narrative. Heights come from the
 * print theme (`tokens.css`: `--vo-print-margin`, `--vo-print-row`, `--vo-print-fs`; `print.test.ts` compares them);
 * a panel that does not fit the rest of a page starts the next one, as `break-inside: avoid` makes the browser do
 * (Panel.svelte); the parameters panel (`breakable`) may continue on the next page between its groups, its heading
 * staying with the first. The page shows the plan beside the Print button; the acceptance limit is three pages.
 */
import {
	burdenShares,
	TWIN_TILES,
	materialGroups,
	narrativeBlocks,
	parameterGroups,
	reportTiles,
	temperatureRows,
	type ReportDetail
} from './reports.ts';

export const PRINT = {
	/** A4 portrait, mm. */
	pageHeightMm: 297,
	pageWidthMm: 210,
	/** `--vo-print-margin`, every side. */
	marginMm: 12,
	/** `--vo-print-row`: one table row (DataTable `td` height in print). */
	rowMm: 5,
	/** `--vo-print-fs`: body text. */
	fontPt: 8.5,
	lineHeight: 1.35,
	/** KPI tiles per row in print (`ReportView.svelte`). */
	tilesPerRow: 5,
	/** Most pages a report may take (acceptance criterion of #296). */
	maxPages: 3
} as const;

const PT_MM = 25.4 / 72;
/** One line of body text, mm. */
export const LINE_MM = PRINT.fontPt * PT_MM * PRINT.lineHeight;
/** Printable height and width of a page, mm. */
export const BODY_HEIGHT_MM = PRINT.pageHeightMm - 2 * PRINT.marginMm;
export const BODY_WIDTH_MM = PRINT.pageWidthMm - 2 * PRINT.marginMm;
/** Characters per line of body text (mean advance of Inter about 0.55 em). */
export const CHARS_PER_LINE = Math.floor(BODY_WIDTH_MM / (0.55 * PRINT.fontPt * PT_MM));
/** Panel heading (title and note) with its rule and padding, mm. */
export const PANEL_MM = 11;
/** One KPI tile in print (label, value, status, rule), mm. */
export const TILE_MM = 22;
/** Report header: title, period and meta lines, mm (flags add a line each). */
export const HEADER_MM = 18;

export interface PrintBlock {
	name: string;
	mm: number;
}

const lines = (text: string, indent = 0): number =>
	Math.max(1, Math.ceil((text.length + indent) / CHARS_PER_LINE));
const table = (rows: number, caption = false): number => (rows + 1 + (caption ? 1 : 0)) * PRINT.rowMm;

/** Heights of the printed panels of a report, in print order. */
export function reportBlocks(
	detail: ReportDetail,
	options: { twin: boolean } = { twin: true }
): PrintBlock[] {
	const d = detail.data;
	const flags = d.status_flags?.length ?? 0;
	const tiles = reportTiles(detail).length;
	const materials = materialGroups(d.used_materials);
	const materialRows = materials.reduce((sum, g) => sum + 1 + g.items.length, 0);
	const burdenBar = burdenShares(d.burden_ratio) ? 8 : LINE_MM;
	// Fuel rates, material charged and the burden facts are one-row tables; materials one row per line.
	const fuel = 3 * table(1, true) + burdenBar + materialRows * PRINT.rowMm;
	const params = parameterGroups(d).map((g) => table(g.rows.length, true));
	const temps = table(temperatureRows(d).length) + table(1, true); // quadrants, hearth pad
	const narrative = narrativeBlocks(detail.narrative ?? detail.narrative_error ?? '').reduce(
		(sum, b) =>
			sum +
			1.5 +
			(b.kind === 'p'
				? lines(b.parts.map((p) => p.text).join('')) * LINE_MM
				: b.items.reduce((s, item) => s + lines(item.map((p) => p.text).join(''), 4) * LINE_MM, 0)),
		0
	);
	const blocks: PrintBlock[] = [
		{ name: 'header', mm: HEADER_MM + flags * LINE_MM },
		{ name: 'kpis', mm: PANEL_MM + Math.ceil(tiles / PRINT.tilesPerRow) * TILE_MM + 2 * LINE_MM },
		{ name: 'fuel-burden', mm: PANEL_MM + fuel },
		...params.map((mm, i) => ({
			name: i ? `parameters-${i + 1}` : 'parameters',
			mm: (i ? 0 : PANEL_MM) + mm
		})),
		{ name: 'temperatures', mm: PANEL_MM + temps },
		...(options.twin
			? [{ name: 'twin', mm: PANEL_MM + Math.ceil(TWIN_TILES / PRINT.tilesPerRow) * TILE_MM }]
			: []),
		{ name: 'narrative', mm: PANEL_MM + Math.max(narrative, LINE_MM) + LINE_MM }
	];
	return blocks;
}

export interface PrintPlan {
	pages: number;
	/** Index of the first block on each page after the first. */
	breaks: number[];
	/** Height used on each page, mm. */
	used: number[];
}

/** Greedy pagination: a block that does not fit the rest of the page starts the next; a block taller than a page
 * runs over as many pages as it needs. */
export function paginate(blocks: readonly PrintBlock[], pageMm: number = BODY_HEIGHT_MM): PrintPlan {
	const used: number[] = [0];
	const breaks: number[] = [];
	blocks.forEach((b, i) => {
		const current = used[used.length - 1] ?? 0;
		if (current > 0 && current + b.mm > pageMm) {
			breaks.push(i);
			used.push(0);
		}
		let rest = b.mm;
		while (rest > 0) {
			const room = pageMm - (used[used.length - 1] ?? 0);
			const take = Math.min(room, rest);
			used[used.length - 1] = (used[used.length - 1] ?? 0) + take;
			rest -= take;
			if (rest > 0) used.push(0);
		}
	});
	return { pages: used.length, breaks, used };
}

export function printPlan(detail: ReportDetail, options: { twin: boolean } = { twin: true }): PrintPlan {
	return paginate(reportBlocks(detail, options));
}
