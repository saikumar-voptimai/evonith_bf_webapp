/**
 * Geometry of the BF-02 schematic (SV-0.6, #275), ported from `ui/furnace_status/schematic.py`: the furnace profile
 * of `config/setting_ds_dv.yml` (half-widths in metres at elevations in metres, wall side negative), drawn in a
 * 560 × 532 view at 22 px per metre around the centre line x = 280, with the base at y = 512.
 *
 * `fixtures/schematic.json` holds the Python drawing; the Vitest suite checks that this port draws the same outline,
 * lining and zone dividers to the decimal (acceptance criterion of #275).
 */

export type ProfilePoint = [x: number, y: number];

export interface Zone {
	name: string;
	lower_m: number;
	upper_m: number;
}

export const SVG_W = 560;
export const SVG_H = 532;
export const PX_PER_M = 22;
export const CX = SVG_W / 2;
export const Y0 = 512;
export const GUTTER = 168;
/** Elevation of the tuyeres (hot blast) and of the taphole, m. */
export const BLAST_M = 8;
export const TAP_M = 1;
export const LINING_M = 0.32;

/** Furnace profile of `config/setting_ds_dv.yml` (plot.geometry.geometry_points). */
export const BF2_PROFILE: readonly ProfilePoint[] = [
	[-2.8, 4.374],
	[-2.8, 6.795],
	[-3.15, 8.335],
	[-3.15, 11.29],
	[-3.65, 14.39],
	[-3.65, 15.89],
	[-2.898, 20]
];

/** Zones of `config/furnace_status.yml` (ui.zones), top to bottom. */
export const BF2_ZONES: readonly Zone[] = [
	{ name: 'Stack', lower_m: 15, upper_m: 20 },
	{ name: 'Belly', lower_m: 12.9, upper_m: 15 },
	{ name: 'Bosh', lower_m: 10.5, upper_m: 12.9 },
	{ name: 'Tuyere', lower_m: 5.5, upper_m: 10.5 },
	{ name: 'Hearth', lower_m: 0, upper_m: 5.5 }
];

export const sx = (xM: number): number => CX + xM * PX_PER_M;
export const sy = (yM: number): number => Y0 - yM * PX_PER_M;

/** Python's `f"{v:.1f}"` (round half to even on the binary value, as `toFixed` does for non-ties). */
export const fmt = (v: number): string => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);

export const top = (profile: readonly ProfilePoint[]): number => Math.max(...profile.map(([, y]) => y));

/** Inner radius of the furnace at elevation `y` (m), linear between the profile points; the base takes the first. */
export function halfWidth(profile: readonly ProfilePoint[], y: number): number {
	const first = profile[0];
	if (!first) return 0;
	const points = [
		[0, Math.abs(first[0])] as [number, number],
		...profile.map(([x, yy]) => [yy, Math.abs(x)] as [number, number])
	].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	const [y0, x0] = points[0] as [number, number];
	if (y <= y0) return x0;
	for (let i = 0; i + 1 < points.length; i += 1) {
		const [ya, xa] = points[i] as [number, number];
		const [yb, xb] = points[i + 1] as [number, number];
		if (y <= yb) return xa + (xb - xa) * (yb > ya ? (y - ya) / (yb - ya) : 0);
	}
	return (points[points.length - 1] as [number, number])[1];
}

/** Outline polygon: the wall side from the base up, then the mirrored side down (as `build_furnace_svg`). */
export function outlinePoints(profile: readonly ProfilePoint[]): [number, number][] {
	const first = profile[0];
	if (!first) return [];
	const left: ProfilePoint[] = [[first[0], 0], ...profile];
	return [
		...left.map(([x, y]) => [sx(x), sy(y)] as [number, number]),
		...[...left].reverse().map(([x, y]) => [sx(-x), sy(y)] as [number, number])
	];
}

export const pointsAttr = (points: [number, number][]): string =>
	points.map(([x, y]) => `${fmt(x)},${fmt(y)}`).join(' ');

/** Lining lines, wall side then mirrored side: 0.32 m inside the wall from 0.3 m to 0.25 m below the top. */
export function liningPolylines(profile: readonly ProfilePoint[]): string[] {
	const t = top(profile);
	const ys = [...new Set([0.3, ...profile.map(([, y]) => y), t - 0.25])]
		.sort((a, b) => a - b)
		.filter((y) => y >= 0.3 && y <= t - 0.25);
	const lining = ys.map((y) => [Math.max(halfWidth(profile, y) - LINING_M, 0.5), y] as [number, number]);
	return [-1, 1].map((side) => lining.map(([r, y]) => `${fmt(sx(side * r))},${fmt(sy(y))}`).join(' '));
}

/** Dividers at the lower edge of each zone above the base: [x1, y1, x2, y2]. */
export function zoneDividers(
	profile: readonly ProfilePoint[],
	zones: readonly Zone[]
): [number, number, number, number][] {
	return zones
		.filter((z) => z.lower_m > 0)
		.map((z) => {
			const r = halfWidth(profile, z.lower_m);
			return [
				Number(fmt(sx(-r))),
				Number(fmt(sy(z.lower_m))),
				Number(fmt(sx(r))),
				Number(fmt(sy(z.lower_m)))
			];
		});
}

/** Burden surface lines in the stack at 15.9–18.9 m, sagging 9 px at the centre line. */
export function burdenArcs(profile: readonly ProfilePoint[]): { y: number; d: string }[] {
	return [15.9, 16.9, 17.9, 18.9].map((elevation) => {
		const r = halfWidth(profile, elevation) - LINING_M - 0.1;
		const y = sy(elevation);
		return { y, d: `M${fmt(sx(-r))},${fmt(y)} Q${fmt(CX)},${fmt(y + 9)} ${fmt(sx(r))},${fmt(y)}` };
	});
}

export type Anchor = 'wall' | 'blast' | 'raceway';

export interface CalloutSpec {
	key: string;
	side: -1 | 1;
	/** Vertical position of the label in the view, px. */
	y: number;
	anchor: Anchor;
	/** Elevation of a wall anchor, m. */
	elevation_m: number;
}

/** Callout positions of the Streamlit schematic (`CALLOUTS`). */
export const CALLOUTS: readonly CalloutSpec[] = [
	{ key: 'top_pressure', side: -1, y: 96, anchor: 'wall', elevation_m: 19.4 },
	{ key: 'permeability', side: -1, y: 206, anchor: 'wall', elevation_m: 13.6 },
	{ key: 'hbt', side: -1, y: 284, anchor: 'blast', elevation_m: BLAST_M },
	{ key: 'raft', side: -1, y: 396, anchor: 'raceway', elevation_m: BLAST_M },
	{ key: 'furnace_level', side: 1, y: 96, anchor: 'wall', elevation_m: 19 },
	{ key: 'heat_load_total', side: 1, y: 206, anchor: 'wall', elevation_m: 12.5 },
	{ key: 'blast_pressure', side: 1, y: 284, anchor: 'blast', elevation_m: BLAST_M },
	{ key: 'hearth_temp_d', side: 1, y: 442, anchor: 'wall', elevation_m: 4.4 }
];

/** Point on the drawing a callout's leader line ends at. */
export function calloutAnchor(profile: readonly ProfilePoint[], c: CalloutSpec): [number, number] {
	if (c.anchor === 'wall') {
		const r = halfWidth(profile, c.elevation_m);
		return [sx(c.side * r), sy(c.elevation_m)];
	}
	const r = halfWidth(profile, BLAST_M);
	const offset = c.anchor === 'blast' ? 22 : -14;
	return [sx(c.side * r) + c.side * offset, sy(BLAST_M)];
}

/** Arrow from (x1, y1) to (x2, y2): the shaft ends 10 px before the tip, under a 10 × 10 px head. */
export function arrow(
	x1: number,
	y1: number,
	x2: number,
	y2: number
): { shaft: [number, number, number, number]; head: string } {
	const length = Math.hypot(x2 - x1, y2 - y1) || 1;
	const ux = (x2 - x1) / length;
	const uy = (y2 - y1) / length;
	const bx = x2 - ux * 10;
	const by = y2 - uy * 10;
	const px = -uy * 5;
	const py = ux * 5;
	return {
		shaft: [x1, y1, bx, by],
		head: `${fmt(bx + px)},${fmt(by + py)} ${fmt(x2)},${fmt(y2)} ${fmt(bx - px)},${fmt(by - py)}`
	};
}
