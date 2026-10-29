/**
 * Burden column of the Live Twin (SV-1.9, #289) drawn into the furnace schematic: the charges between the stockline
 * and the raceways, in the three zones of `utils.coke_energy.burden_column` (above, in and below the cohesive zone),
 * each charge as an ore layer over a coke layer. Charges of the latest blend change (newer than it) take the "new
 * blend" colour. Illustrative positions: the zone boundaries are the profile's belly and bosh points, not a measured
 * cohesive zone; the charge counts are the ledger's.
 */
import {
	BF2_PROFILE,
	BLAST_M,
	fmt,
	halfWidth,
	LINING_M,
	type ProfilePoint,
	sx,
	sy
} from '../furnace/geometry.ts';
import type { Transit } from './live-twin.ts';

/** Elevations of the zone boundaries, m: stockline, cohesive-zone top (belly) and bottom (bosh), raceways. */
export const COLUMN_M = { stockline: 19.0, czTop: 14.39, czBottom: 11.29, raceway: BLAST_M } as const;

export const ZONE_SPAN_M: Record<string, [upper: number, lower: number]> = {
	above_cz: [COLUMN_M.stockline, COLUMN_M.czTop],
	in_cz: [COLUMN_M.czTop, COLUMN_M.czBottom],
	below_cz: [COLUMN_M.czBottom, COLUMN_M.raceway]
};

/** Share of a charge's height taken by its coke layer (reference volume fraction 0.387). */
export const COKE_SHARE = 0.387;

export interface ColumnLayer {
	d: string;
	part: 'ore' | 'coke';
	/** Charge of the blend after the latest change. */
	fresh: boolean;
	zone: string;
}

function band(profile: readonly ProfilePoint[], upper: number, lower: number): string {
	const ru = halfWidth(profile, upper) - LINING_M;
	const rl = halfWidth(profile, lower) - LINING_M;
	return (
		`M${fmt(sx(-ru))},${fmt(sy(upper))} L${fmt(sx(ru))},${fmt(sy(upper))} ` +
		`L${fmt(sx(rl))},${fmt(sy(lower))} L${fmt(sx(-rl))},${fmt(sy(lower))} Z`
	);
}

/**
 * Layers of the column: per zone of `transit.bands` its charges (rounded, at least one) spread evenly over the zone's
 * height, newest at the top; a charge counts as fresh while fewer than `blend_change.charges_since` charges lie above
 * it (every charge once the change has reached the raceways).
 */
export function columnLayers(
	transit: Transit | null | undefined,
	profile: readonly ProfilePoint[] = BF2_PROFILE
): ColumnLayer[] {
	const change = transit?.blend_change ?? null;
	const since = change ? (change.zone === 'burnt' ? Infinity : change.charges_since) : 0;
	const out: ColumnLayer[] = [];
	let above = 0;
	for (const b of transit?.bands ?? []) {
		const span = ZONE_SPAN_M[b.zone];
		if (!span) continue;
		const [upper, lower] = span;
		const n = Math.max(1, Math.round(b.charges));
		const h = (upper - lower) / n;
		for (let j = 0; j < n; j += 1) {
			const top = upper - j * h;
			const split = top - (1 - COKE_SHARE) * h;
			// The charge's position in the ledger: charges above it, scaled from the drawn count to the ledger count.
			const fresh = above + ((j + 0.5) / n) * b.charges < since;
			out.push({ d: band(profile, top, split), part: 'ore', fresh, zone: b.zone });
			out.push({ d: band(profile, split, top - h), part: 'coke', fresh, zone: b.zone });
		}
		above += b.charges;
	}
	return out;
}

/** Elevation of the first charge of the latest blend change in the column, m; null when it is not in the column. */
export function changeElevation(transit: Transit | null | undefined): number | null {
	const change = transit?.blend_change;
	if (!change || change.zone === 'burnt') return null;
	let above = 0;
	for (const b of transit?.bands ?? []) {
		const span = ZONE_SPAN_M[b.zone];
		if (!span) continue;
		if (change.charges_since <= above + b.charges && b.charges > 0) {
			const [upper, lower] = span;
			return upper - ((change.charges_since - above) / b.charges) * (upper - lower);
		}
		above += b.charges;
	}
	return null;
}

/** Horizontal line across the furnace at elevation `y` (m), as `[x1, y1, x2, y2]` in view-box units. */
export function levelLine(
	y: number,
	profile: readonly ProfilePoint[] = BF2_PROFILE
): [number, number, number, number] {
	const r = halfWidth(profile, y) - LINING_M;
	return [sx(-r), sy(y), sx(r), sy(y)];
}
