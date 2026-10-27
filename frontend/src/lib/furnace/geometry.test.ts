import { describe, expect, it } from 'vitest';
import fixture from './fixtures/schematic.json';
import {
	BF2_PROFILE,
	BF2_ZONES,
	CALLOUTS,
	calloutAnchor,
	halfWidth,
	liningPolylines,
	outlinePoints,
	pointsAttr,
	SVG_H,
	SVG_W,
	zoneDividers
} from './geometry.ts';

describe('furnace schematic geometry against ui/furnace_status/schematic.py', () => {
	it('uses the configured profile, zones and callouts', () => {
		expect(BF2_PROFILE).toEqual(fixture.profile);
		expect(BF2_ZONES).toEqual(fixture.zones);
		expect(CALLOUTS).toEqual(fixture.callouts);
		expect([SVG_W, SVG_H]).toEqual(fixture.view_box);
	});

	it('draws the same outline polygon, point by point', () => {
		expect(pointsAttr(outlinePoints(BF2_PROFILE))).toBe(fixture.outline);
	});

	it('draws the same lining and zone dividers', () => {
		expect(liningPolylines(BF2_PROFILE)).toEqual(fixture.lining);
		expect(zoneDividers(BF2_PROFILE, BF2_ZONES)).toEqual(fixture.zone_dividers);
	});

	it('anchors the callouts at the same points', () => {
		CALLOUTS.forEach((c, i) => {
			const [x, y] = calloutAnchor(BF2_PROFILE, c);
			const [ex, ey] = fixture.callout_anchors[i] as [number, number];
			expect(x).toBeCloseTo(ex, 5);
			expect(y).toBeCloseTo(ey, 5);
		});
	});

	it('interpolates the radius between profile points and holds it at the ends', () => {
		expect(halfWidth(BF2_PROFILE, 0)).toBe(2.8);
		expect(halfWidth(BF2_PROFILE, 15)).toBe(3.65);
		expect(halfWidth(BF2_PROFILE, (8.335 + 11.29) / 2)).toBeCloseTo(3.15, 9);
		expect(halfWidth(BF2_PROFILE, 7.565)).toBeCloseTo(2.975, 9);
		expect(halfWidth(BF2_PROFILE, 25)).toBe(2.898);
	});
});
