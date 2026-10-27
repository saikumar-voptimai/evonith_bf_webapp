/**
 * Colour science for the chart palette checks (SV-0.6, #275): OKLab and OKLCH from sRGB, colour-vision-deficiency
 * simulation (Machado, Oliveira and Fernandes 2009, severity 1.0, applied in linear sRGB) and the OKLab distance
 * ΔE ×100. `categoricalReport` measures the gates of a categorical palette:
 *
 * - lightness band, OKLCH L 0.43–0.77 on light surfaces (a series reads as a colour, neither ink nor tint);
 * - chroma floor, OKLCH C ≥ 0.10 (below it a series reads as grey);
 * - CVD separation, min ΔE under protanopia and deuteranopia ≥ 8 between neighbouring slots;
 * - normal-vision separation, min ΔE ≥ 15 between neighbouring slots;
 * - contrast against the surface ≥ 3:1 (WCAG 2.2 non-text contrast).
 */
import { contrastRatio } from './contrast.ts';

type Vec3 = [number, number, number];

const MACHADO: Record<'protan' | 'deutan', [Vec3, Vec3, Vec3]> = {
	protan: [
		[0.152286, 1.052583, -0.204868],
		[0.114503, 0.786281, 0.099216],
		[-0.003882, -0.048116, 1.051998]
	],
	deutan: [
		[0.367322, 0.860646, -0.227968],
		[0.280085, 0.672501, 0.047413],
		[-0.01182, 0.04294, 0.968881]
	]
};

function linear(hex: string): Vec3 {
	const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
	if (!m) throw new Error(`Not a #rrggbb colour: ${hex}`);
	return [m[1], m[2], m[3]].map((h) => {
		const c = parseInt(h ?? '0', 16) / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	}) as Vec3;
}

function oklabFromLinear([r, g, b]: Vec3): Vec3 {
	const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
	const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
	const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
	];
}

export function oklab(hex: string): Vec3 {
	return oklabFromLinear(linear(hex));
}

/** OKLCH lightness and chroma. */
export function oklch(hex: string): { l: number; c: number } {
	const [l, a, b] = oklab(hex);
	return { l, c: Math.hypot(a, b) };
}

export type Vision = 'normal' | 'protan' | 'deutan';

function simulate(hex: string, vision: Vision): Vec3 {
	const rgb = linear(hex);
	if (vision === 'normal') return rgb;
	const clamp = (v: number) => Math.min(1, Math.max(0, v));
	return MACHADO[vision].map((row) => clamp(row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2])) as Vec3;
}

/** OKLab distance ×100 between two colours as seen with `vision`. */
export function deltaE(a: string, b: string, vision: Vision = 'normal'): number {
	const [l1, a1, b1] = oklabFromLinear(simulate(a, vision));
	const [l2, a2, b2] = oklabFromLinear(simulate(b, vision));
	return 100 * Math.hypot(l1 - l2, a1 - a2, b1 - b2);
}

export const GATES = {
	band: [0.43, 0.77] as const,
	chromaFloor: 0.1,
	cvd: 8,
	normal: 15,
	contrast: 3
};

export interface CategoricalReport {
	/** Slots outside the OKLCH lightness band. */
	offBand: string[];
	/** Slots below the chroma floor. */
	lowChroma: string[];
	/** Smallest ΔE under protanopia or deuteranopia over the checked pairs. */
	worstCvd: number;
	/** Smallest ΔE under normal vision over the checked pairs. */
	worstNormal: number;
	/** Smallest contrast ratio against the surface. */
	worstContrast: number;
}

/** Gate measurements of a categorical palette; `pairs: 'adjacent'` for lines, bars and stacks, `'all'` for scatter. */
export function categoricalReport(
	palette: readonly string[],
	surface: string,
	pairs: 'adjacent' | 'all' = 'adjacent'
): CategoricalReport {
	const list: [string, string][] = [];
	palette.forEach((a, i) => {
		if (pairs === 'adjacent') {
			const b = palette[i + 1];
			if (b) list.push([a, b]);
		} else {
			for (const b of palette.slice(i + 1)) list.push([a, b]);
		}
	});
	const min = (values: number[]) => (values.length ? Math.min(...values) : Infinity);
	return {
		offBand: palette.filter((c) => {
			const { l } = oklch(c);
			return l < GATES.band[0] || l > GATES.band[1];
		}),
		lowChroma: palette.filter((c) => oklch(c).c < GATES.chromaFloor),
		worstCvd: min(list.flatMap(([a, b]) => [deltaE(a, b, 'protan'), deltaE(a, b, 'deutan')])),
		worstNormal: min(list.map(([a, b]) => deltaE(a, b))),
		worstContrast: min(palette.map((c) => contrastRatio(c, surface)))
	};
}
