/**
 * Labels are written with Unicode subscript digits (SiO₂, Al₂O₃, CO₂). Montserrat has no glyphs for them, so
 * `Formula.svelte` renders each run of subscript digits as <sub> with the regular Montserrat digits.
 */

const SUB_DIGITS = '₀₁₂₃₄₅₆₇₈₉';

export interface FormulaPart {
	text: string;
	sub: boolean;
}

export function formulaParts(label: string): FormulaPart[] {
	const parts: FormulaPart[] = [];
	for (const ch of label) {
		const index = SUB_DIGITS.indexOf(ch);
		const sub = index >= 0;
		const text = sub ? String(index) : ch;
		const last = parts[parts.length - 1];
		if (last && last.sub === sub) last.text += text;
		else parts.push({ text, sub });
	}
	return parts;
}
