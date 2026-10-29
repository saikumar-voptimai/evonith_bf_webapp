<!--
@component
BF-02 cross-section (SV-0.6, #275), drawn from the configured furnace profile exactly as the Streamlit schematic
(`ui/furnace_status/schematic.py`): outline, lining, zones, burden layers in the stack, hot blast at the tuyeres,
top gas and tapping, and readings in callouts on both sides with leader lines to the part of the furnace they
describe. The zone buttons highlight a zone and the readings related to it. Illustrative: the shading is not a
measured temperature map and the callouts are related readings, not sensor positions. `overlay` draws inside the
furnace (view-box units of `geometry.ts`, clipped to the profile, under the outline), e.g. the Live Twin's burden
column (SV-1.9); without it the schematic is unchanged.
-->
<script lang="ts" module>
	import type { StatusKind } from '../components/status.ts';

	export interface SchematicReading {
		label: string;
		/** Formatted value, or null when there is no reading. */
		value: string | null;
		unit?: string;
		/** Shown with its icon and word under the value when not normal. */
		state?: StatusKind;
		/** Word of the state when more specific than the default, e.g. "Above band" or "No reading". */
		word?: string;
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';
	import { STATUS_ICON, STATUS_WORD } from '../components/status.ts';
	import {
		arrow,
		BF2_PROFILE,
		BF2_ZONES,
		BLAST_M,
		burdenArcs,
		CALLOUTS,
		calloutAnchor,
		CX,
		fmt,
		GUTTER,
		halfWidth,
		LINING_M,
		liningPolylines,
		outlinePoints,
		pointsAttr,
		type ProfilePoint,
		PX_PER_M,
		SVG_H,
		SVG_W,
		sx,
		sy,
		TAP_M,
		top as topOf,
		type Zone,
		zoneDividers
	} from './geometry.ts';

	interface Props {
		profile?: readonly ProfilePoint[];
		zones?: readonly Zone[];
		/** Readings by callout key (`top_pressure`, `hbt`, ...); missing keys show a dash. */
		readings?: Record<string, SchematicReading>;
		/** Callout keys related to each zone, highlighted with the zone. */
		zoneReadings?: Record<string, readonly string[]>;
		selected?: string | null;
		/** Draw the illustrative burden layers in the stack. */
		layers?: boolean;
		/** Drawing inside the furnace, clipped to the profile and drawn under the outline. */
		overlay?: Snippet;
	}

	let {
		profile = BF2_PROFILE,
		zones = BF2_ZONES,
		readings = {},
		zoneReadings = {},
		selected = $bindable(null),
		layers = true,
		overlay
	}: Props = $props();

	const uid = $props.id();
	const ALT =
		'Illustrative BF-02 cross-section drawn from the configured furnace profile: stack, belly, bosh, tuyere and ' +
		'hearth zones, hot blast entering at the tuyeres, top gas leaving the top, hot metal and slag leaving the hearth.';

	const outline = $derived(pointsAttr(outlinePoints(profile)));
	const lining = $derived(liningPolylines(profile));
	const dividers = $derived(zoneDividers(profile, zones));
	const arcs = $derived(burdenArcs(profile));
	const furnaceTop = $derived(topOf(profile));
	const belly = $derived(Math.max(...profile.map(([x]) => Math.abs(x))));

	/** Bands between consecutive burden surfaces, alternately ore and coke from the top down. */
	const bands = $derived(
		arcs.slice(0, -1).map((lower, i) => {
			const upper = arcs[i + 1] as { y: number; d: string };
			const rl = halfWidth(profile, 15.9 + i) - LINING_M - 0.1;
			const ru = halfWidth(profile, 16.9 + i) - LINING_M - 0.1;
			const d =
				`M${fmt(sx(-rl))},${fmt(lower.y)} Q${fmt(CX)},${fmt(lower.y + 9)} ${fmt(sx(rl))},${fmt(lower.y)} ` +
				`L${fmt(sx(ru))},${fmt(upper.y)} Q${fmt(CX)},${fmt(upper.y + 9)} ${fmt(sx(-ru))},${fmt(upper.y)} Z`;
			return { d, kind: (arcs.length - 2 - i) % 2 === 0 ? 'ore' : 'coke' };
		})
	);

	const blast = $derived.by(() => {
		const r = halfWidth(profile, BLAST_M);
		const y = sy(BLAST_M);
		return [-1, 1].map((side) => {
			const wall = sx(side * r);
			return {
				glow: [wall - side * 14, y] as const,
				arrow: arrow(wall + side * 70, y, wall + side * 1.5, y)
			};
		});
	});
	const tap = $derived.by(() => {
		const wall = sx(-halfWidth(profile, TAP_M));
		return arrow(wall - 1.5, sy(TAP_M), wall - 66, sy(TAP_M));
	});
	const gas = $derived(arrow(CX, sy(furnaceTop) - 3, CX, sy(furnaceTop) - 34));

	const callouts = $derived(
		CALLOUTS.map((c) => {
			const [ax, ay] = calloutAnchor(profile, c);
			const start = c.side < 0 ? GUTTER + 4 : SVG_W - GUTTER - 4;
			return { ...c, ax, ay, start, reading: readings[c.key] };
		})
	);
	/** Readings from the top of the furnace down, left before right: the order of the compact list. */
	const ordered = $derived([...callouts].sort((a, b) => a.y - b.y || a.side - b.side));
	const linked = $derived(new Set(selected ? (zoneReadings[selected] ?? []) : []));
	const selectedZone = $derived(zones.find((z) => z.name === selected));

	/**
	 * Narrowest drawing that keeps the callouts beside the furnace. Below it the view box drops the callout gutters
	 * (the furnace and its arrows stay) and the readings move into a list under the drawing.
	 */
	const CALLOUT_MIN_WIDTH = 440;
	const CROP = 60;
	let width = $state(0);
	const compact = $derived(width > 0 && width < CALLOUT_MIN_WIDTH);
	const viewX = $derived(compact ? CROP : 0);
	const viewW = $derived(compact ? SVG_W - 2 * CROP : SVG_W);

	const caption = $derived(
		(compact
			? 'Arrows: hot blast enters at the tuyeres, hot metal and slag leave at the taphole, top gas leaves at the ' +
				'throat. '
			: '') +
			`Illustrative schematic of the configured profile, ${fmt(2 * belly)} m across the belly and ` +
			`${fmt(furnaceTop)} m from the hearth bottom to the throat. Shading and burden layers are not measured; ` +
			`${compact ? 'the readings are' : 'callouts are'} related readings, not sensor positions.`
	);

	const pct = (value: number, whole: number) => `${((value / whole) * 100).toFixed(2)}%`;
	/** Horizontal position in the view box as a share of the drawing's width. */
	const pctX = (x: number) => pct(x - viewX, viewW);
	const zoneLabelY = (z: Zone) => {
		// The stack label sits below the burden layers.
		const mid =
			z.name === 'Stack' && layers ? z.lower_m + 0.45 : (z.lower_m + Math.min(z.upper_m, furnaceTop)) / 2;
		return sy(mid);
	};

	function toggle(name: string): void {
		selected = selected === name ? null : name;
	}
</script>

{#snippet readingText(r: SchematicReading | undefined, key: string)}
	{@const Icon = r?.state && r.state !== 'ok' ? STATUS_ICON[r.state] : null}
	<span class="c-label">{r?.label ?? key}</span>
	<span class="c-value">
		<span class="vo-num">{r?.value ?? '—'}</span>
		{#if r?.value && r.unit}<span class="c-unit">{r.unit}</span>{/if}
	</span>
	{#if Icon && r?.state}
		<span class="c-state"
			><Icon size={13} strokeWidth={2.25} aria-hidden="true" />{r.word ?? STATUS_WORD[r.state]}</span
		>
	{/if}
{/snippet}

<figure class={['schematic', compact && 'compact']} bind:clientWidth={width}>
	<div class="drawing">
		<svg viewBox="{viewX} 0 {viewW} {SVG_H}" role="img" aria-label={ALT}>
			<defs>
				<linearGradient id="{uid}-body" x1="0" y1="0" x2="0" y2="1">
					<stop offset="0" stop-color="var(--vo-furnace-body-top)" />
					<stop offset="0.45" stop-color="var(--vo-furnace-body-top)" />
					<stop offset="0.7" stop-color="var(--vo-furnace-body-mid)" />
					<stop offset="0.86" stop-color="var(--vo-furnace-body-low)" />
					<stop offset="1" stop-color="var(--vo-furnace-body-hot)" />
				</linearGradient>
				<radialGradient id="{uid}-glow">
					<stop offset="0" stop-color="var(--vo-furnace-glow)" stop-opacity="0.75" />
					<stop offset="0.45" stop-color="var(--vo-furnace-warm)" stop-opacity="0.28" />
					<stop offset="1" stop-color="var(--vo-furnace-warm)" stop-opacity="0" />
				</radialGradient>
				<clipPath id="{uid}-clip"><polygon points={outline} /></clipPath>
			</defs>

			<polygon points={outline} fill="url(#{uid}-body)" />
			{#if selectedZone}
				<rect
					x="0"
					y={sy(Math.min(selectedZone.upper_m, furnaceTop))}
					width={SVG_W}
					height={(Math.min(selectedZone.upper_m, furnaceTop) - selectedZone.lower_m) * PX_PER_M}
					clip-path="url(#{uid}-clip)"
					class="zone-fill"
				/>
			{/if}
			{#if layers}
				{#each bands as band, i (i)}
					<path d={band.d} class={['layer', band.kind]} />
				{/each}
				{#each arcs as arc, i (i)}
					<path d={arc.d} class="arc" />
				{/each}
			{/if}
			{#if overlay}
				<g clip-path="url(#{uid}-clip)">{@render overlay()}</g>
			{/if}
			<polygon points={outline} class="outline" />
			{#each lining as points, i (i)}
				<polyline {points} class="lining" />
			{/each}
			{#each dividers as [x1, y1, x2, y2], i (i)}
				<line {x1} {y1} {x2} {y2} class="divider" />
			{/each}
			{#each zones as zone (zone.name)}
				<text x={CX} y={zoneLabelY(zone)} class={['zone-label', zone.name === selected && 'on']}
					>{zone.name}</text
				>
			{/each}

			{#each blast as b, i (i)}
				<ellipse cx={b.glow[0]} cy={b.glow[1]} rx="20" ry="12" fill="url(#{uid}-glow)" />
				<line
					x1={b.arrow.shaft[0]}
					y1={b.arrow.shaft[1]}
					x2={b.arrow.shaft[2]}
					y2={b.arrow.shaft[3]}
					class="flow warm"
				/>
				<polygon points={b.arrow.head} class="head warm" />
			{/each}
			<line x1={tap.shaft[0]} y1={tap.shaft[1]} x2={tap.shaft[2]} y2={tap.shaft[3]} class="flow warm thin" />
			<polygon points={tap.head} class="head warm" />
			<line x1={gas.shaft[0]} y1={gas.shaft[1]} x2={gas.shaft[2]} y2={gas.shaft[3]} class="flow gas thin" />
			<polygon points={gas.head} class="head gas" />

			{#if !compact}
				{#each callouts as c (c.key)}
					<line x1={c.start} y1={c.y} x2={c.ax} y2={c.ay} class={['leader', linked.has(c.key) && 'on']} />
					<circle cx={c.ax} cy={c.ay} r="2.6" class={['anchor', linked.has(c.key) && 'on']} />
				{/each}
			{/if}
		</svg>

		<span class="annot centre" style:left={pctX(CX)} style:top={pct(sy(furnaceTop) - 54, SVG_H)}>Top gas</span
		>
		{#if !compact}
			<span
				class="annot end"
				style:right={pct(SVG_W - (sx(-halfWidth(profile, BLAST_M)) - 74), SVG_W)}
				style:top={pct(sy(BLAST_M), SVG_H)}>Hot blast</span
			>
			<span
				class="annot end"
				style:right={pct(SVG_W - (sx(-halfWidth(profile, TAP_M)) - 72), SVG_W)}
				style:top={pct(sy(TAP_M), SVG_H)}>Hot metal and slag</span
			>

			{#each callouts as c (c.key)}
				<div
					class={[
						'callout',
						c.side < 0 ? 'left' : 'right',
						linked.has(c.key) && 'on',
						c.reading?.state && `s-${c.reading.state}`
					]}
					style:top={pct(c.y, SVG_H)}
					style:max-width={pct(GUTTER - 6, SVG_W)}
					style:right={c.side < 0 ? pct(SVG_W - GUTTER, SVG_W) : undefined}
					style:left={c.side > 0 ? pct(SVG_W - GUTTER, SVG_W) : undefined}
				>
					{@render readingText(c.reading, c.key)}
				</div>
			{/each}
		{/if}
	</div>

	{#if compact}
		<ul class="readings" aria-label="Readings">
			{#each ordered as c (c.key)}
				<li class={['reading', linked.has(c.key) && 'on', c.reading?.state && `s-${c.reading.state}`]}>
					{@render readingText(c.reading, c.key)}
				</li>
			{/each}
		</ul>
	{/if}

	<div class="zones" role="group" aria-label="Highlight a zone and its related readings">
		{#each zones as zone (zone.name)}
			<button type="button" aria-pressed={zone.name === selected} onclick={() => toggle(zone.name)}
				>{zone.name}</button
			>
		{/each}
	</div>
	<figcaption>{caption}</figcaption>
</figure>

<style>
	.schematic {
		margin: 0;
		min-width: 0;
	}
	.drawing {
		position: relative;
		width: 100%;
		max-width: 620px;
		margin: 0 auto;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.outline {
		fill: none;
		stroke: var(--vo-furnace-outline);
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.lining {
		fill: none;
		stroke: var(--vo-furnace-lining);
		stroke-width: 1;
		stroke-linejoin: round;
	}
	.divider {
		stroke: var(--vo-furnace-divider);
		stroke-width: 1;
		stroke-dasharray: 3 4;
	}
	.zone-fill {
		fill: var(--vo-accent-soft);
	}
	.layer.ore {
		fill: var(--vo-furnace-ore);
		fill-opacity: 0.2;
	}
	.layer.coke {
		fill: var(--vo-furnace-coke);
		fill-opacity: 0.14;
	}
	.arc {
		fill: none;
		stroke: var(--vo-text-muted);
		stroke-opacity: 0.35;
		stroke-width: 1;
	}
	.zone-label {
		fill: var(--vo-text-muted);
		font-size: 12px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-anchor: middle;
		dominant-baseline: middle;
	}
	.zone-label.on {
		fill: var(--vo-text);
	}
	.flow {
		stroke-width: 2.4;
		stroke-linecap: round;
	}
	.flow.thin {
		stroke-width: 2;
	}
	.warm {
		stroke: var(--vo-furnace-warm);
		fill: var(--vo-furnace-warm);
	}
	.gas {
		stroke: var(--vo-text-muted);
		fill: var(--vo-text-muted);
	}
	.head {
		stroke: none;
	}
	.leader {
		stroke: var(--vo-chart-axis);
		stroke-width: 1;
	}
	.leader.on {
		stroke: var(--vo-accent-strong);
		stroke-width: 1.5;
	}
	.anchor {
		fill: var(--vo-text-muted);
	}
	.anchor.on {
		fill: var(--vo-accent-ink);
	}

	.annot {
		position: absolute;
		transform: translateY(-50%);
		font-size: 12px;
		font-weight: 600;
		color: var(--vo-text-muted);
		white-space: nowrap;
		pointer-events: none;
	}
	.annot.centre {
		transform: translate(-50%, -50%);
	}
	.callout {
		position: absolute;
		transform: translateY(-50%);
		display: flex;
		flex-direction: column;
		gap: 1px;
		padding: 4px 8px;
		border: 1px solid transparent;
		border-radius: var(--vo-radius-small);
		background: rgb(255 255 255 / 0.86);
	}
	.callout.left {
		align-items: flex-end;
		text-align: right;
	}
	.callout.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-cream-raised);
	}
	.c-label {
		font-size: 12px;
		line-height: 1.3;
		color: var(--vo-text-muted);
	}
	.c-value {
		display: inline-flex;
		align-items: baseline;
		gap: 4px;
		font-size: var(--vo-fs-text);
		font-weight: 600;
		line-height: 1.25;
	}
	.c-state {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
		font-weight: 600;
		line-height: 1.3;
	}
	.s-warning .c-state {
		color: var(--vo-warning);
	}
	.s-critical .c-state {
		color: var(--vo-critical);
	}
	.s-stale .c-state {
		color: var(--vo-stale);
	}
	.s-info .c-state {
		color: var(--vo-info);
	}
	.s-stale .c-value {
		color: var(--vo-text-muted);
	}
	.c-unit {
		font-size: 12px;
		font-weight: 400;
		color: var(--vo-text-muted);
	}

	.readings {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--vo-space-1);
		margin: var(--vo-space-2) 0 0;
		padding: 0;
		list-style: none;
	}
	.reading {
		display: flex;
		flex-direction: column;
		gap: 1px;
		padding: 6px 8px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
	}
	.reading.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-cream-raised);
	}
	.zones {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: var(--vo-space-1);
		margin-top: var(--vo-space-3);
	}
	.zones button {
		min-height: 30px;
		padding: 0 12px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-label);
	}
	.zones button:hover {
		color: var(--vo-text);
		border-color: var(--vo-border-input);
	}
	.zones button[aria-pressed='true'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
		color: var(--vo-text);
	}
	figcaption {
		margin-top: var(--vo-space-2);
		font-size: 12px;
		line-height: 1.5;
		color: var(--vo-text-muted);
		text-align: center;
		text-wrap: pretty;
	}
</style>
