<!--
@component
One section of the V-Board (SV-2.2): its parameters as rows, or as tiles for the uptake and hearth strips (with the
spread of the first four in the header), and for the heat-load section the quadrant shares and the peak quadrant.
Each reading opens its trend; readings related to the selected zone are highlighted.
-->
<script lang="ts">
	import Panel from '../components/Panel.svelte';
	import { formatNumber } from '../format/number.ts';
	import ReadingRow from './ReadingRow.svelte';
	import {
		HEAT_QUADRANT_KEYS,
		HEAT_TOTAL_KEY,
		parameterMap,
		peakQuadrants,
		quadrantShares,
		SECTION,
		sectionParameters,
		shareSummary,
		type Snapshot,
		STRIPS,
		stripSpread
	} from './status.ts';

	interface Props {
		section: string;
		snapshot: Snapshot;
		/** Catalogue keys related to the selected zone. */
		linked?: ReadonlySet<string>;
		updatedAt?: Date | null;
		warnAfterMin?: number;
		staleAfterMin?: number;
		onopen?: (key: string) => void;
	}

	let {
		section,
		snapshot,
		linked = new Set(),
		updatedAt,
		warnAfterMin,
		staleAfterMin,
		onopen
	}: Props = $props();

	const byKey = $derived(parameterMap(snapshot));
	const strip = $derived(STRIPS[section]);
	const parameters = $derived(
		strip
			? strip.flatMap((k) => {
					const p = byKey.get(k);
					return p ? [p] : [];
				})
			: sectionParameters(snapshot, section)
	);
	const heat = $derived(section === SECTION.heatLoad);
	const shares = $derived(
		heat ? quadrantShares(HEAT_QUADRANT_KEYS.map((k) => byKey.get(k)?.value ?? null)) : null
	);
	const peaks = $derived(shares ? peakQuadrants(shares) : new Set<number>());
	const spread = $derived.by(() => {
		if (!strip) return null;
		const first = byKey.get(strip[0] ?? '');
		const value = stripSpread(strip.slice(0, 4).map((k) => byKey.get(k)?.value ?? null));
		if (value === null || !first) return null;
		const unit = first.unit ? ` ${first.unit}` : '';
		return `Spread ${formatNumber(value, first.decimals)}${unit}`;
	});
	const note = $derived(
		heat
			? `Rows R6–R10 · ${shares ? shareSummary(shares) : ''}`
			: strip
				? (spread ?? 'Spread needs all four readings')
				: undefined
	);
</script>

<Panel title={section} {note} level={3} {updatedAt} {warnAfterMin} {staleAfterMin}>
	<div class={['list', strip && 'tiles']}>
		{#each parameters as p (p.key)}
			<ReadingRow
				parameter={p}
				variant={strip ? 'tile' : 'row'}
				linked={linked.has(p.key)}
				peak={heat && p.key !== HEAT_TOTAL_KEY && peaks.has(HEAT_QUADRANT_KEYS.indexOf(p.key))}
				{onopen}
			/>
		{/each}
	</div>
	{#if heat}
		<p class="fine">
			Shares use unrounded quadrant readings; Q1–Q4 are stave groups, not a confirmed physical orientation.
		</p>
	{/if}
</Panel>

<style>
	.list {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin: 0 calc(-1 * var(--vo-space-2));
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8.5rem, 1fr));
		gap: var(--vo-space-2);
		margin: 0;
	}
	.fine {
		margin-top: var(--vo-space-3);
		font-size: 0.75rem;
		line-height: 1.5;
		color: var(--vo-text-muted);
	}
</style>
