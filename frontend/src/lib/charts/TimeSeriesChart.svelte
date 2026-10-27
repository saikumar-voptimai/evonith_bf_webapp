<!--
@component
Quantities over time on an IST axis (SV-0.6, #275): lines with an optional uncertainty band, a plant band, shift
boundaries and annotations. The legend toggles series; the table view lists every value. Colours come from the
series tokens (`series-trace`, `cat-2`, ...).
-->
<script lang="ts">
	import { formatNumber } from '../format/number.ts';
	import DataTable from '../components/DataTable.svelte';
	import type { Column } from '../components/table.ts';
	import ChartHeader, { type LegendItem } from './ChartHeader.svelte';
	import EChart from './EChart.svelte';
	import { readTheme, type ChartTheme } from './theme.ts';
	import {
		type PlantBand,
		type TimeAnnotation,
		type TimeSeries,
		timeSeriesOption,
		timeSeriesSummary
	} from './timeseries.ts';
	import { axisTimeTitle, toAxis } from './time.ts';

	interface Props {
		/** Accessible name of the chart and caption of its table, e.g. "Coke rate by model, last 72 h". */
		title: string;
		series: TimeSeries[];
		unit: string;
		digits?: number;
		height?: number;
		shifts?: boolean;
		annotations?: TimeAnnotation[];
		band?: PlantBand;
		yMin?: number;
		yMax?: number;
		/** Direct labels at the end of each line; left out below `END_LABEL_MIN_WIDTH`, where the legend suffices. */
		endLabels?: boolean;
		loading?: boolean;
		onrender?: (ms: number) => void;
	}

	let {
		title,
		series,
		unit,
		digits = 1,
		height = 280,
		shifts = true,
		annotations = [],
		band,
		yMin,
		yMax,
		endLabels = false,
		loading = false,
		onrender
	}: Props = $props();

	/** Narrowest chart that keeps direct end labels: below it they would take a third of the plot. */
	const END_LABEL_MIN_WIDTH = 520;

	let theme = $state.raw<ChartTheme | null>(null);
	let hidden = $state<string[]>([]);
	let view = $state<'chart' | 'table'>('chart');
	let width = $state(0);

	$effect(() => {
		theme = readTheme();
	});

	const option = $derived(
		theme
			? timeSeriesOption(theme, {
					series,
					unit,
					digits,
					shifts,
					annotations,
					band,
					yMin,
					yMax,
					endLabels: endLabels && width >= END_LABEL_MIN_WIDTH,
					hidden: new Set(hidden)
				})
			: null
	);

	const legend = $derived<LegendItem[]>(
		theme
			? [
					...series.flatMap((s) => {
						const colour = theme!.token(s.colour);
						const items: LegendItem[] = [
							{ id: s.id, label: s.label, colour, mark: s.dashed ? 'dashed' : 'line' }
						];
						if (s.band)
							items.push({
								id: `${s.id}-band`,
								label: `${s.label}, ${s.band.label}`,
								colour: theme!.token(s.band.fill),
								mark: 'band',
								fixed: true
							});
						return items;
					}),
					...(band
						? [
								{
									id: '__plant_band__',
									label: band.label,
									colour: theme.token('track'),
									mark: 'band' as const,
									fixed: true,
									outlined: true
								}
							]
						: [])
				]
			: []
	);

	interface Row {
		t: number;
		values: Record<string, number | null>;
	}

	const rows = $derived.by<Row[]>(() => {
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- local to this derivation, not state
		const byTime = new Map<number, Row>();
		const row = (t: number): Row => {
			let r = byTime.get(t);
			if (!r) {
				r = { t, values: {} };
				byTime.set(t, r);
			}
			return r;
		};
		for (const s of series) {
			for (const [t, v] of s.points) row(t).values[s.id] = v;
			for (const [t, lo, hi] of s.band?.points ?? []) {
				row(t).values[`${s.id}:lo`] = lo;
				row(t).values[`${s.id}:hi`] = hi;
			}
		}
		return [...byTime.values()].sort((a, b) => b.t - a.t);
	});

	const columns = $derived<Column<Row>[]>([
		{
			key: 't',
			label: 'Time',
			unit: 'IST',
			value: (r) => r.t,
			format: (r) => axisTimeTitle(toAxis(r.t)).replace(' IST', '')
		},
		...series.flatMap((s): Column<Row>[] => {
			const value: Column<Row> = {
				key: s.id,
				label: s.label,
				unit,
				numeric: true,
				digits,
				value: (r) => r.values[s.id] ?? null
			};
			if (!s.band) return [value];
			return [
				value,
				{
					key: `${s.id}:band`,
					label: `${s.label}, ${s.band.label}`,
					unit,
					align: 'right',
					sortable: false,
					value: (r) => r.values[`${s.id}:lo`] ?? null,
					format: (r) => {
						const lo = r.values[`${s.id}:lo`];
						const hi = r.values[`${s.id}:hi`];
						return lo == null || hi == null ? '—' : `${formatNumber(lo, digits)}–${formatNumber(hi, digits)}`;
					}
				}
			];
		})
	]);

	const summary = $derived(timeSeriesSummary(title, { series, unit, digits, annotations, band, yMin, yMax }));
</script>

<div class="chart" bind:clientWidth={width}>
	<ChartHeader items={legend} bind:hidden bind:view />
	{#if view === 'table'}
		<DataTable {rows} {columns} caption="{title} (table)" rowKey={(r) => r.t} dense maxHeight="{height}px" />
	{:else if option}
		<EChart {option} {height} label={summary} {loading} {onrender} />
	{:else}
		<div style:height="{height}px"></div>
	{/if}
</div>

<style>
	.chart {
		min-width: 0;
	}
</style>
