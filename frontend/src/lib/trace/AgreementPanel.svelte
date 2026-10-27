<!--
@component
Burnt (calibrated) against charged coke on the TRACE page (SV-1.10, #290): Lin's CCC, the Bland–Altman bias and
limits of agreement per shift or per week, and the Bland–Altman scatter of the pairs. Every number comes from
`GET /api/trace/agreement` (`transit.summary` over the view's shifts); the panel only lays them out.
-->
<script lang="ts">
	import Scale from '@lucide/svelte/icons/scale';
	import ScatterChart from '../charts/ScatterChart.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import Tabs from '../components/Tabs.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber, isNumber } from '../format/number.ts';
	import {
		agreementLines,
		BASES,
		basisPoints,
		basisStats,
		blandAltman,
		cccNote,
		cccState,
		closureText,
		KG,
		limitsText,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		type AgreementStats,
		type Basis,
		type TraceAgreement
	} from './trace.ts';

	interface Props {
		agreement: TraceAgreement | null;
		loading: boolean;
		error: unknown;
		/** The view has not been built: nothing to compare yet. */
		missing: boolean;
		onretry: () => void;
	}

	let { agreement, loading, error, missing, onretry }: Props = $props();

	let tab = $state<string>('shift');
	const basis = $derived<Basis>(tab === 'week' ? 'week' : 'shift');
	const stats = $derived(basisStats(agreement, basis));
	const level = $derived(cccState(stats?.ccc));
	const points = $derived(basisPoints(agreement, basis));
	const updatedAt = $derived(agreement?.built_at ? new Date(agreement.built_at) : null);
	const closure = $derived(closureText(agreement?.closure ?? null));

	const columns: Column<AgreementStats>[] = [
		{ key: 'window', label: 'Window', sortable: false },
		{ key: 'n', label: 'n', numeric: true, digits: 0, sortable: false },
		{ key: 'ccc', label: 'CCC', numeric: true, digits: 3, sortable: false },
		{ key: 'bias', label: 'Bias', unit: KG, numeric: true, digits: 1, sortable: false },
		{
			key: 'limits',
			label: 'Limits',
			unit: KG,
			numeric: true,
			sortable: false,
			format: (r) => limitsText(r, false)
		}
	];
</script>

<Panel
	title="Burnt against charged coke"
	icon={Scale}
	note="Lin’s concordance (CCC) and Bland–Altman limits of TRACE against the coke charged, over every balanced shift."
	updatedAt={agreement ? updatedAt : undefined}
	warnAfterMin={VIEW_WARN_MIN}
	staleAfterMin={VIEW_STALE_MIN}
	busy={loading}
	flush
>
	{#if missing}
		<div class="pad">
			<EmptyState
				title="Nothing to compare yet"
				message="The comparison is built with the TRACE view, every hour at :20."
				compact
			/>
		</div>
	{:else if error && !agreement}
		<div class="pad">
			<ErrorState {error} title="Agreement not loaded" {onretry} retrying={loading} compact />
		</div>
	{:else}
		<div class="tabs">
			<Tabs label="Averaging window" tabs={BASES.map((b) => ({ id: b.id, label: b.label }))} bind:value={tab}>
				{#snippet panel()}
					<KpiStrip min="11rem">
						<KpiTile
							label="CCC"
							value={stats?.ccc}
							digits={3}
							{loading}
							status={level?.kind}
							statusLabel={level?.label}
							note={cccNote(stats, basis)}
						/>
						<KpiTile
							label="Bias"
							value={stats?.bias}
							unit={KG}
							digits={1}
							{loading}
							note="Mean of burnt − charged"
						/>
						<KpiTile
							label="Lower limit"
							value={stats?.loa_lo}
							unit={KG}
							digits={1}
							{loading}
							note="Bias − 1.96 SD"
						/>
						<KpiTile
							label="Upper limit"
							value={stats?.loa_hi}
							unit={KG}
							digits={1}
							{loading}
							note="Bias + 1.96 SD"
						/>
					</KpiStrip>
					<div class="pad chart">
						{#if !loading && points.length === 0}
							<EmptyState
								title="No pairs to plot"
								message="A window counts once it has both burnt and charged coke; per week needs three weeks of shifts."
								compact
							/>
						{:else}
							<ScatterChart
								title="Bland–Altman plot of burnt against charged coke, {basis === 'shift'
									? 'per shift'
									: 'per week'}"
								series={blandAltman(points, basis)}
								x={{ label: 'Mean of burnt and charged', unit: KG, digits: 0 }}
								y={{ label: 'Burnt − charged', unit: KG, digits: 0 }}
								references={agreementLines(stats)}
								height={280}
								{loading}
							/>
						{/if}
					</div>
				{/snippet}
			</Tabs>
		</div>
	{/if}

	{#snippet footer()}
		{#if agreement && !missing}
			<div class="foot">
				<DataTable
					caption="Agreement per averaging window: n windows, limits = bias ± 1.96 SD"
					showCaption
					rows={agreement.table}
					{columns}
					rowKey={(r) => r.window}
					dense
				/>
				<p class="vo-muted">
					{#if closure}{closure}.{/if}
					{#if isNumber(agreement.peak_lag_h)}
						Burnt follows charged by
						<span class="vo-num">{formatNumber(agreement.peak_lag_h, 0)}&nbsp;h</span> at the correlation peak (coke
						in transit).
					{/if}
				</p>
			</div>
		{/if}
	{/snippet}
</Panel>

<style>
	.pad {
		padding: 0 var(--vo-space-5) var(--vo-space-4);
	}
	.tabs {
		padding-top: var(--vo-space-2);
	}
	.tabs :global([role='tablist']) {
		padding: 0 var(--vo-space-5);
	}
	.chart {
		padding-top: var(--vo-space-4);
	}
	.foot {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
		min-width: 0;
	}
	.foot p {
		font-size: var(--vo-fs-label);
	}
</style>
