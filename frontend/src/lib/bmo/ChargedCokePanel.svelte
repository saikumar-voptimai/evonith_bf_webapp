<!--
@component
Charged-coke forecast of page 9's Data-Driven coke level (the worker's ledger, `forecast.charged_coke` at HH:07): the
4-h block with its 90 % range and the charges it sums, the path ahead, the forecast's state (cruise, unsettled,
off cruise are shown and used; a pause state is not), and why it is out of date.
-->
<script lang="ts">
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import {
		CHARGED_STALE_MIN,
		CHARGED_WARN_MIN,
		chargedStatus,
		chargeWindow,
		issuerText,
		type BmoChargedView,
		type ChargedPoint
	} from './bmo.ts';

	interface Props {
		view: BmoChargedView | null;
		loading: boolean;
		error: unknown;
		onretry: () => void;
	}

	let { view, loading, error, onretry }: Props = $props();

	const f = $derived(view?.forecast ?? null);
	const status = $derived(chargedStatus(view));
	const issued = $derived(f ? new Date(f.issued_at) : null);

	const columns: Column<ChargedPoint>[] = [
		{ key: 'horizon', label: 'Ahead', unit: 'h', numeric: true, digits: 0, sortable: false, width: '6rem' },
		{
			key: 'at',
			label: 'Block ends, IST',
			sortable: false,
			format: (p) => istDateTime(new Date(p.at))
		},
		{
			key: 'prediction_kg_thm',
			label: 'Charged coke',
			unit: 'kg/tHM',
			numeric: true,
			digits: 1,
			sortable: false
		},
		{
			key: 'range',
			label: '90 % range',
			unit: 'kg/tHM',
			sortable: false,
			align: 'right',
			format: (p) => `${formatNumber(p.lower_kg_thm, 1)}–${formatNumber(p.upper_kg_thm, 1)}`
		}
	];
</script>

<Panel
	title="Charged-coke forecast"
	note="4-h charged coke of the latest hourly build, the Data-Driven coke level of the plan’s fuel cost."
	updatedAt={issued}
	warnAfterMin={CHARGED_WARN_MIN}
	staleAfterMin={CHARGED_STALE_MIN}
	busy={loading && !!view}
>
	{#if error && !view}
		<ErrorState {error} title="Charged-coke forecast not loaded" {onretry} retrying={loading} compact />
	{:else}
		{#if error}
			<ErrorState
				{error}
				title="Charged-coke forecast not refreshed"
				message="The values shown are from the last successful load."
				{onretry}
				retrying={loading}
				compact
			/>
		{/if}
		<KpiStrip min="11rem">
			<KpiTile
				label="Charged coke, 4-h block"
				value={f?.prediction_kg_thm}
				unit="kg/tHM"
				digits={1}
				series="charged"
				status={status.kind}
				statusLabel={status.label}
				note={f
					? `90 % range ${formatNumber(f.lower_kg_thm, 1)}–${formatNumber(f.upper_kg_thm, 1)}; charges ${chargeWindow(f)}`
					: (view?.note ?? undefined)}
				updatedAt={issued}
				warnAfterMin={CHARGED_WARN_MIN}
				staleAfterMin={CHARGED_STALE_MIN}
				loading={loading && !view}
			/>
		</KpiStrip>
		{#if f}
			<DataTable
				caption="Charged-coke path from the build of {istDateTime(new Date(f.data_row))} IST"
				rows={f.path ?? []}
				{columns}
				rowKey={(p) => p.horizon}
				dense
				emptyTitle="No path"
				emptyMessage="This forecast carries the 4-h block only."
			/>
		{:else if view && !loading}
			<EmptyState
				compact
				title="No current forecast"
				message={view.last_issued_row
					? `The ledger’s latest row is ${istDateTime(new Date(view.last_issued_row))} IST. The worker issues one at HH:07 (forecast.charged_coke).`
					: 'The worker issues one at HH:07 (forecast.charged_coke); check it under Admin, Jobs and health.'}
			/>
		{/if}
		<ul class="notes">
			{#if view?.note && f}<li>{view.note}</li>{/if}
			{#each [...(f?.reasons ?? []), ...(view?.reasons ?? [])] as reason, i (i)}<li>{reason}</li>{/each}
			{#if view}<li>
					{issuerText(view.issuer, view.worker_active)}{f?.model_version ? ` Model ${f.model_version}.` : ''}
				</li>{/if}
		</ul>
	{/if}
</Panel>

<style>
	.notes {
		margin: var(--vo-space-3) 0 0;
		padding-left: var(--vo-space-5);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
