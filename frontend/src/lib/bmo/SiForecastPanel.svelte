<!--
@component
Hot-metal Si forecast of page 9 (the worker's ledger, `forecast.si` every 5 min): the path Now, +1 h, +2 h, +3 h with
its 90 % range, the latest cast Si, and why a forecast is out of date or not usable.
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
	import { issuerText, SI_STALE_MIN, SI_WARN_MIN, siStatus, type BmoSiView, type SiPoint } from './bmo.ts';

	interface Props {
		view: BmoSiView | null;
		loading: boolean;
		error: unknown;
		onretry: () => void;
	}

	let { view, loading, error, onretry }: Props = $props();

	const f = $derived(view?.forecast ?? null);
	const status = $derived(siStatus(view));
	const issued = $derived(f ? new Date(f.issued_at) : null);
	const now = $derived(f?.horizons?.find((h) => h.horizon_minutes === 0) ?? null);
	const hhmm = (iso: string): string => istDateTime(new Date(iso)).split(' ').at(-1) ?? '';

	const columns: Column<SiPoint>[] = [
		{ key: 'label', label: 'Horizon', sortable: false, width: '6rem' },
		{
			key: 'target_start',
			label: 'Cast window, IST',
			sortable: false,
			format: (h) => `${hhmm(h.target_start)}–${hhmm(h.target_end)}`
		},
		{ key: 'si_pct', label: 'Si', unit: '%', numeric: true, digits: 2, sortable: false },
		{
			key: 'range',
			label: '90 % range',
			unit: '%',
			sortable: false,
			align: 'right',
			format: (h) => `${formatNumber(h.lower_pct, 2)}–${formatNumber(h.upper_pct, 2)}`
		}
	];
</script>

<Panel
	title="Hot-metal Si forecast"
	note="Page 9's furnace Si model, read from the worker's ledger; the optimiser's slag balance uses the Now value."
	updatedAt={issued}
	warnAfterMin={SI_WARN_MIN}
	staleAfterMin={SI_STALE_MIN}
	busy={loading && !!view}
>
	{#if error && !view}
		<ErrorState {error} title="Si forecast not loaded" {onretry} retrying={loading} compact />
	{:else if view?.mode === 'legacy'}
		<EmptyState
			compact
			title="The Si forecast is off"
			message={view.note ?? 'bmo.si_forecast.mode is legacy: page 9 runs without a forecast Si.'}
		/>
	{:else}
		{#if error}
			<ErrorState
				{error}
				title="Si forecast not refreshed"
				message="The values shown are from the last successful load."
				{onretry}
				retrying={loading}
				compact
			/>
		{/if}
		<KpiStrip min="11rem">
			<KpiTile
				label="Si, now"
				value={f?.si_pct}
				unit="%"
				digits={2}
				status={status.kind}
				statusLabel={status.label}
				note={now
					? `90 % range ${formatNumber(now.lower_pct, 2)}–${formatNumber(now.upper_pct, 2)} %`
					: undefined}
				updatedAt={issued}
				warnAfterMin={SI_WARN_MIN}
				staleAfterMin={SI_STALE_MIN}
				loading={loading && !view}
			/>
			<KpiTile
				label="Latest cast Si"
				value={f?.last_sample_si_pct}
				unit="%"
				digits={2}
				note={f?.last_sample_at
					? `Cast of ${istDateTime(new Date(f.last_sample_at))} IST`
					: 'No cast in the record'}
				loading={loading && !view}
			/>
		</KpiStrip>
		{#if f}
			<DataTable
				caption="Si path of the origin {istDateTime(new Date(f.origin_at))} IST"
				rows={f.horizons ?? []}
				{columns}
				rowKey={(h) => h.horizon_minutes}
				dense
				emptyTitle="No path"
				emptyMessage="The forecast at this origin carries no horizons."
			/>
		{:else if view && !loading}
			<EmptyState
				compact
				title="No forecast in the ledger"
				message="The worker issues one every 5 min (forecast.si); check it under Admin, Jobs and health."
			/>
		{/if}
		<ul class="notes">
			{#if view?.note}<li>{view.note}</li>{/if}
			{#each f?.reasons ?? [] as reason (reason)}<li>{reason}</li>{/each}
			{#if view}<li>
					{issuerText(view.issuer, view.worker_active)}{f?.model_version
						? ` Model ${f.model_version}${view.mode === 'shadow' ? ', shadow mode' : ''}.`
						: ''}
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
