<!--
@component
Side-by-side results of up to four what-if cases (SV-4.3, #306). FORGE: the calibration point, then each case with
its change against it (two-zone balance with ω held, and the oxygen-balance model with ηCO held). STRATA: each plan,
with its change against the first case, and the blends. Every column states its calibration and the fact-register
version it used.
-->
<script lang="ts">
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import {
		basisText,
		cell,
		FORGE_OUTPUTS,
		forgeBase,
		mixedBasis,
		STRATA_OUTPUTS,
		type StrataCaseResult,
		type WhatIfCase,
		type WhatIfKind
	} from './whatif.ts';

	interface Props {
		kind: WhatIfKind;
		/** The cases compared, in column order. */
		cases: WhatIfCase[];
	}

	let { kind, cases }: Props = $props();

	interface Row {
		key: string;
		label: string;
		values: string[];
	}

	const show = (c: { text: string; delta: string | null }): string =>
		c.delta ? `${c.text} (${c.delta})` : c.text;

	const base = $derived(kind === 'forge' ? forgeBase(cases) : null);
	const ran = $derived(cases.filter((c) => c.forge_run || c.strata_run));
	const strataRef = $derived(
		cases.find((c) => c.strata_run?.result.status === 'ok')?.strata_run?.result ?? null
	);

	const rows = $derived.by((): Row[] => {
		if (kind === 'forge') {
			return FORGE_OUTPUTS.map((o) => ({
				key: o.key,
				label: o.unit ? `${o.label}, ${o.unit}` : o.label,
				values: [
					show(cell(o, base, null)),
					...cases.map((c) => show(cell(o, c.forge_run?.result ?? null, c.forge_run?.base ?? null)))
				]
			}));
		}
		const out: Row[] = STRATA_OUTPUTS.map((o) => ({
			key: o.key,
			label: o.unit ? `${o.label}, ${o.unit}` : o.label,
			values: cases.map((c) => show(cell(o, c.strata_run?.result ?? null, strataRef)))
		}));
		const ids: Record<string, string> = {};
		for (const c of cases) for (const b of c.strata_run?.result.blend ?? []) ids[b.id] = b.name;
		for (const [id, name] of Object.entries(ids)) {
			out.push({
				key: `blend:${id}`,
				label: `${name}, % of wet ferrous`,
				values: cases.map((c) => {
					const b = (c.strata_run?.result.blend ?? []).find((x) => x.id === id);
					return c.strata_run ? formatNumber(b?.share_pct ?? 0, 1) : '—';
				})
			});
		}
		return out;
	});

	const columns = $derived.by((): Column<Row>[] => {
		const head: Column<Row>[] = [{ key: 'label', label: 'Quantity', sortable: false, width: '18rem' }];
		const names =
			kind === 'forge' ? ['Calibration point', ...cases.map((c) => c.name)] : cases.map((c) => c.name);
		return [
			...head,
			...names.map((label, i) => ({
				key: `v${i}`,
				label,
				align: 'right' as const,
				sortable: false,
				value: (r: Row) => r.values[i] ?? '—',
				format: (r: Row) => r.values[i] ?? '—'
			}))
		];
	});

	function chip(r: StrataCaseResult | { status: string }): 'ok' | 'warning' | 'critical' {
		return r.status === 'ok' ? 'ok' : r.status === 'infeasible' ? 'warning' : 'critical';
	}
</script>

{#if !ran.length}
	<EmptyState
		title="No results yet"
		message="Run the selected cases: each column then shows its result, its calibration and the fact-register version it used."
		compact
	/>
{:else}
	<ul class="legend">
		{#each cases as c (c.id)}
			{@const run = c.forge_run ?? c.strata_run}
			{@const result = c.forge_run?.result ?? c.strata_run?.result}
			<li>
				<span class="name">{c.name}</span>
				{#if run && result}
					<StatusChip
						kind={chip(result)}
						label={result.status === 'ok'
							? 'Solved'
							: result.status === 'infeasible'
								? 'Infeasible'
								: 'Failed'}
						size="sm"
					/>
					<span class="meta">{basisText(run.basis)}</span>
					<span class="meta">Fact register {run.facts_version}</span>
					<span class="meta">Run {istDateTime(new Date(run.computed_at))} IST</span>
					{#if result.error}<span class="problem">{result.error}</span>{/if}
					{#each result.warnings ?? [] as w (w)}
						<span class="warn"><StatusChip kind="warning" size="sm" /> {w}</span>
					{/each}
					{#if c.forge_run}
						{#each c.forge_run.result.notes ?? [] as n (n)}<span class="meta">{n}</span>{/each}
					{/if}
				{:else}
					<StatusChip kind="info" label="Not run" size="sm" />
					<span class="meta">No result since the case was saved or changed.</span>
				{/if}
			</li>
		{/each}
	</ul>
	{#if mixedBasis(cases)}
		<p class="mixed">
			<StatusChip kind="warning" label="Mixed basis" size="sm" />
			The columns come from runs on different calibrations or register versions: run them together to compare on
			one basis.
		</p>
	{/if}
	<DataTable
		caption={kind === 'forge' ? 'FORGE what-if cases side by side' : 'STRATA plans side by side'}
		{rows}
		{columns}
		rowKey={(r) => r.key}
		dense
	/>
	<p class="note">
		{kind === 'forge'
			? 'Changes in brackets are against the calibration point of the same model. Two-zone: ω, the zone heat losses (GJ/h) and r_H held, ηCO and top gas computed. Oxygen balance: ηCO and top gas held.'
			: 'Changes in brackets are against the first solved plan. A forced blend lifts the slag-quality and charging limits and reports them; the cheapest blend meets them.'}
	</p>
{/if}

<style>
	.legend {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr));
		gap: var(--vo-space-3);
		margin: 0 0 var(--vo-space-4);
		padding: 0;
		list-style: none;
	}
	.legend li {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
		padding: var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface-raised);
	}
	.name {
		font-weight: 650;
	}
	.meta,
	.note {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.problem {
		font-size: var(--vo-fs-label);
		color: var(--vo-critical);
	}
	.warn {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-label);
	}
	.mixed {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.note {
		margin: var(--vo-space-3) 0 0;
	}
</style>
