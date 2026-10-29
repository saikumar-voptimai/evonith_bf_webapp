<!--
@component
Dataset panel: the first rows of the dataset an answer produced (as the agent service previews it), its size and
source, and the CSV download of the full DataFrame (page 7's download) while the API holds it; afterwards the
preview rows are offered as CSV.
-->
<script lang="ts">
	import Download from '@lucide/svelte/icons/download';
	import Table from '@lucide/svelte/icons/table';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { formatQuantity } from '#lib/format/number.ts';
	import { describeError } from '#lib/api/describe.ts';
	import {
		columnDigits,
		csvName,
		datasetSize,
		indexText,
		previewCsv,
		previewRows,
		saveText,
		type ArtifactRef,
		type DatasetPart,
		type PreviewRow
	} from './furnacemind.ts';

	interface Props {
		artifact: ArtifactRef | null;
		/** The full DataFrame as CSV, or null when the API no longer holds it. */
		fullCsv: (part: DatasetPart) => Promise<string | null>;
	}

	let { artifact, fullCsv }: Props = $props();

	const part = $derived(artifact?.part.type === 'dataset' ? artifact.part : null);
	const rows = $derived(part?.preview ? previewRows(part.preview) : []);
	const columns = $derived.by((): Column<PreviewRow>[] => {
		const preview = part?.preview;
		if (!preview) return [];
		return [
			{ key: '_index', label: 'Index', format: (r) => indexText(r._index), width: '11rem' },
			...preview.columns.map((label, k) => {
				const digits = columnDigits(preview.data.map((r) => r[k]));
				return digits === null ? { key: `c${k}`, label } : { key: `c${k}`, label, numeric: true, digits };
			})
		];
	});
	const source = $derived.by(() => {
		const meta = part?.meta ?? {};
		return ['source', 'window', 'resample', 'start_time', 'end_time']
			.map((k) => meta[k])
			.filter((v) => typeof v === 'string' && v)
			.join(' · ');
	});

	let busy = $state(false);
	let note = $state<string | null>(null);

	async function save(): Promise<void> {
		if (!part) return;
		busy = true;
		note = null;
		try {
			const text = await fullCsv(part);
			if (text !== null) saveText(text, csvName(part), 'text/csv');
			else if (part.preview) {
				saveText(previewCsv(part.preview), csvName(part, '_preview'), 'text/csv');
				note = `The API no longer holds the full dataset; downloaded the first ${formatQuantity(part.preview_rows, 0, 'rows')}. Ask again for the full data.`;
			} else note = 'The full dataset is no longer held and has no preview. Ask again for the data.';
		} catch (error) {
			note = `Download failed: ${describeError(error).message}`;
		} finally {
			busy = false;
		}
	}
</script>

<Panel
	title="Dataset"
	icon={Table}
	level={3}
	note={part
		? `${part.dataset_id || 'Dataset'} · ${datasetSize(part)}`
		: 'Data an answer fetched appears here.'}
	flush={!!part?.preview}
>
	{#snippet actions()}
		{#if part}
			<Button size="sm" icon={Download} disabled={busy} onclick={save}
				>{busy ? 'Preparing…' : 'Download CSV'}</Button
			>
		{/if}
	{/snippet}
	{#if !part}
		<EmptyState
			title="No dataset yet"
			message="Ask for data, e.g. “Fetch the fuel rate of the last 7 days”, or run a quick skill."
			compact
		/>
	{:else if !part.preview}
		<EmptyState
			title="No preview"
			message="The rows of this dataset could not be shown; the CSV download still has them while the API holds them."
			compact
		/>
	{:else}
		<DataTable
			{rows}
			{columns}
			caption="First {part.preview_rows} rows of {part.dataset_id || 'the dataset'}"
			rowKey={(r) => r._key}
			dense
			maxHeight="24rem"
		/>
	{/if}
	{#snippet footer()}
		{#if part}
			<span
				>Showing the first {formatQuantity(part.preview_rows, 0, part.preview_rows === 1 ? 'row' : 'rows')} of {formatQuantity(
					part.rows,
					0,
					part.rows === 1 ? 'row' : 'rows'
				)}{source ? ` · ${source}` : ''}. Times in IST.</span
			>
			{#if note}<p class="note" role="status">{note}</p>{/if}
		{/if}
	{/snippet}
</Panel>

<style>
	.note {
		margin-top: var(--vo-space-1);
		color: var(--vo-warning);
	}
</style>
