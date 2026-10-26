<!--
@component
Starts a background job (optimiser run, re-calibration, report) and shows its state: queued, running with progress
and elapsed time, finished or failed. The parent owns the job and passes its state; the job queue (SV-0.8, #277)
publishes the same states on the event stream.
-->
<script lang="ts" module>
	export type JobPhase = 'idle' | 'queued' | 'running' | 'done' | 'failed';
</script>

<script lang="ts">
	import CircleCheck from '@lucide/svelte/icons/circle-check';
	import LoaderCircle from '@lucide/svelte/icons/loader-circle';
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
	import Play from '@lucide/svelte/icons/play';
	import Timer from '@lucide/svelte/icons/timer';
	import { formatDuration } from '../time/format.ts';
	import Button from './Button.svelte';

	interface Props {
		label: string;
		phase?: JobPhase;
		/** Fraction done, 0–1, when the job reports it. */
		progress?: number | null;
		startedAt?: Date | null;
		finishedAt?: Date | null;
		/** Typical duration in seconds, shown before a run. */
		expectedS?: number;
		/** Current stage while running, or the reason after a failure. */
		message?: string;
		onrun: () => void;
		oncancel?: () => void;
		disabled?: boolean;
	}

	let {
		label,
		phase = 'idle',
		progress = null,
		startedAt = null,
		finishedAt = null,
		expectedS,
		message,
		onrun,
		oncancel,
		disabled = false
	}: Props = $props();

	const uid = $props.id();
	const busy = $derived(phase === 'queued' || phase === 'running');
	const pct = $derived(progress === null ? null : Math.round(Math.min(1, Math.max(0, progress)) * 100));

	let now = $state(Date.now());
	$effect(() => {
		if (!busy) return;
		now = Date.now();
		const timer = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(timer);
	});

	const elapsed = $derived.by(() => {
		if (!startedAt) return null;
		const end = finishedAt ? finishedAt.getTime() : busy ? now : null;
		return end === null ? null : formatDuration((end - startedAt.getTime()) / 1000, 'seconds');
	});

	/** Announced on state changes only; the ticking elapsed time stays outside the live region. */
	const status = $derived.by(() => {
		switch (phase) {
			case 'queued':
				return 'Queued, waiting for a worker';
			case 'running':
				return message ?? 'Running';
			case 'done':
				return elapsed ? `Finished in ${elapsed}` : 'Finished';
			case 'failed':
				return message ?? 'The run failed';
			default:
				return expectedS ? `Takes about ${formatDuration(expectedS, 'seconds')}` : '';
		}
	});

	function onclick(): void {
		if (busy || disabled) return;
		onrun();
	}
</script>

<div class="job">
	<button
		type="button"
		class={['run', busy && 'busy']}
		aria-disabled={busy || disabled || undefined}
		aria-describedby="{uid}-status"
		{onclick}
	>
		{#if busy}
			<span class="spin"><LoaderCircle size={17} strokeWidth={2.25} /></span>
			<span>{phase === 'queued' ? 'Queued' : pct === null ? 'Running' : `Running ${pct} %`}</span>
		{:else}
			<Play size={16} strokeWidth={2.25} />
			<span>{label}</span>
		{/if}
		{#if phase === 'running' && pct !== null}
			<span class="bar" aria-hidden="true"><span style:width="{pct}%"></span></span>
		{/if}
	</button>

	{#if busy && oncancel}
		<Button variant="ghost" size="sm" onclick={oncancel}>Cancel run</Button>
	{/if}

	<p class={['status', phase]}>
		{#if phase === 'done'}<CircleCheck size={14} strokeWidth={2.25} />{/if}
		{#if phase === 'failed'}<OctagonAlert size={14} strokeWidth={2.25} />{/if}
		<span id="{uid}-status" aria-live="polite">{status}</span>
		{#if busy && elapsed}
			<span class="elapsed" title="Elapsed time"><Timer size={13} strokeWidth={2} />{elapsed}</span>
		{/if}
	</p>
</div>

<style>
	.job {
		display: inline-flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-3);
	}
	.run {
		position: relative;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: var(--vo-target);
		min-width: 11rem;
		padding: 0 18px;
		overflow: hidden;
		border: 1px solid transparent;
		border-radius: var(--vo-radius-control);
		background: var(--vo-yellow-strong);
		color: var(--vo-text-on-accent);
		font-size: var(--vo-fs-body);
		font-weight: 600;
		white-space: nowrap;
	}
	.run:hover:not(.busy, [aria-disabled='true']) {
		background: var(--vo-yellow);
	}
	.run.busy {
		cursor: progress;
		background: var(--vo-surface-raised);
		border-color: var(--vo-yellow);
		color: var(--vo-text);
		font-variant-numeric: tabular-nums;
	}
	.run[aria-disabled='true']:not(.busy) {
		cursor: not-allowed;
		opacity: 0.5;
	}
	.bar {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		height: 3px;
		background: var(--vo-track);
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--vo-yellow);
		transition: width var(--vo-duration) linear;
	}
	.spin {
		display: inline-flex;
		color: var(--vo-yellow);
	}
	@media (prefers-reduced-motion: no-preference) {
		.spin {
			animation: spin 1s linear infinite;
		}
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.status {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.status.done {
		color: var(--vo-ok);
	}
	.status.failed {
		color: var(--vo-critical);
	}
	.elapsed {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-left: 4px;
		font-variant-numeric: tabular-nums;
		color: var(--vo-text);
	}
</style>
