<!--
@component
Zone 7 of the Live Twin (SV-1.9): open alarms of the twin (`/api/twin/alarms`, updated by `alarm.raised` and
`alarm.cleared`): shaft-efficiency drift, STRATA − TRACE gap, stale source, failed job, implied lag. Critical first,
then newest first.
-->
<script lang="ts">
	import BellRing from '@lucide/svelte/icons/bell-ring';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { istDateTime } from '../time/format.ts';
	import { alarmLabel, type AlarmView } from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		alarms: AlarmView[];
		zone: ZoneState;
		onretry: () => void;
	}

	let { alarms, zone, onretry }: Props = $props();
</script>

<Panel
	title="Alarms"
	eyebrow="Twin"
	icon={BellRing}
	note="Open alarms; each clears itself when its condition ends"
	updatedAt={zone.at ? new Date(zone.at) : null}
	warnAfterMin={10}
	staleAfterMin={20}
	busy={zone.loading}
>
	{#if zone.error && alarms.length === 0}
		<ErrorState compact title="Alarms not loaded" message={zone.error} {onretry} />
	{:else if alarms.length === 0 && !zone.loading}
		<EmptyState
			compact
			title="No open alarm"
			message="Drift, model gap, stale sources, job failures and the implied lag are watched every hour."
		/>
	{:else}
		<ul class="alarms" aria-live="polite">
			{#each alarms as a (a.id)}
				<li>
					<StatusChip
						kind={a.severity === 'critical' ? 'critical' : 'warning'}
						label={a.severity === 'critical' ? 'Critical' : 'Warning'}
						size="sm"
					/>
					<div class="text">
						<p class="kind">
							{alarmLabel(a.kind)}{#if a.subject}<span class="muted">, {a.subject}</span>{/if}
						</p>
						<p class="message">{a.message || 'No message'}</p>
						<p class="muted">Raised {istDateTime(new Date(Date.parse(a.raised_at)))} IST</p>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</Panel>

<style>
	.alarms {
		display: grid;
		gap: var(--vo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.alarms li {
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: start;
		gap: var(--vo-space-2);
		padding-bottom: var(--vo-space-2);
		border-bottom: 1px solid var(--vo-border);
	}
	.kind {
		font-size: var(--vo-fs-body);
		font-weight: 650;
	}
	.message {
		font-size: var(--vo-fs-body);
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
		font-weight: 400;
	}
	.text {
		min-width: 0;
	}
</style>
