<!--
@component
Current shift in the top bar: the letter, the 8-hour window, one segment per hour filled as the shift runs, and the
time left. Hourly analyses run at the segment boundaries; the shift report closes at the end of the band.
-->
<script lang="ts">
	import { formatDuration } from '../../time/format.ts';
	import { istClock, shiftAt } from '../../time/shift.ts';

	interface Props {
		now: Date;
		/** Hide the window text (narrow top bar). */
		compact?: boolean;
	}

	let { now, compact = false }: Props = $props();

	const SHIFT_MIN = 480;
	const shift = $derived(shiftAt(now));
	const elapsed = $derived(
		Math.min(SHIFT_MIN, Math.max(0, (now.getTime() - shift.start.getTime()) / 60_000))
	);
	const hours = $derived(
		Array.from({ length: 8 }, (_, i) => Math.min(1, Math.max(0, (elapsed - i * 60) / 60)))
	);
	const shiftWindow = $derived(`${istClock(shift.start)}–${istClock(shift.end)}`);
	const left = $derived(`${formatDuration((SHIFT_MIN - elapsed) * 60)} left`);
</script>

<div class={['band', compact && 'compact']} role="group" aria-label="Current shift">
	<span class="letter" aria-hidden="true">{shift.letter}</span>
	<div class="detail">
		<p class="line">
			<span class="name">Shift {shift.letter}</span>
			{#if !compact}<span class="window vo-num">{shiftWindow}</span>{/if}
			<span class="sr-only">, {left}</span>
		</p>
		<span class="hours" aria-hidden="true">
			{#each hours as fill, i (i)}
				<span class="hour"><span class="fill" style:width="{fill * 100}%"></span></span>
			{/each}
		</span>
	</div>
	<span class="left vo-num" aria-hidden="true">{left}</span>
</div>

<style>
	.band {
		display: flex;
		align-items: center;
		gap: 10px;
		min-width: 0;
	}
	.letter {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 30px;
		height: 30px;
		border-radius: var(--vo-radius-control);
		background: var(--vo-yellow);
		color: var(--vo-text-on-accent);
		font-size: var(--vo-fs-title);
		font-weight: 700;
		line-height: 1;
	}
	.detail {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
	.line {
		display: flex;
		align-items: baseline;
		gap: 8px;
		font-size: var(--vo-fs-label);
		line-height: 14px;
		white-space: nowrap;
	}
	.name {
		font-weight: 600;
	}
	.window {
		color: var(--vo-text-muted);
		font-weight: 500;
	}
	.hours {
		display: grid;
		grid-template-columns: repeat(8, 1fr);
		gap: 2px;
		width: 152px;
	}
	.compact .hours {
		width: 96px;
	}
	.hour {
		height: 5px;
		border-radius: 1px;
		background: var(--vo-track);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--vo-yellow);
	}
	.left {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
</style>
