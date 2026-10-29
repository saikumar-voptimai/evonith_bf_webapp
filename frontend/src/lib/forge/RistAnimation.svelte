<!--
@component
Rist diagram animated over the stable shifts (SV-1.11, #291): play, pause, step and a slider over the frames; the
diagram is `RistDiagram` (shift line, FORGE's line at the fixed shaft efficiency, ideal line, points).
-->
<script lang="ts">
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';
	import Pause from '@lucide/svelte/icons/pause';
	import Play from '@lucide/svelte/icons/play';
	import RistDiagram from '../charts/RistDiagram.svelte';
	import Button from '../components/Button.svelte';
	import { frameCaption, shiftLabel, toRistShift, type ForgeRistFrame } from './forge.ts';
	import { RistPlayer } from './player.svelte.ts';

	interface Props {
		frames: ForgeRistFrame[];
		height?: number;
	}

	let { frames, height = 420 }: Props = $props();

	const uid = $props.id();
	const player = new RistPlayer();

	$effect(() => {
		player.reset(frames.length);
		return () => player.pause();
	});

	const frame = $derived(frames[Math.min(player.index, frames.length - 1)]);
	const shift = $derived(frame ? toRistShift(frame) : null);
</script>

{#if frame && shift}
	<div class="player">
		<div class="controls" role="group" aria-label="Shift of the diagram">
			<Button
				size="sm"
				icon={ChevronLeft}
				iconOnly
				label="Previous shift"
				disabled={player.index <= 0}
				onclick={() => player.step(-1)}
			/>
			<Button
				size="sm"
				variant="primary"
				icon={player.playing ? Pause : Play}
				disabled={frames.length < 2}
				aria-pressed={player.playing}
				onclick={() => player.toggle()}>{player.playing ? 'Pause' : 'Play'}</Button
			>
			<Button
				size="sm"
				icon={ChevronRight}
				iconOnly
				label="Next shift"
				disabled={player.index >= frames.length - 1}
				onclick={() => player.step(1)}
			/>
			<label class="slider" for="{uid}-slider">
				<span class="sr-only">Shift</span>
				<input
					id="{uid}-slider"
					type="range"
					min="0"
					max={frames.length - 1}
					step="1"
					value={player.index}
					aria-valuetext={shiftLabel(frame.skey)}
					oninput={(e) => player.seek(Number(e.currentTarget.value))}
				/>
			</label>
			<span class="count vo-num">{player.index + 1} of {frames.length}</span>
		</div>
		<p class="caption" aria-live={player.playing ? 'off' : 'polite'}>{frameCaption(frame)}</p>
		<RistDiagram title="Rist diagram, {shiftLabel(frame.skey)}" {shift} {height} />
	</div>
{/if}

<style>
	.player {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		min-width: 0;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.slider {
		display: flex;
		flex: 1 1 12rem;
		align-items: center;
		min-width: 10rem;
		min-height: 40px;
	}
	.slider input {
		width: 100%;
		accent-color: var(--vo-accent-strong);
	}
	.count {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
	.caption {
		margin: 0;
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
		font-variant-numeric: tabular-nums;
	}
</style>
