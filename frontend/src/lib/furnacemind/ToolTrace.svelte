<!--
@component
Collapsible tool trace of an answer: each tool the agent called, its progress text, the arguments it passed and the
result returned to the model, with a status (Running, Done, Failed, No result).
-->
<script lang="ts">
	import StatusChip from '#lib/components/StatusChip.svelte';
	import { formatQuantity } from '#lib/format/number.ts';
	import { toolTrace, type Part } from './furnacemind.ts';

	interface Props {
		parts: readonly Part[];
		/** The turn is still streaming (a call without result is Running, not No result). */
		running?: boolean;
		open?: boolean;
	}

	let { parts, running = false, open = false }: Props = $props();

	const steps = $derived(toolTrace(parts, running));
	const failed = $derived(steps.filter((s) => s.status === 'critical').length);
	const summary = $derived(
		`${steps.length} tool ${steps.length === 1 ? 'call' : 'calls'}${failed ? `, ${failed} failed` : ''}`
	);
</script>

{#if steps.length}
	<details class="trace" {open}>
		<summary>
			<span class="label">Tool trace</span>
			<span class="muted">{summary}</span>
			{#if running}<StatusChip kind="info" label="Running" size="sm" />{:else if failed}<StatusChip
					kind="warning"
					label="Tool failed"
					size="sm"
				/>{/if}
		</summary>
		<ol>
			{#each steps as step, i (step.call.call_id ?? i)}
				<li>
					<div class="head">
						<code class="name">{step.call.name}</code>
						<StatusChip kind={step.status} label={step.word} size="sm" />
						{#if step.call.label}<span class="muted">{step.call.label}</span>{/if}
					</div>
					{#if Object.keys(step.call.arguments ?? {}).length}
						<p class="k">Arguments</p>
						<!-- A scrolling region must take keyboard focus (axe: scrollable-region-focusable). -->
						<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
						<pre tabindex="0" aria-label="Arguments of {step.call.name}">{JSON.stringify(
								step.call.arguments,
								null,
								2
							)}</pre>
					{/if}
					{#if step.result}
						<p class="k">
							Result
							{#if step.result.truncated}
								<span class="muted"
									>(first {formatQuantity(step.result.content.length, 0, 'characters')} of {formatQuantity(
										step.result.chars,
										0,
										'characters'
									)})</span
								>
							{/if}
						</p>
						<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
						<pre tabindex="0" aria-label="Result of {step.call.name}">{step.result.content}</pre>
					{/if}
				</li>
			{/each}
		</ol>
	</details>
{/if}

<style>
	.trace {
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
	}
	summary {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		min-height: 36px;
		padding: 4px var(--vo-space-3);
		cursor: pointer;
	}
	summary:focus-visible {
		outline-offset: -2px;
	}
	.label {
		font-weight: 600;
	}
	.muted {
		color: var(--vo-text-muted);
	}
	ol {
		margin: 0;
		padding: 0 var(--vo-space-3) var(--vo-space-3) 2.2em;
		display: grid;
		gap: var(--vo-space-3);
	}
	.head {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
	.name {
		font-weight: 600;
	}
	.k {
		margin-top: var(--vo-space-1);
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	pre {
		max-height: 14rem;
		overflow: auto;
		margin-top: 2px;
		padding: var(--vo-space-2);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface-raised);
		font-size: 12px;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
</style>
