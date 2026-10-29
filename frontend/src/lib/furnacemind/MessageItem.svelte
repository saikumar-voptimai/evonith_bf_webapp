<!--
@component
One message of a FurnaceMind conversation. A question shows its text (a quick skill's display text); an answer shows
its Markdown, the tool trace, buttons for the figures and datasets it produced (they open in the side panels), its
time, duration and model, a status when the model was unavailable, and the feedback controls.
-->
<script lang="ts">
	import BarChart3 from '@lucide/svelte/icons/chart-column';
	import Sparkles from '@lucide/svelte/icons/sparkles';
	import Table from '@lucide/svelte/icons/table';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import { formatQuantity } from '#lib/format/number.ts';
	import { istDateTime } from '#lib/time/format.ts';
	import type { FeedbackState } from './chat.svelte.ts';
	import FeedbackControls from './FeedbackControls.svelte';
	import {
		answerFacts,
		datasetSize,
		figureTitle,
		isSkillQuestion,
		messageArtifacts,
		questionText,
		type ArtifactRef,
		type Message
	} from './furnacemind.ts';
	import MarkdownText from './MarkdownText.svelte';
	import ToolTrace from './ToolTrace.svelte';

	interface Props {
		message: Message;
		/** Keys of the artifacts shown in the panels. */
		shown: (string | null)[];
		feedback: FeedbackState | undefined;
		/** First token of this answer, ms (answers streamed in this tab). */
		firstTokenMs?: number;
		readonly?: boolean;
		onpick: (ref: ArtifactRef) => void;
		onrate: (seq: number, rating: 'up' | 'down', comment: string) => Promise<boolean>;
	}

	let { message, shown, feedback, firstTokenMs, readonly = false, onpick, onrate }: Props = $props();

	const artifacts = $derived(messageArtifacts(message));
	const facts = $derived(answerFacts(message));
	const at = $derived(istDateTime(new Date(message.created_at)));
</script>

{#if message.role === 'user'}
	<div class="question">
		<div class="bubble">
			{#if isSkillQuestion(message)}
				<span class="skill"><Sparkles size={14} strokeWidth={2} aria-hidden="true" /> Quick skill</span>
			{/if}
			<p>{questionText(message)}</p>
		</div>
		<p class="meta">{at} IST</p>
	</div>
{:else}
	<article class="answer" aria-label="Answer {message.seq}">
		{#if message.status === 'failed'}
			<StatusChip
				kind="warning"
				label="Model unavailable"
				title={String(message.meta?.['model_error'] ?? '')}
			/>
		{/if}
		<MarkdownText text={message.content} />
		<ToolTrace parts={message.parts ?? []} />
		{#if artifacts.length}
			<div class="artifacts">
				{#each artifacts as ref (ref.key)}
					<button
						type="button"
						class={['artifact', shown.includes(ref.key) && 'on']}
						aria-pressed={shown.includes(ref.key)}
						onclick={() => onpick(ref)}
					>
						{#if ref.part.type === 'plotly'}
							<BarChart3 size={15} strokeWidth={2} aria-hidden="true" />
							<span>{ref.part.omitted ? 'Chart (too large to keep)' : figureTitle(ref.part)}</span>
						{:else}
							<Table size={15} strokeWidth={2} aria-hidden="true" />
							<span>{ref.part.dataset_id || 'Dataset'} · {datasetSize(ref.part)}</span>
						{/if}
					</button>
				{/each}
			</div>
		{/if}
		<p class="meta">
			{at} IST{#each facts as fact (fact)}<span> · {fact}</span>{/each}{#if firstTokenMs !== undefined}<span>
					· first token {formatQuantity(firstTokenMs / 1000, 1, 's')}</span
				>{/if}
		</p>
		<FeedbackControls
			seq={message.seq}
			saved={message.feedback}
			status={feedback}
			{readonly}
			onsave={(rating, comment) => onrate(message.seq, rating, comment)}
		/>
	</article>
{/if}

<style>
	.question {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 2px;
	}
	.bubble {
		max-width: min(46rem, 92%);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-control);
		background: var(--vo-cream);
		border: 1px solid var(--vo-cream-line);
		font-size: var(--vo-fs-body);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.skill {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 2px;
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-accent-ink);
	}
	.answer {
		display: grid;
		gap: var(--vo-space-2);
		min-width: 0;
	}
	.meta {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
		font-variant-numeric: tabular-nums;
	}
	.artifacts {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
	.artifact {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-height: 32px;
		max-width: 100%;
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
		color: var(--vo-text);
		font: inherit;
		font-size: var(--vo-fs-label);
		cursor: pointer;
	}
	.artifact span {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.artifact:hover {
		background: var(--vo-surface-raised);
	}
	.artifact.on {
		border-color: var(--vo-accent-strong);
		box-shadow: inset 0 0 0 1px var(--vo-accent-strong);
	}
</style>
