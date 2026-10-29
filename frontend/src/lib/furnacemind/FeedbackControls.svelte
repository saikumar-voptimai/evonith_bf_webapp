<!--
@component
Feedback on one answer (page 7's thumbs): Useful or Needs work, an optional comment, saved through the API with its
audit row. A saved rating is shown with its time; changing it saves again.
-->
<script lang="ts">
	import ThumbsDown from '@lucide/svelte/icons/thumbs-down';
	import ThumbsUp from '@lucide/svelte/icons/thumbs-up';
	import Button from '#lib/components/Button.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import { istDateTime } from '#lib/time/format.ts';
	import type { Feedback } from './furnacemind.ts';
	import type { FeedbackState } from './chat.svelte.ts';

	interface Props {
		seq: number;
		saved: Feedback | null | undefined;
		status: FeedbackState | undefined;
		/** Only the owner rates; others see the saved rating. */
		readonly?: boolean;
		onsave: (rating: 'up' | 'down', comment: string) => Promise<boolean>;
	}

	let { seq, saved, status, readonly = false, onsave }: Props = $props();

	const uid = $props.id();
	let rating = $state<'up' | 'down' | null>(null);
	let comment = $state('');
	let editing = $state(false);

	// Follow the stored feedback until the user starts a change.
	$effect(() => {
		if (editing) return;
		rating = saved?.rating ?? null;
		comment = saved?.comment ?? '';
	});

	const dirty = $derived(
		rating !== null && (rating !== saved?.rating || comment.trim() !== (saved?.comment ?? ''))
	);

	function choose(value: 'up' | 'down'): void {
		rating = value;
		editing = true;
	}

	async function save(): Promise<void> {
		if (!rating) return;
		if (await onsave(rating, comment)) editing = false;
	}
</script>

<div class="feedback" role="group" aria-label="Feedback on answer {seq}">
	{#if readonly}
		{#if saved}
			<StatusChip
				kind={saved.rating === 'up' ? 'ok' : 'warning'}
				label={saved.rating === 'up' ? 'Rated useful' : 'Rated needs work'}
				size="sm"
			/>
			{#if saved.comment}<span class="muted">“{saved.comment}”</span>{/if}
		{/if}
	{:else}
		<div class="row">
			<span class="k" id="{uid}-k">Was this useful?</span>
			<button
				type="button"
				class={['thumb', rating === 'up' && 'on']}
				aria-pressed={rating === 'up'}
				onclick={() => choose('up')}
			>
				<ThumbsUp size={15} strokeWidth={2} aria-hidden="true" /> Useful
			</button>
			<button
				type="button"
				class={['thumb', rating === 'down' && 'on']}
				aria-pressed={rating === 'down'}
				onclick={() => choose('down')}
			>
				<ThumbsDown size={15} strokeWidth={2} aria-hidden="true" /> Needs work
			</button>
			{#if saved && !dirty}
				<StatusChip
					kind="ok"
					label="Saved"
					size="sm"
					title="Saved {istDateTime(new Date(saved.updated_at))} IST"
				/>
			{/if}
		</div>
		{#if rating !== null && (editing || dirty)}
			<div class="comment">
				<label for="{uid}-c" class="sr-only">Comment on answer {seq}</label>
				<textarea
					id="{uid}-c"
					rows="2"
					maxlength="2000"
					placeholder={rating === 'down' ? 'What was wrong or missing?' : 'What was useful? (optional)'}
					bind:value={comment}
					oninput={() => (editing = true)}></textarea>
				<Button size="sm" variant="primary" disabled={!dirty || status?.saving} onclick={save}>
					{status?.saving ? 'Saving…' : 'Save feedback'}
				</Button>
			</div>
		{/if}
		{#if status?.error}
			<p class="error" role="alert">Feedback not saved: {status.error}</p>
		{/if}
	{/if}
</div>

<style>
	.feedback {
		display: grid;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-label);
	}
	.row {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
	.k,
	.muted {
		color: var(--vo-text-muted);
	}
	.thumb {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		min-height: 32px;
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text);
		font: inherit;
		font-weight: 550;
		cursor: pointer;
	}
	.thumb:hover {
		background: var(--vo-surface-raised);
	}
	.thumb.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent);
	}
	.comment {
		display: flex;
		align-items: flex-end;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
	textarea {
		flex: 1 1 16rem;
		min-height: 40px;
		padding: 6px 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font: inherit;
		resize: vertical;
	}
	textarea::placeholder {
		color: var(--vo-text-placeholder);
	}
	.error {
		color: var(--vo-critical);
	}
</style>
