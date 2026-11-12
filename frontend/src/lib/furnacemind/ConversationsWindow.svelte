<!--
@component
The user's FurnaceMind conversations in a floating window (user decision 12 Nov 2026): grouped by IST day, newest
first, those older than a week folded away behind "Show older", with a date filter and a text filter. Choosing one
opens it in the chat; New starts a conversation.
-->
<script lang="ts">
	import { SvelteMap } from 'svelte/reactivity';
	import MessageSquarePlus from '@lucide/svelte/icons/message-square-plus';
	import MessagesSquare from '@lucide/svelte/icons/messages-square';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import FloatingWindow from '../components/FloatingWindow.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import { istDay, istDateTime } from '../time/format.ts';
	import type { ConversationSummary } from './furnacemind.ts';

	interface Props {
		open: boolean;
		conversations: ConversationSummary[];
		currentId: string | null;
		loading?: boolean;
		error?: string | null;
		/** The chat is busy: no switching while an answer streams. */
		busy?: boolean;
		onopen: (id: string) => void;
		onnew: () => void;
		onretry: () => void;
	}

	let {
		open = $bindable(false),
		conversations,
		currentId,
		loading = false,
		error = null,
		busy = false,
		onopen,
		onnew,
		onretry
	}: Props = $props();

	const WEEK_MS = 7 * 86_400_000;
	let showOlder = $state(false);
	let day = $state('');
	let query = $state('');

	const recentFrom = $derived(Date.now() - WEEK_MS);
	const older = $derived(conversations.filter((c) => Date.parse(c.updated_at) < recentFrom).length);
	const filtered = $derived.by(() => {
		const q = query.trim().toLowerCase();
		return conversations
			.filter((c) => showOlder || day || Date.parse(c.updated_at) >= recentFrom)
			.filter((c) => !day || istDay(new Date(Date.parse(c.updated_at))) === day)
			.filter((c) => !q || (c.title || 'Untitled').toLowerCase().includes(q))
			.sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
	});
	const groups = $derived.by(() => {
		const map = new SvelteMap<string, ConversationSummary[]>();
		for (const c of filtered) {
			const d = istDay(new Date(Date.parse(c.updated_at)));
			if (!map.has(d)) map.set(d, []);
			map.get(d)!.push(c);
		}
		const today = istDay(new Date());
		const yesterday = istDay(new Date(Date.now() - 86_400_000));
		return [...map.entries()].map(([d, list]) => ({
			day: d,
			label: d === today ? 'Today' : d === yesterday ? 'Yesterday' : d,
			list
		}));
	});
	const today = istDay(new Date());
</script>

<FloatingWindow
	bind:open
	title="Conversations"
	icon={MessagesSquare}
	storageKey="fm-conversations"
	width={400}
	height={560}
>
	{#snippet actions()}
		<Button size="sm" icon={MessageSquarePlus} disabled={busy} onclick={onnew}>New</Button>
	{/snippet}
	<div class="filters">
		<input
			type="search"
			placeholder="Find by title"
			bind:value={query}
			aria-label="Find a conversation by title"
		/>
		<input type="date" bind:value={day} max={today} aria-label="Conversations of a day" />
		{#if day || query}
			<Button
				size="sm"
				variant="ghost"
				onclick={() => {
					day = '';
					query = '';
				}}>Clear</Button
			>
		{/if}
	</div>
	{#if loading && !conversations.length}
		<div class="pad"><Skeleton lines={4} label="Loading the conversations" /></div>
	{:else if error && !conversations.length}
		<div class="pad"><ErrorState compact title="Conversations not loaded" message={error} {onretry} /></div>
	{:else if !conversations.length}
		<div class="pad">
			<EmptyState compact title="No conversations yet" message="Ask a question to start one." />
		</div>
	{:else if !filtered.length}
		<div class="pad">
			<EmptyState
				compact
				title={day || query ? 'Nothing matches' : 'Nothing in the last week'}
				message={day || query ? 'Clear the filters to see every conversation.' : 'Show the older ones below.'}
			/>
		</div>
	{:else}
		{#each groups as g (g.day)}
			<section class="day" aria-label={g.label}>
				<h3>{g.label}</h3>
				<ul>
					{#each g.list as c (c.id)}
						<li>
							<button
								type="button"
								class:on={c.id === currentId}
								aria-current={c.id === currentId ? 'true' : undefined}
								disabled={busy}
								onclick={() => onopen(c.id)}
							>
								<span class="t">{c.title || 'Untitled'}</span>
								<span class="s"
									>{istDateTime(new Date(Date.parse(c.updated_at)))} IST · {c.message_count}
									{c.message_count === 1 ? 'message' : 'messages'}</span
								>
							</button>
						</li>
					{/each}
				</ul>
			</section>
		{/each}
	{/if}
	{#if older > 0 && !day}
		<div class="pad older">
			<Button size="sm" variant="ghost" onclick={() => (showOlder = !showOlder)}>
				{showOlder ? 'Hide' : 'Show'} the {older} older than a week
			</Button>
		</div>
	{/if}
</FloatingWindow>

<style>
	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-bottom: 1px solid var(--vo-border);
	}
	.filters input {
		flex: 1;
		min-width: 9rem;
		min-height: 34px;
		padding: 0 8px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
	}
	.pad {
		padding: var(--vo-space-3);
	}
	.older {
		border-top: 1px solid var(--vo-border);
	}
	.day h3 {
		margin: 0;
		padding: 6px var(--vo-space-3);
		background: var(--vo-surface-raised);
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	li button {
		display: grid;
		gap: 2px;
		width: 100%;
		padding: 8px var(--vo-space-3);
		border: 0;
		border-bottom: 1px solid var(--vo-border);
		background: none;
		text-align: left;
	}
	li button:hover:not(:disabled) {
		background: var(--vo-bg);
	}
	li button.on {
		background: var(--vo-cream);
		box-shadow: inset 3px 0 0 var(--vo-accent);
	}
	li button:disabled {
		cursor: not-allowed;
	}
	.t {
		font-size: var(--vo-fs-body);
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.s {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
