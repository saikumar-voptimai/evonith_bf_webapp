<script lang="ts">
	/**
	 * FurnaceMind (SV-5.2, #313): page 7's agent on the SPA, through the agent service of SV-5.1. Conversations of the
	 * user (stored in the twin store), the answer streamed token by token with the tool trace, the figures (Plotly.js,
	 * loaded only here) and datasets (first rows, full CSV) in the side panels, page 7's quick skills, and feedback on
	 * every answer. State and the stream live in `FurnaceMindChat`; destroying the page aborts the stream read (the
	 * turn completes and is stored on the server) and stops the poll timer.
	 *
	 * Layouts: three columns from 1600 px (conversations, chat, panels), two from 1024 px (the conversations become a
	 * menu in the chat header), one below.
	 */
	import Brain from '@lucide/svelte/icons/brain';
	import MessageSquarePlus from '@lucide/svelte/icons/message-square-plus';
	import MessagesSquare from '@lucide/svelte/icons/messages-square';
	import Send from '@lucide/svelte/icons/send';
	import Sparkles from '@lucide/svelte/icons/sparkles';
	import Button from '#lib/components/Button.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import { FurnaceMindChat } from '#lib/furnacemind/chat.svelte.ts';
	import DatasetPanel from '#lib/furnacemind/DatasetPanel.svelte';
	import FigurePanel from '#lib/furnacemind/FigurePanel.svelte';
	import {
		messageArtifacts,
		REASONING_LEVELS,
		runningLabel,
		toolTrace
	} from '#lib/furnacemind/furnacemind.ts';
	import MarkdownText from '#lib/furnacemind/MarkdownText.svelte';
	import MessageItem from '#lib/furnacemind/MessageItem.svelte';
	import ToolTrace from '#lib/furnacemind/ToolTrace.svelte';
	import { session } from '#lib/session/session.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const chat = new FurnaceMindChat();

	$effect(() => chat.start());

	const EXAMPLES = [
		'Plot the fuel rate and ETA CO of the last 3 days.',
		'Which shifts last week had the highest coke rate, and why?',
		'Show the heat loads of rows R6 to R10 for the last 8 hours.'
	];

	let draft = $state('');
	let log = $state<HTMLDivElement>();

	const conversation = $derived(chat.conversation);
	const messages = $derived(conversation?.messages ?? []);
	const live = $derived(chat.live);
	const liveLabel = $derived(live ? runningLabel(toolTrace(live.parts, true)) : null);
	const liveArtifacts = $derived(live ? messageArtifacts({ seq: -1, parts: live.parts }, true) : []);
	const shown = $derived([chat.figure?.key ?? null, chat.dataset?.key ?? null]);
	const readonly = $derived(!!conversation && !!session.user && conversation.owner !== session.user.username);
	const title = $derived(conversation?.title || 'New conversation');
	const skills = $derived(chat.skills?.skills ?? []);

	// Keep the newest message in view while the answer streams, unless the reader has scrolled up.
	let pinned = true;
	function onscroll(): void {
		if (!log) return;
		pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
	}
	$effect(() => {
		void messages.length;
		void live?.text;
		void live?.parts.length;
		if (log && pinned) log.scrollTop = log.scrollHeight;
	});

	async function submit(event: SubmitEvent): Promise<void> {
		event.preventDefault();
		const text = draft;
		if (!text.trim() || chat.busy) return;
		draft = '';
		pinned = true;
		// The API did not take the question (nothing stored): give the text back unless a new one was typed.
		if (!(await chat.send(text)) && !draft) draft = text;
	}

	function onkeydown(event: KeyboardEvent): void {
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
			event.preventDefault();
			(event.currentTarget as HTMLTextAreaElement).form?.requestSubmit();
		}
	}

	function open(id: string): void {
		if (id) void chat.open(id);
		else chat.newConversation();
	}
</script>

<div class="fm">
	<aside class="history">
		<Panel title="Conversations" icon={MessagesSquare} note="Yours, newest first.">
			{#snippet actions()}
				<Button size="sm" icon={MessageSquarePlus} disabled={!!live} onclick={() => chat.newConversation()}
					>New</Button
				>
			{/snippet}
			{#if chat.listState.loading && !chat.conversations.length}
				<Skeleton lines={4} label="Loading the conversations" />
			{:else if chat.listState.error && !chat.conversations.length}
				<ErrorState
					title="Conversations not loaded"
					message={chat.listState.error}
					onretry={() => void chat.loadList()}
					compact
				/>
			{:else if !chat.conversations.length}
				<EmptyState title="No conversations yet" message="Ask a question to start one." compact />
			{:else}
				{#if chat.listState.error}
					<p class="warn">Refresh failed: {chat.listState.error} Showing the last list.</p>
				{/if}
				<ul class="list">
					{#each chat.conversations as c (c.id)}
						<li>
							<button
								type="button"
								class={['item', conversation?.id === c.id && 'current']}
								aria-current={conversation?.id === c.id ? 'true' : undefined}
								disabled={!!live}
								onclick={() => void chat.open(c.id)}
							>
								<span class="t">{c.title || 'Untitled'}</span>
								<span class="s"
									>{istDateTime(new Date(c.updated_at))} IST · {c.message_count}
									{c.message_count === 1 ? 'message' : 'messages'}</span
								>
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</Panel>
	</aside>

	<section class="chat" aria-label="FurnaceMind chat">
		<Panel {title} icon={Brain} note="Answers use plant data through FurnaceMind's tools. Times in IST.">
			{#snippet actions()}
				<label class="compact-history">
					<span class="k">Conversation</span>
					<select
						value={conversation?.id ?? ''}
						disabled={!!live}
						onchange={(e) => open(e.currentTarget.value)}
					>
						<option value="">New conversation</option>
						{#each chat.conversations as c (c.id)}
							<option value={c.id}>{c.title || 'Untitled'}</option>
						{/each}
					</select>
				</label>
				<label class="reasoning">
					<span class="k">Reasoning</span>
					<select bind:value={chat.reasoning} disabled={chat.busy}>
						{#each REASONING_LEVELS as level (level)}
							<option value={level}>{level}</option>
						{/each}
					</select>
				</label>
			{/snippet}

			<!-- A scrolling region must take keyboard focus (axe: scrollable-region-focusable). -->
			<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
			<div
				class="log"
				bind:this={log}
				{onscroll}
				tabindex="0"
				role="region"
				aria-label="Messages"
				aria-busy={!!live || undefined}
			>
				{#if chat.conversationState.loading && !conversation}
					<Skeleton variant="block" height="240px" label="Loading the conversation" />
				{:else if chat.conversationState.error && !conversation}
					<ErrorState
						title="Conversation not loaded"
						message={chat.conversationState.error}
						onretry={() => void chat.loadList(true)}
						compact
					/>
				{:else if !messages.length && !live}
					<EmptyState
						icon={Brain}
						title="Ask about the furnace"
						message="FurnaceMind fetches the plant data it needs, draws charts and explains them. Try one of these:"
					>
						{#snippet action()}
							<div class="examples">
								{#each EXAMPLES as example (example)}
									<Button size="sm" variant="ghost" onclick={() => (draft = example)}>{example}</Button>
								{/each}
							</div>
						{/snippet}
					</EmptyState>
				{:else}
					<ol class="messages">
						{#each messages as message (message.seq)}
							<li>
								<MessageItem
									{message}
									{shown}
									{readonly}
									feedback={chat.feedback[message.seq]}
									firstTokenMs={chat.firstToken[message.seq]}
									onpick={(ref) => chat.pick(ref)}
									onrate={(seq, rating, comment) => chat.rate(seq, rating, comment)}
								/>
							</li>
						{/each}
						{#if live}
							{#if !live.question}
								<li class="question-pending">
									<p class="bubble">{live.pending}</p>
								</li>
							{/if}
							<li class="answer">
								<div class="status" role="status">
									{#if liveLabel}
										<StatusChip kind="info" label="Running" size="sm" />
										<span class="muted">{liveLabel}</span>
									{:else if live.text}
										<StatusChip kind="info" label="Answering" size="sm" />
									{:else}
										<StatusChip kind="info" label="Thinking" size="sm" />
										<span class="muted">Waiting for the first words of the answer…</span>
									{/if}
								</div>
								{#if live.text}<MarkdownText text={live.text} />{/if}
								<ToolTrace parts={live.parts} running open />
								{#if liveArtifacts.length}
									<div class="live-artifacts">
										{#each liveArtifacts as ref (ref.key)}
											<Button size="sm" variant="ghost" onclick={() => chat.pick(ref)}>
												{ref.part.type === 'plotly' ? 'Chart ready' : 'Dataset ready'}
											</Button>
										{/each}
									</div>
								{/if}
							</li>
						{:else if conversation?.running}
							<li class="answer">
								<div class="status" role="status">
									<StatusChip kind="info" label="Answer in progress" size="sm" />
									<span class="muted"
										>A turn of this conversation is running (another tab, or before a reload). Its answer
										appears here when it is stored.</span
									>
								</div>
							</li>
						{/if}
					</ol>
				{/if}
			</div>

			{#if chat.sendError}
				<p class="error" role="alert">{chat.sendError}</p>
			{/if}

			<div class="skills" aria-label="Quick skills" role="group">
				<span class="k"><Sparkles size={14} strokeWidth={2} aria-hidden="true" /> Quick skills</span>
				{#if chat.skillsState.loading && !chat.skills}
					<Skeleton lines={1} width="18rem" label="Loading the quick skills" />
				{:else if chat.skillsState.error}
					<span class="warn">Quick skills not loaded: {chat.skillsState.error}</span>
					<Button size="sm" variant="ghost" onclick={() => void chat.loadSkills()}>Retry</Button>
				{:else if !skills.length}
					<span class="muted">No active quick skills.</span>
				{:else}
					{#each skills as skill (skill.slug)}
						<Button
							size="sm"
							disabled={chat.busy || readonly}
							title={`${skill.description} (${skill.display})`}
							onclick={() => {
								pinned = true;
								void chat.runSkill(skill.slug);
							}}>{skill.name}</Button
						>
					{/each}
					{#if chat.skills}
						<span class="muted"
							>Shift skills use shift {chat.skills.shift_label} of {chat.skills.shift_date}.</span
						>
					{/if}
				{/if}
			</div>

			<form class="composer" onsubmit={submit}>
				<label for="fm-question" class="sr-only">Question</label>
				<textarea
					id="fm-question"
					rows="2"
					maxlength="8000"
					placeholder={readonly
						? 'This conversation belongs to another user.'
						: 'Ask FurnaceMind… (Enter sends, Shift+Enter starts a new line)'}
					bind:value={draft}
					{onkeydown}
					disabled={readonly}></textarea>
				<Button type="submit" variant="primary" icon={Send} disabled={!draft.trim() || chat.busy || readonly}>
					{live ? 'Answering…' : 'Send'}
				</Button>
			</form>
		</Panel>
	</section>

	<section class="side" aria-label="Answer artifacts">
		<FigurePanel artifact={chat.figure} />
		<DatasetPanel artifact={chat.dataset} fullCsv={(part) => chat.fullCsv(part)} />
	</section>
</div>

<style>
	.fm {
		display: grid;
		gap: var(--vo-space-4);
		grid-template-columns: minmax(0, 1fr);
		grid-template-areas: 'chat' 'side';
		align-items: start;
	}
	.history {
		grid-area: history;
		display: none;
	}
	.chat {
		grid-area: chat;
		min-width: 0;
	}
	.side {
		grid-area: side;
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	@media (min-width: 1024px) {
		.fm {
			grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
			grid-template-areas: 'chat side';
		}
	}
	@media (min-width: 1600px) {
		.fm {
			grid-template-columns: 17rem minmax(0, 1.2fr) minmax(0, 1fr);
			grid-template-areas: 'history chat side';
		}
		.history {
			display: block;
		}
		.compact-history {
			display: none;
		}
	}
	.k {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.compact-history,
	.reasoning {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	select {
		min-height: 32px;
		max-width: 14rem;
		padding: 0 8px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.list {
		display: grid;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
		max-height: calc(100vh - 16rem);
		overflow-y: auto;
	}
	.item {
		display: grid;
		gap: 2px;
		width: 100%;
		min-height: 40px;
		padding: 6px 10px;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text);
		font: inherit;
		text-align: left;
		cursor: pointer;
	}
	.item:hover:not(:disabled) {
		background: var(--vo-surface-raised);
	}
	.item.current {
		background: var(--vo-accent-soft);
	}
	.item .t {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-body);
		font-weight: 550;
	}
	.item .s {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.log {
		max-height: calc(100vh - 22rem);
		min-height: 16rem;
		overflow-y: auto;
		padding-right: 4px;
		scrollbar-width: thin;
	}
	.log:focus-visible {
		outline-offset: 2px;
	}
	@media (max-width: 1023px) {
		.log {
			max-height: none;
		}
	}
	.messages {
		display: grid;
		gap: var(--vo-space-4);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.question-pending {
		display: flex;
		justify-content: flex-end;
	}
	.bubble {
		max-width: min(46rem, 92%);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-control);
		background: var(--vo-cream);
		border: 1px solid var(--vo-cream-line);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.answer {
		display: grid;
		gap: var(--vo-space-2);
	}
	.status {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-label);
	}
	.live-artifacts,
	.examples {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
	.examples {
		justify-content: center;
	}
	.muted {
		color: var(--vo-text-muted);
	}
	.warn {
		color: var(--vo-warning);
		font-size: var(--vo-fs-label);
	}
	.error {
		margin-top: var(--vo-space-3);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-small);
		background: var(--vo-critical-bg);
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
	}
	.skills {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		margin-top: var(--vo-space-3);
		padding-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-border);
		font-size: var(--vo-fs-label);
	}
	.composer {
		display: flex;
		align-items: flex-end;
		gap: var(--vo-space-2);
		margin-top: var(--vo-space-3);
	}
	textarea {
		flex: 1;
		min-height: 44px;
		max-height: 12rem;
		padding: 8px 12px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font: inherit;
		font-size: var(--vo-fs-body);
		resize: vertical;
	}
	textarea::placeholder {
		color: var(--vo-text-placeholder);
	}
</style>
