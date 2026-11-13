<script lang="ts">
	import { untrack } from 'svelte';
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
	import ChartColumn from '@lucide/svelte/icons/chart-column';
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
	import ArtifactsWindow from '#lib/furnacemind/ArtifactsWindow.svelte';
	import ConversationsWindow from '#lib/furnacemind/ConversationsWindow.svelte';
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

	const chat = new FurnaceMindChat();

	// start() writes state it also reads: untracked, so the effect does not restart itself (a request loop)
	$effect(() => untrack(() => chat.start()));

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

	// The windows (user decision 12 Nov 2026): the conversations on request; the artifacts open by themselves when an
	// answer brings a new chart or dataset, on the tab of the newcomer.
	let conversationsOpen = $state(false);
	let artifactsOpen = $state(false);
	let artifactsTab = $state<'chart' | 'dataset'>('chart');
	let lastFigure: string | null = null;
	let lastDataset: string | null = null;
	$effect(() => {
		const f = chat.figure?.key ?? null;
		const d = chat.dataset?.key ?? null;
		untrack(() => {
			const newFigure = f !== null && f !== lastFigure && lastFigure !== null;
			const newDataset = d !== null && d !== lastDataset && lastDataset !== null;
			if (newFigure || newDataset) {
				artifactsTab = newFigure ? 'chart' : 'dataset';
				artifactsOpen = true;
			}
			lastFigure = f;
			lastDataset = d;
		});
	});
</script>

<div class="fm">
	<section class="chat" aria-label="FurnaceMind chat">
		<Panel {title} icon={Brain} note="Answers use plant data through FurnaceMind's tools. Times in IST.">
			{#snippet actions()}
				<Button size="sm" icon={MessagesSquare} onclick={() => (conversationsOpen = !conversationsOpen)}
					>Conversations{chat.conversations.length ? ` (${chat.conversations.length})` : ''}</Button
				>
				<Button size="sm" icon={MessageSquarePlus} disabled={!!live} onclick={() => chat.newConversation()}
					>New</Button
				>
				<Button
					size="sm"
					icon={ChartColumn}
					disabled={!chat.figure && !chat.dataset}
					onclick={() => (artifactsOpen = !artifactsOpen)}
					aria-expanded={artifactsOpen}>Artifacts</Button
				>
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
</div>

<ConversationsWindow
	bind:open={conversationsOpen}
	conversations={chat.conversations}
	currentId={conversation?.id ?? null}
	loading={chat.listState.loading}
	error={chat.listState.error}
	busy={!!live}
	onopen={(id) => {
		open(id);
		conversationsOpen = false;
	}}
	onnew={() => {
		chat.newConversation();
		conversationsOpen = false;
	}}
	onretry={() => void chat.loadList()}
/>
<ArtifactsWindow
	bind:open={artifactsOpen}
	bind:tab={artifactsTab}
	figure={chat.figure}
	dataset={chat.dataset}
	fullCsv={(part) => chat.fullCsv(part)}
/>

<style>
	.fm {
		display: grid;
		gap: var(--vo-space-4);
		grid-template-columns: minmax(0, 1fr);
		align-items: start;
	}
	.chat {
		min-width: 0;
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
	.log {
		max-height: calc(100vh - 20rem);
		min-height: 24rem;
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
	/* minmax(0, 1fr): the column follows the log's width, not the longest unbreakable line of a message. */
	.messages {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
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
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-2);
		min-width: 0;
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
