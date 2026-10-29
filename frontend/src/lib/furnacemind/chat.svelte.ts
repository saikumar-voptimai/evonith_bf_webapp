/**
 * State of the FurnaceMind page (SV-5.2, #313): the user's conversations, the open conversation with its messages,
 * the turn that streams in this tab, page 7's quick skills, the feedback on answers and the artifacts shown in the
 * figure and dataset panels. `start()` loads the page; the function it returns aborts a running stream read and
 * stops the poll timer (the page calls it on destroy). The turn itself continues on the server and is stored: a
 * conversation opened while its turn runs elsewhere is polled until the answer is stored.
 */
import { api, ApiError } from '../api/client.ts';
import { describeError } from '../api/describe.ts';
import {
	latestArtifacts,
	messageArtifacts,
	parseStreamEvent,
	type ArtifactRef,
	type Conversation,
	type ConversationSummary,
	type DatasetPart,
	type Feedback,
	type Message,
	type Part,
	type ReasoningLevel,
	type SkillList,
	type StreamEvent
} from './furnacemind.ts';
import { postEventStream } from './sse.ts';

export interface AskBody {
	content?: string;
	skill?: string;
	reasoning_level?: ReasoningLevel;
}

export interface FmApi {
	list(): Promise<ConversationSummary[]>;
	get(id: string): Promise<Conversation>;
	create(level: ReasoningLevel): Promise<Conversation>;
	skills(): Promise<SkillList>;
	rate(id: string, seq: number, rating: 'up' | 'down', comment: string): Promise<Feedback>;
	/** The full dataset as CSV; an `ApiError` 404 when the API no longer holds it. */
	csv(id: string, artifactId: string): Promise<string>;
	ask(id: string, body: AskBody, onEvent: (event: StreamEvent) => void, signal: AbortSignal): Promise<void>;
}

export const fmApi: FmApi = {
	async list() {
		return (await api.GET('/api/fm/conversations', { params: { query: { limit: 50 } } })).data!;
	},
	async get(id) {
		return (
			await api.GET('/api/fm/conversations/{conversation_id}', { params: { path: { conversation_id: id } } })
		).data!;
	},
	async create(level) {
		return (await api.POST('/api/fm/conversations', { body: { title: '', reasoning_level: level } })).data!;
	},
	async skills() {
		return (await api.GET('/api/fm/skills')).data!;
	},
	async rate(id, seq, rating, comment) {
		const { data } = await api.PUT('/api/fm/conversations/{conversation_id}/messages/{seq}/feedback', {
			params: { path: { conversation_id: id, seq } },
			body: { rating, comment }
		});
		return data!;
	},
	async csv(id, artifactId) {
		const { data } = await api.GET('/api/fm/conversations/{conversation_id}/datasets/{artifact_id}/csv', {
			params: { path: { conversation_id: id, artifact_id: artifactId } },
			parseAs: 'text'
		});
		return data as unknown as string;
	},
	async ask(id, body, onEvent, signal) {
		await postEventStream(
			`/api/fm/conversations/${encodeURIComponent(id)}/messages`,
			body,
			(frame) => {
				const event = parseStreamEvent(frame.data);
				if (event) onEvent(event);
			},
			{ signal }
		);
	}
};

export interface LoadState {
	loading: boolean;
	error: string | null;
}

const idle = (): LoadState => ({ loading: false, error: null });

/** The turn streaming in this tab. */
export interface LiveTurn {
	/** The stored question (null until `message.user`). */
	question: Message | null;
	/** What was sent, shown until the stored question arrives. */
	pending: string;
	/** Visible text of the current model call. */
	text: string;
	step: number;
	parts: Part[];
	sentAt: number;
	firstTokenMs: number | null;
}

export interface FeedbackState {
	saving: boolean;
	error: string | null;
}

export interface Clock {
	now(): number;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimeout(id: unknown): void;
}

const browserClock: Clock = {
	now: () => performance.now(),
	setTimeout: (fn, ms) => setTimeout(fn, ms),
	clearTimeout: (id) => clearTimeout(id as ReturnType<typeof setTimeout>)
};

/** Poll period while a turn of the open conversation runs elsewhere. */
export const POLL_MS = 2_000;

export class FurnaceMindChat {
	conversations = $state<ConversationSummary[]>([]);
	listState = $state<LoadState>(idle());
	conversation = $state<Conversation | null>(null);
	conversationState = $state<LoadState>(idle());
	skills = $state<SkillList | null>(null);
	skillsState = $state<LoadState>(idle());
	live = $state<LiveTurn | null>(null);
	sendError = $state<string | null>(null);
	reasoning = $state<ReasoningLevel>('Medium');
	/** Keys of the figure and the dataset picked in the chat (null: the newest). */
	picked = $state<{ figure: string | null; dataset: string | null }>({ figure: null, dataset: null });
	feedback = $state<Record<number, FeedbackState>>({});
	/** First-token time of the answers streamed in this tab, ms by answer seq. */
	firstToken = $state<Record<number, number>>({});

	private abort: AbortController | null = null;
	private poll: unknown = null;
	private disposed = false;

	constructor(
		private readonly client: FmApi = fmApi,
		private readonly clock: Clock = browserClock
	) {}

	/** Every artifact of the open conversation and of the streaming turn, oldest first. */
	get artifacts(): ArtifactRef[] {
		const refs = (this.conversation?.messages ?? []).flatMap((m) => messageArtifacts(m));
		if (this.live) refs.push(...messageArtifacts({ seq: -1, parts: this.live.parts }, true));
		return refs;
	}

	get figure(): ArtifactRef | null {
		const refs = this.artifacts;
		return refs.find((r) => r.key === this.picked.figure) ?? latestArtifacts(refs).figure;
	}

	get dataset(): ArtifactRef | null {
		const refs = this.artifacts;
		return refs.find((r) => r.key === this.picked.dataset) ?? latestArtifacts(refs).dataset;
	}

	get busy(): boolean {
		return this.live !== null || !!this.conversation?.running;
	}

	start(): () => void {
		this.disposed = false;
		void this.loadSkills();
		void this.loadList(true);
		return () => this.stop();
	}

	stop(): void {
		this.disposed = true;
		this.abort?.abort();
		this.abort = null;
		this.stopPolling();
	}

	async loadList(openNewest = false): Promise<void> {
		this.listState = { loading: true, error: null };
		try {
			this.conversations = await this.client.list();
			this.listState = idle();
			const newest = this.conversations[0];
			if (openNewest && newest && !this.conversation) await this.open(newest.id);
		} catch (error) {
			this.listState = { loading: false, error: describeError(error).message };
		}
	}

	async loadSkills(): Promise<void> {
		this.skillsState = { loading: true, error: null };
		try {
			this.skills = await this.client.skills();
			this.skillsState = idle();
		} catch (error) {
			this.skillsState = { loading: false, error: describeError(error).message };
		}
	}

	async open(id: string): Promise<void> {
		if (this.live) return;
		this.stopPolling();
		this.conversationState = { loading: true, error: null };
		this.sendError = null;
		try {
			const conversation = await this.client.get(id);
			if (this.disposed) return;
			this.conversation = conversation;
			this.reasoning = conversation.reasoning_level;
			this.picked = { figure: null, dataset: null };
			this.feedback = {};
			this.conversationState = idle();
			this.schedulePoll();
		} catch (error) {
			this.conversationState = { loading: false, error: describeError(error).message };
		}
	}

	/** Start an empty conversation (it is created on the first question). */
	newConversation(): void {
		if (this.live) return;
		this.stopPolling();
		this.conversation = null;
		this.conversationState = idle();
		this.sendError = null;
		this.picked = { figure: null, dataset: null };
		this.feedback = {};
	}

	/** Ask a typed question. False when the API did not take it (nothing stored: the text can be sent again). */
	async send(text: string): Promise<boolean> {
		const content = text.trim();
		if (!content) return false;
		return this.ask({ content }, content);
	}

	/** Run a quick skill (page 7's button); its prompt becomes the question. */
	async runSkill(slug: string): Promise<boolean> {
		const skill = this.skills?.skills?.find((s) => s.slug === slug);
		return this.ask({ skill: slug }, skill?.display ?? slug);
	}

	private async ask(body: AskBody, pending: string): Promise<boolean> {
		if (this.busy) return false;
		this.sendError = null;
		this.live = {
			question: null,
			pending,
			text: '',
			step: 0,
			parts: [],
			sentAt: this.clock.now(),
			firstTokenMs: null
		};
		let conversation = this.conversation;
		if (!conversation) {
			try {
				conversation = await this.client.create(this.reasoning);
				this.conversation = conversation;
			} catch (error) {
				this.sendError = describeError(error).message;
				this.live = null;
				return false;
			}
		}
		const id = conversation.id;
		const controller = new AbortController();
		this.abort = controller;
		let accepted = false;
		let ended = false;
		try {
			await this.client.ask(
				id,
				{ ...body, reasoning_level: this.reasoning },
				(event) => {
					accepted = true;
					if (event.type === 'message.assistant' || event.type === 'error') ended = true;
					this.apply(id, event);
				},
				controller.signal
			);
			accepted = true;
			if (!ended && !this.disposed)
				this.sendError = 'The answer stream ended early. Reloading the conversation.';
		} catch (error) {
			if (controller.signal.aborted) return accepted;
			this.sendError =
				error instanceof ApiError && error.status === 409
					? 'A turn of this conversation is still running. Its answer appears here when it is stored.'
					: describeError(error).message;
		} finally {
			if (this.abort === controller) this.abort = null;
			this.live = null;
		}
		if (this.disposed) return accepted;
		if (!ended) await this.refresh(id);
		void this.loadList();
		return accepted;
	}

	private apply(id: string, event: StreamEvent): void {
		const live = this.live;
		if (!live || this.conversation?.id !== id) return;
		switch (event.type) {
			case 'message.user':
				live.question = event.message;
				this.conversation.messages = [...(this.conversation.messages ?? []), event.message];
				break;
			case 'token':
				if (event.step !== live.step) {
					live.step = event.step;
					live.text = '';
				}
				live.text += event.text;
				if (live.firstTokenMs === null && event.text) live.firstTokenMs = this.clock.now() - live.sentAt;
				break;
			case 'tool.call':
			case 'tool.result':
				live.parts = [...live.parts, event.part];
				break;
			case 'artifact':
				live.parts = [...live.parts, event.part];
				this.picked = {
					figure: event.part.type === 'plotly' ? null : this.picked.figure,
					dataset: event.part.type === 'dataset' ? null : this.picked.dataset
				};
				break;
			case 'message.assistant':
				this.conversation.messages = [...(this.conversation.messages ?? []), event.message];
				this.conversation.message_count = event.message.seq;
				if (live.firstTokenMs !== null)
					this.firstToken = { ...this.firstToken, [event.message.seq]: live.firstTokenMs };
				break;
			case 'error':
				this.sendError = event.detail ? `${event.title}. ${event.detail}` : event.title;
				void this.refresh(id);
				break;
		}
	}

	/** Re-read the conversation (after an error or an interrupted stream), keeping the panel picks. */
	async refresh(id: string): Promise<void> {
		try {
			const conversation = await this.client.get(id);
			if (this.disposed || (this.conversation && this.conversation.id !== id)) return;
			this.conversation = conversation;
			this.schedulePoll();
		} catch (error) {
			this.conversationState = { loading: false, error: describeError(error).message };
		}
	}

	private schedulePoll(): void {
		this.stopPolling();
		const conversation = this.conversation;
		if (this.disposed || !conversation?.running || this.live) return;
		this.poll = this.clock.setTimeout(() => {
			this.poll = null;
			void this.refresh(conversation.id);
		}, POLL_MS);
	}

	private stopPolling(): void {
		if (this.poll !== null) this.clock.clearTimeout(this.poll);
		this.poll = null;
	}

	pick(ref: ArtifactRef): void {
		this.picked =
			ref.part.type === 'plotly' ? { ...this.picked, figure: ref.key } : { ...this.picked, dataset: ref.key };
	}

	async rate(seq: number, rating: 'up' | 'down', comment: string): Promise<boolean> {
		const conversation = this.conversation;
		if (!conversation) return false;
		this.feedback = { ...this.feedback, [seq]: { saving: true, error: null } };
		try {
			const saved = await this.client.rate(conversation.id, seq, rating, comment);
			conversation.messages = (conversation.messages ?? []).map((m) =>
				m.seq === seq ? { ...m, feedback: saved } : m
			);
			this.feedback = { ...this.feedback, [seq]: { saving: false, error: null } };
			return true;
		} catch (error) {
			this.feedback = { ...this.feedback, [seq]: { saving: false, error: describeError(error).message } };
			return false;
		}
	}

	/**
	 * CSV of a dataset: the full DataFrame while the API holds it, else null (the caller offers the preview).
	 * Other failures throw.
	 */
	async fullCsv(part: DatasetPart): Promise<string | null> {
		const conversation = this.conversation;
		if (!conversation || !part.artifact_id) return null;
		try {
			return await this.client.csv(conversation.id, part.artifact_id);
		} catch (error) {
			if (error instanceof ApiError && error.status === 404) return null;
			throw error;
		}
	}
}
