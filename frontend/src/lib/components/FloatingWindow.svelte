<!--
@component
A floating window: a non-modal panel the operator moves aside (drag by its header), resizes (the corner), folds to
its header or closes, while the page beneath stays live. Its place and size are remembered per `storageKey` in the
viewer's browser. Below 1024 px it is a bottom sheet instead. Used for lists that would otherwise take the content's
width: optimiser runs, FurnaceMind conversations and artifacts.
-->
<script lang="ts">
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import ChevronUp from '@lucide/svelte/icons/chevron-up';
	import X from '@lucide/svelte/icons/x';
	import { untrack, type Component, type Snippet } from 'svelte';

	interface Props {
		open: boolean;
		title: string;
		icon?: Component<{
			size?: number | string;
			strokeWidth?: number | string;
			'aria-hidden'?: boolean | 'true';
		}>;
		/** One line under the title. */
		note?: string;
		/** Remembers place, size and fold under this key (`localStorage`); none: the defaults every time. */
		storageKey?: string;
		width?: number;
		height?: number;
		minWidth?: number;
		minHeight?: number;
		/** Header controls (buttons, filters). */
		actions?: Snippet;
		children: Snippet;
	}

	let {
		open = $bindable(false),
		title,
		icon: Icon,
		note,
		storageKey,
		width = 440,
		height = 560,
		minWidth = 300,
		minHeight = 160,
		actions,
		children
	}: Props = $props();

	interface Placement {
		left: number;
		top: number;
		width: number;
		height: number;
		folded: boolean;
	}

	// Stacking: the window last touched sits on top.
	let zTop = 60;
	const uid = $props.id();

	let box = $state<Placement>({ left: -1, top: 96, width: 0, height: 0, folded: false });
	let z = $state(zTop);
	let frame = $state<HTMLDivElement>();
	let sheet = $state(false);

	function read(): Placement | null {
		if (!storageKey) return null;
		try {
			const raw = localStorage.getItem(`vo-window:${storageKey}`);
			if (!raw) return null;
			const p = JSON.parse(raw) as Partial<Placement>;
			if (typeof p.left !== 'number' || typeof p.top !== 'number') return null;
			return {
				left: p.left,
				top: p.top,
				width: typeof p.width === 'number' ? p.width : width,
				height: typeof p.height === 'number' ? p.height : height,
				folded: !!p.folded
			};
		} catch {
			return null;
		}
	}

	function remember(): void {
		if (!storageKey) return;
		try {
			localStorage.setItem(`vo-window:${storageKey}`, JSON.stringify(box));
		} catch {
			/* private window, blocked storage: the defaults next time */
		}
	}

	/** Keep the window inside the viewport with its header reachable. */
	function clamp(p: Placement): Placement {
		if (typeof window === 'undefined') return p;
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		const w = Math.min(Math.max(p.width, minWidth), Math.max(minWidth, vw - 16));
		const h = Math.min(Math.max(p.height, minHeight), Math.max(minHeight, vh - 16));
		const left =
			p.left < 0 ? Math.max(8, vw - w - 24) : Math.min(Math.max(8, p.left), Math.max(8, vw - w - 8));
		const top = Math.min(Math.max(8, p.top), Math.max(8, vh - 56));
		return { ...p, left, top, width: w, height: h };
	}

	function raise(): void {
		z = ++zTop;
	}

	$effect(() => {
		if (!open) return;
		const media = window.matchMedia('(max-width: 1023px)');
		const apply = () => (sheet = media.matches);
		apply();
		media.addEventListener('change', apply);
		// the window's own state is read and written here: untracked, or the effect would restart itself
		untrack(() => {
			box = clamp(read() ?? { ...box, left: -1, top: 96, width, height });
			raise();
		});
		const onResize = () => (box = clamp(box));
		window.addEventListener('resize', onResize);
		return () => {
			media.removeEventListener('change', apply);
			window.removeEventListener('resize', onResize);
		};
	});

	// The corner resize (CSS) is read back and remembered.
	$effect(() => {
		const el = frame;
		if (!el || !open || sheet) return;
		const observer = new ResizeObserver(() => {
			const r = el.getBoundingClientRect();
			if (Math.abs(r.width - box.width) > 1 || (!box.folded && Math.abs(r.height - box.height) > 1)) {
				box = { ...box, width: r.width, height: box.folded ? box.height : r.height };
				remember();
			}
		});
		observer.observe(el);
		return () => observer.disconnect();
	});

	let drag: { x: number; y: number; left: number; top: number } | null = null;
	function onPointerDown(e: PointerEvent): void {
		if (sheet || e.button !== 0) return;
		const target = e.target as HTMLElement;
		if (target.closest('button, a, input, select')) return;
		drag = { x: e.clientX, y: e.clientY, left: box.left, top: box.top };
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		raise();
	}
	function onPointerMove(e: PointerEvent): void {
		if (!drag) return;
		box = clamp({ ...box, left: drag.left + e.clientX - drag.x, top: drag.top + e.clientY - drag.y });
	}
	function onPointerUp(): void {
		if (!drag) return;
		drag = null;
		remember();
	}

	function fold(): void {
		box = { ...box, folded: !box.folded };
		remember();
	}
	function close(): void {
		open = false;
	}
	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.stopPropagation();
			close();
		}
	}
</script>

{#if open}
	<div
		bind:this={frame}
		class={['window', sheet && 'sheet', box.folded && 'folded']}
		role="dialog"
		aria-modal="false"
		aria-labelledby="{uid}-title"
		tabindex="-1"
		style:left={sheet ? undefined : `${box.left}px`}
		style:top={sheet ? undefined : `${box.top}px`}
		style:width={sheet ? undefined : `${box.width}px`}
		style:height={sheet || box.folded ? undefined : `${box.height}px`}
		style:z-index={z}
		onkeydown={onKeydown}
		onpointerdown={raise}
	>
		<header
			class="head"
			role="toolbar"
			tabindex="-1"
			aria-label="Window bar: drag to move"
			onpointerdown={onPointerDown}
			onpointermove={onPointerMove}
			onpointerup={onPointerUp}
			onpointercancel={onPointerUp}
		>
			{#if Icon}<span class="icon"><Icon size={16} strokeWidth={2} aria-hidden="true" /></span>{/if}
			<div class="titles">
				<h2 id="{uid}-title">{title}</h2>
				{#if note}<p class="note">{note}</p>{/if}
			</div>
			{#if actions}<div class="actions">{@render actions()}</div>{/if}
			<button
				type="button"
				class="ctl"
				onclick={fold}
				aria-label={box.folded ? 'Unfold' : 'Fold to the title'}
			>
				{#if box.folded}<ChevronUp size={16} aria-hidden="true" />{:else}<ChevronDown
						size={16}
						aria-hidden="true"
					/>{/if}
			</button>
			<button type="button" class="ctl" onclick={close} aria-label="Close {title}">
				<X size={16} aria-hidden="true" />
			</button>
		</header>
		{#if !box.folded}
			<div class="body">{@render children()}</div>
		{/if}
	</div>
{/if}

<style>
	.window {
		position: fixed;
		display: flex;
		flex-direction: column;
		min-width: 18rem;
		max-width: calc(100vw - 16px);
		max-height: calc(100vh - 16px);
		border: 1px solid var(--vo-border-strong);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
		box-shadow: 0 12px 36px rgb(0 0 0 / 0.18);
		resize: both;
		overflow: hidden;
	}
	.window.folded {
		resize: horizontal;
	}
	.window:focus-visible {
		outline-offset: -2px;
	}
	.head {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		padding: 8px 8px 8px 14px;
		border-bottom: 1px solid var(--vo-border);
		background: var(--vo-surface-raised);
		cursor: move;
		user-select: none;
		touch-action: none;
	}
	.folded .head {
		border-bottom: 0;
	}
	.icon {
		display: inline-flex;
		color: var(--vo-accent-icon);
	}
	.titles {
		flex: 1;
		min-width: 0;
	}
	h2 {
		font-size: var(--vo-fs-body);
		font-weight: 650;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.note {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.ctl {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text-muted);
	}
	.ctl:hover {
		background: var(--vo-surface);
		color: var(--vo-text);
	}
	.body {
		flex: 1;
		min-height: 0;
		overflow: auto;
		scrollbar-width: thin;
	}
	/* Tablet and phone: a sheet along the bottom edge. */
	.window.sheet {
		left: 0;
		right: 0;
		bottom: 0;
		top: auto;
		width: auto;
		max-height: 70vh;
		border-radius: var(--vo-radius-panel) var(--vo-radius-panel) 0 0;
		resize: none;
	}
	.sheet .head {
		cursor: default;
	}
	@media print {
		.window {
			display: none;
		}
	}
</style>
