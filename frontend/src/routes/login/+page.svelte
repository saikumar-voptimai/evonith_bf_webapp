<!--
Sign-in page (SV-0.4, #273), in the V-OptimAIse product layout: product panel on the left, form on the right.
Accounts and roles are those of the Streamlit app (Neon identity.users).
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import ArrowRight from '@lucide/svelte/icons/arrow-right';
	import Eye from '@lucide/svelte/icons/eye';
	import EyeOff from '@lucide/svelte/icons/eye-off';
	import Flame from '@lucide/svelte/icons/flame';
	import Lock from '@lucide/svelte/icons/lock';
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
	import ShieldCheck from '@lucide/svelte/icons/shield-check';
	import { ApiError } from '#lib/api/client.ts';
	import { describeError } from '#lib/api/describe.ts';
	import ShiftBand from '#lib/components/shell/ShiftBand.svelte';
	import { safeNext } from '#lib/session/guard.ts';
	import { session } from '#lib/session/session.svelte.ts';
	import { clock } from '#lib/time/clock.svelte.ts';
	import { istDayMonth } from '#lib/time/format.ts';
	import { istClock } from '#lib/time/shift.ts';

	let username = $state('');
	let password = $state('');
	let reveal = $state(false);
	let busy = $state(false);
	let failure = $state<string | null>(null);

	$effect(() => clock.start());

	// BF-02 lining profile (setting_ds_dv.yml, metres): hearth to throat, mirrored about the centre line.
	const PROFILE: [number, number][] = [
		[2.8, 0],
		[2.8, 4.374],
		[2.8, 6.795],
		[3.15, 8.335],
		[3.15, 11.29],
		[3.65, 14.39],
		[3.65, 15.89],
		[2.898, 20]
	];
	const outline = (scale: number): string => {
		const side = PROFILE.map(([x, y]) => [x * scale, y * scale] as const);
		const left = side.map(([x, y]) => `${(-x).toFixed(3)},${(-y).toFixed(3)}`);
		const right = [...side].reverse().map(([x, y]) => `${x.toFixed(3)},${(-y).toFixed(3)}`);
		return `M ${left.join(' L ')} L ${right.join(' L ')} Z`;
	};
	/** Two bands around the profile (stroke widths in metres), as the arcs of the V-OptimAIse sign-in page. */
	const BANDS = [
		{ width: 4.2, colour: 'var(--vo-chrome-band)' },
		{ width: 2.1, colour: 'var(--vo-chrome-band-strong)' }
	];

	function message(e: unknown): string {
		if (e instanceof ApiError && (e.status === 401 || e.status === 429 || e.status === 503) && e.detail)
			return e.detail;
		if (e instanceof TypeError)
			return 'The server did not answer. Check the network connection, then try again.';
		return describeError(e).message;
	}

	async function submit(event: SubmitEvent): Promise<void> {
		event.preventDefault();
		if (busy) return;
		if (!username.trim() || !password) {
			failure = 'Enter the user name and the password.';
			return;
		}
		busy = true;
		failure = null;
		try {
			await session.login(username.trim(), password);
			password = '';
			// safeNext keeps the target on this origin; resolve() cannot take a runtime path.
			// eslint-disable-next-line svelte/no-navigation-without-resolve
			await goto(safeNext(page.url.searchParams.get('next')), { replaceState: true });
		} catch (e) {
			failure = message(e);
			password = '';
		} finally {
			busy = false;
		}
	}
</script>

<div class="login">
	<section class="product" aria-label="V-OptimAIse BF-02 console">
		<div class="brand">
			<span class="mark" aria-hidden="true"><Flame size={20} strokeWidth={2.25} /></span>
			<span class="word">V-OptimAIse</span>
		</div>

		<div class="pitch">
			<p class="eyebrow">BF-02 digital twin, Evonith Steel</p>
			<p class="headline">
				<span>Coke rate, HMT and blend</span>
				<span class="accent">from one furnace model.</span>
			</p>
			<p class="lede">
				TRACE, FORGE and STRATA evaluated every hour on plant data, with the lowest-cost blend for the next
				shift.
			</p>
		</div>

		<svg
			class="profile"
			viewBox="-6.2 -22.6 12.4 14.6"
			aria-hidden="true"
			preserveAspectRatio="xMidYMin meet"
		>
			{#each BANDS as band (band.width)}
				<path class="band" d={outline(1)} style:stroke={band.colour} style:stroke-width={band.width} />
			{/each}
			<path class="inside" d={outline(1)} />
		</svg>

		<p class="foot">
			<Lock size={15} strokeWidth={2} />
			<span class="foot-text"
				><span>Self-hosted on the plant network ·</span> <span>Role-based access ·</span>
				<span>by V-OptimAIse</span></span
			>
		</p>
	</section>

	<main class="access">
		<div class="now">
			<ShiftBand now={clock.now} compact />
			<time class="clock" datetime={clock.now.toISOString()}>
				<span class="day">{istDayMonth(clock.now)}</span>
				<span class="hm">{istClock(clock.now)}</span>
				<span class="tz">IST</span>
			</time>
		</div>

		<div class="column">
			<p class="eyebrow light">Welcome to the BF-02 console</p>
			<h1>Sign in</h1>
			<p class="intro">Use your Streamlit account; the roles are the same.</p>

			<form onsubmit={submit} novalidate aria-describedby={failure ? 'login-failure' : undefined}>
				<label class="field">
					<span>User name</span>
					<!-- svelte-ignore a11y_autofocus -->
					<input
						name="username"
						autocomplete="username"
						autocapitalize="none"
						spellcheck="false"
						placeholder="e.g. operator.b"
						bind:value={username}
						autofocus
						required
					/>
				</label>
				<div class="field">
					<label for="login-password">Password</label>
					<div class="password">
						<input
							id="login-password"
							name="password"
							type={reveal ? 'text' : 'password'}
							autocomplete="current-password"
							bind:value={password}
							required
						/>
						<button
							type="button"
							class="reveal"
							aria-label={reveal ? 'Hide password' : 'Show password'}
							aria-pressed={reveal}
							onclick={() => (reveal = !reveal)}
						>
							{#if reveal}<EyeOff size={18} strokeWidth={1.75} />{:else}<Eye
									size={18}
									strokeWidth={1.75}
								/>{/if}
						</button>
					</div>
				</div>
				<button type="submit" class="submit" aria-disabled={busy || undefined}>
					<span>{busy ? 'Signing in' : 'Sign in to the console'}</span>
					<ArrowRight size={18} strokeWidth={2} />
				</button>
				<p id="login-failure" class="failure" role="alert">
					{#if failure}<OctagonAlert size={15} strokeWidth={2.25} /><span>{failure}</span>{/if}
				</p>
			</form>

			<p class="note">
				<ShieldCheck size={18} strokeWidth={1.75} />
				<span
					>Ask your administrator for an account or a different role. The Streamlit pages stay available under
					<a href="/legacy/" rel="external">/legacy</a>.</span
				>
			</p>
		</div>
	</main>
</div>

<style>
	.login {
		display: grid;
		grid-template-columns: minmax(0, 52fr) minmax(0, 48fr);
		min-height: 100vh;
		min-height: 100dvh;
	}

	/* Product panel */
	.product {
		position: relative;
		display: flex;
		flex-direction: column;
		padding: 48px 64px 40px;
		overflow: hidden;
		background: var(--vo-chrome);
		color: var(--vo-chrome-text);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.mark {
		display: inline-grid;
		place-items: center;
		width: 38px;
		height: 38px;
		border-radius: 9px;
		background: var(--vo-accent);
		color: var(--vo-chrome);
	}
	.word {
		font-size: 1.375rem;
		font-weight: 650;
		letter-spacing: -0.01em;
		color: var(--vo-chrome-strong);
	}
	.pitch {
		position: relative;
		z-index: 1;
		margin: auto 0;
		padding: 48px 0;
	}
	.eyebrow {
		font-size: 0.78rem;
		font-weight: 650;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: var(--vo-chrome-eyebrow);
	}
	.headline {
		display: flex;
		flex-direction: column;
		margin-top: 22px;
		font-size: clamp(2.25rem, 3.3vw, 3.75rem);
		font-weight: 550;
		line-height: 1.08;
		letter-spacing: -0.03em;
		color: var(--vo-chrome-display);
	}
	.accent {
		color: var(--vo-accent);
	}
	.lede {
		margin-top: 28px;
		max-width: 40ch;
		font-size: 1.0625rem;
		line-height: 1.65;
		color: var(--vo-chrome-lede);
		text-wrap: pretty;
	}
	/* Lining profile with two bands, rising from the lower right edge below the headline. */
	.profile {
		position: absolute;
		right: -3%;
		bottom: 0;
		height: 40%;
		aspect-ratio: 12.4 / 14.6;
		pointer-events: none;
	}
	.band {
		fill: none;
		stroke-linejoin: round;
	}
	.inside {
		fill: var(--vo-chrome);
	}
	.foot {
		position: relative;
		z-index: 1;
		display: flex;
		align-items: flex-start;
		gap: 10px;
		font-size: var(--vo-fs-label);
		color: var(--vo-chrome-muted);
	}
	.foot :global(svg) {
		flex: none;
		margin-top: 2px;
	}
	/* Line breaks only between the parts. */
	.foot-text > span {
		white-space: nowrap;
	}

	/* Form panel */
	.access {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 96px 48px 48px;
		background: var(--vo-bg);
	}
	.now {
		position: absolute;
		top: 24px;
		right: 32px;
		display: flex;
		align-items: center;
		gap: var(--vo-space-5);
	}
	.clock {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		white-space: nowrap;
	}
	.day,
	.tz {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.hm {
		font-size: 1.0625rem;
		font-weight: 650;
	}
	.column {
		width: min(440px, 100%);
	}
	.eyebrow.light {
		color: var(--vo-text-muted);
	}
	h1 {
		margin-top: 14px;
		font-size: 2.25rem;
		font-weight: 650;
		letter-spacing: -0.03em;
	}
	.intro {
		margin-top: 12px;
		font-size: 1rem;
		color: var(--vo-text-muted);
		text-wrap: pretty;
	}
	form {
		display: flex;
		flex-direction: column;
		gap: 20px;
		margin-top: 36px;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.field > span,
	.field > label {
		font-size: var(--vo-fs-text);
		font-weight: 550;
	}
	input {
		width: 100%;
		min-height: 48px;
		padding: 0 14px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		font-size: var(--vo-fs-text);
	}
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.password {
		position: relative;
	}
	.password input {
		padding-right: 48px;
	}
	.reveal {
		position: absolute;
		top: 4px;
		right: 4px;
		display: inline-grid;
		place-items: center;
		width: 40px;
		height: 40px;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text-muted);
	}
	.reveal:hover {
		background: var(--vo-surface-raised);
		color: var(--vo-text);
	}
	.submit {
		display: flex;
		align-items: center;
		justify-content: space-between;
		min-height: 50px;
		margin-top: 8px;
		padding: 0 18px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: var(--vo-accent);
		color: var(--vo-text-on-accent);
		font-size: var(--vo-fs-text);
		font-weight: 650;
	}
	.submit:hover:not([aria-disabled='true']) {
		background: var(--vo-accent-hover);
	}
	.submit[aria-disabled='true'] {
		cursor: progress;
		opacity: 0.75;
	}
	.failure {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		min-height: 20px;
		margin-top: -6px;
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
		font-weight: 500;
	}
	.failure:empty {
		display: none;
	}
	.failure :global(svg) {
		flex: none;
		margin-top: 1px;
	}
	.note {
		display: flex;
		align-items: flex-start;
		gap: 12px;
		margin-top: 28px;
		padding-top: 24px;
		border-top: 1px solid var(--vo-border);
		font-size: var(--vo-fs-label);
		line-height: 1.6;
		color: var(--vo-text-muted);
		text-wrap: pretty;
	}
	.note :global(svg) {
		flex: none;
		margin-top: 1px;
	}

	@media (max-width: 1279px) {
		.product {
			padding: 40px 44px 32px;
		}
	}
	@media (max-width: 1199px) {
		.profile {
			height: 30%;
		}
	}
	@media (max-width: 1023px) {
		.login {
			grid-template-columns: minmax(0, 1fr);
			grid-template-rows: auto 1fr;
		}
		.product {
			padding: 24px 24px 28px;
		}
		.pitch {
			margin: 0;
			padding: 28px 0 8px;
		}
		.headline {
			font-size: 2rem;
		}
		.lede,
		.profile,
		.foot {
			display: none;
		}
		.access {
			align-items: flex-start;
			padding: 88px 24px 40px;
		}
	}
	@media (max-width: 479px) {
		.now .clock .day {
			display: none;
		}
	}
</style>
