<!--
Sign-in page (SV-0.4, #273). Accounts and roles are those of the Streamlit app (Neon identity.users).
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import LogIn from '@lucide/svelte/icons/log-in';
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
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
	let busy = $state(false);
	let failure = $state<string | null>(null);

	$effect(() => clock.start());

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
	<header class="bar">
		<ShiftBand now={clock.now} />
		<time class="clock" datetime={clock.now.toISOString()}>
			<span class="day">{istDayMonth(clock.now)}</span>
			<span class="hm">{istClock(clock.now)}</span>
			<span class="tz">IST</span>
		</time>
	</header>

	<main class="body">
		<div class="column">
			<div class="brand">
				<p class="word">V-OptimAIse</p>
				<p class="plant">BF-02 operator console, Evonith Steel</p>
			</div>

			<form
				class="card"
				onsubmit={submit}
				novalidate
				aria-describedby={failure ? 'login-failure' : undefined}
			>
				<h1>Sign in</h1>
				<label class="field">
					<span>User name</span>
					<!-- svelte-ignore a11y_autofocus -->
					<input
						name="username"
						autocomplete="username"
						autocapitalize="none"
						spellcheck="false"
						bind:value={username}
						autofocus
						required
					/>
				</label>
				<label class="field">
					<span>Password</span>
					<input
						name="password"
						type="password"
						autocomplete="current-password"
						bind:value={password}
						required
					/>
				</label>
				<button type="submit" class="submit" aria-disabled={busy || undefined}>
					<LogIn size={17} strokeWidth={2.25} />
					<span>{busy ? 'Signing in' : 'Sign in'}</span>
				</button>
				<p id="login-failure" class="failure" role="alert">
					{#if failure}<OctagonAlert size={15} strokeWidth={2.25} /><span>{failure}</span>{/if}
				</p>
			</form>

			<p class="note">
				Use the account of the Streamlit app; roles are the same.
				<a href="/legacy/" rel="external">Open the Streamlit pages</a>
			</p>
		</div>
	</main>
</div>

<style>
	.login {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
		min-height: 100dvh;
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-4);
		min-height: var(--vo-topbar-height);
		padding: 0 var(--vo-gutter);
		background: var(--vo-black);
		border-bottom: 1px solid var(--vo-border);
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
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.hm {
		font-size: var(--vo-fs-title);
		font-weight: 600;
	}
	.body {
		flex: 1;
		display: flex;
		align-items: center;
		padding: var(--vo-space-7) var(--vo-gutter);
	}
	.column {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-5);
		width: min(400px, 100%);
		margin-left: max(0px, calc(50% - 360px));
	}
	.word {
		font-size: var(--vo-fs-heading);
		font-weight: 700;
	}
	.plant {
		margin-top: 2px;
		color: var(--vo-text-muted);
	}
	.card {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		padding: var(--vo-space-5);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
	}
	h1 {
		font-size: var(--vo-fs-title);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.field span {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	input {
		min-height: var(--vo-target);
		padding: 0 12px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		font-size: var(--vo-fs-body);
	}
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-yellow);
	}
	.submit {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		min-height: var(--vo-target);
		margin-top: var(--vo-space-1);
		border: 0;
		border-radius: var(--vo-radius-control);
		background: var(--vo-yellow-strong);
		color: var(--vo-text-on-accent);
		font-size: var(--vo-fs-body);
		font-weight: 600;
	}
	.submit:hover:not([aria-disabled='true']) {
		background: var(--vo-yellow);
	}
	.submit[aria-disabled='true'] {
		cursor: progress;
		opacity: 0.7;
	}
	.failure {
		display: flex;
		align-items: flex-start;
		gap: 6px;
		min-height: 20px;
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
		font-weight: 500;
	}
	.failure :global(svg) {
		flex: none;
		margin-top: 1px;
	}
	.note {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.note a {
		margin-left: 4px;
		font-weight: 600;
	}
	@media (max-width: 767px) {
		.clock .day {
			display: none;
		}
	}
</style>
