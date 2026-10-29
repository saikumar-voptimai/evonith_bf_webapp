<!--
@component
User accounts (`identity.users`, `identity.user_roles`) and the register form of the Streamlit tool: user name and
password (trimmed, both required), role. The account is written as the Streamlit tool writes it, with its audit row.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import UserPlus from '@lucide/svelte/icons/user-plus';
	import Users from '@lucide/svelte/icons/users';
	import { untrack } from 'svelte';
	import { api } from '../api/client.ts';
	import { describeError } from '../api/describe.ts';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { toasts } from '../components/toast.svelte.ts';
	import { istStamp } from './audit.ts';
	import {
		AUDIT_QUERY,
		registrationProblem,
		ROLE_LABEL,
		roleLabel,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		type RoleName,
		type UsersView,
		type UserView
	} from './plant.ts';

	let view = $state.raw<UsersView | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let updatedAt = $state<Date | null>(null);
	let request = 0;

	let username = $state('');
	let password = $state('');
	let role = $state<RoleName>('user');
	let touched = $state(false);
	let saving = $state(false);

	async function load(): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data } = await api.GET('/api/admin/users');
			if (id !== request) return;
			view = data ?? null;
			error = null;
			updatedAt = new Date();
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		untrack(() => void load());
	});

	const problem = $derived(registrationProblem(username, password));
	const roles = $derived<RoleName[]>(view?.roles ?? ['user', 'supervisor', 'admin']);
	const users = $derived(view?.users ?? []);
	const active = $derived(users.filter((u) => u.is_active).length);

	async function register(): Promise<void> {
		touched = true;
		if (problem || saving) return;
		saving = true;
		try {
			const { data } = await api.POST('/api/admin/users', { body: { username, password, role } });
			toasts.ok('User registered', `${data?.user.username ?? username.trim()} · ${ROLE_LABEL[role]}`);
			username = '';
			password = '';
			role = 'user';
			touched = false;
			void load();
		} catch (e) {
			toasts.critical('User not registered', describeError(e).message);
		} finally {
			saving = false;
		}
	}

	const stamp = (iso: string): string => istStamp(new Date(iso));

	const columns: Column<UserView>[] = [
		{ key: 'username', label: 'User name' },
		{ key: 'role', label: 'Role', format: (u) => roleLabel(u.role) },
		{
			key: 'roles',
			label: 'Role assignments',
			sortable: false,
			format: (u) => (u.roles.length ? u.roles.map(roleLabel).join(', ') : 'None')
		},
		{ key: 'is_active', label: 'Status', sortable: false },
		{
			key: 'created_at',
			label: 'Created',
			unit: 'IST',
			value: (u) => u.created_at,
			format: (u) => stamp(u.created_at)
		},
		{
			key: 'updated_at',
			label: 'Changed',
			unit: 'IST',
			value: (u) => u.updated_at,
			format: (u) => stamp(u.updated_at)
		}
	];
</script>

{#snippet statusCell(u: UserView)}
	{#if u.is_active}
		<StatusChip kind="ok" label="Active" size="sm" />
	{:else}
		<StatusChip kind="stale" label="Inactive" size="sm" title="Cannot sign in" />
	{/if}
{/snippet}

<div class="tab">
	<Panel
		title="User accounts"
		icon={Users}
		note="Every account of the plant database, active or not, by user name. Roles: user, supervisor, administrator."
		{updatedAt}
		warnAfterMin={VIEW_WARN_MIN}
		staleAfterMin={VIEW_STALE_MIN}
		busy={loading && !!view}
		flush
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void load()}
				disabled={loading}
				aria-label="Refresh the accounts">Refresh</Button
			>
		{/snippet}
		{#if error}
			<div class="pad">
				<ErrorState
					{error}
					title={view ? 'Accounts not refreshed' : 'Accounts not loaded'}
					message={view ? 'The list below is from the last successful load.' : undefined}
					onretry={() => void load()}
					retrying={loading}
					compact
				/>
			</div>
		{/if}
		{#if view || loading}
			<DataTable
				caption="User accounts"
				rows={users}
				{columns}
				rowKey={(u) => u.id}
				cells={{ is_active: statusCell }}
				rowClass={(u) => (u.is_active ? undefined : 'muted')}
				loading={loading && !view}
				maxHeight="28rem"
				emptyTitle="No accounts"
				emptyMessage="Register the first account below."
			/>
		{/if}
		{#snippet footer()}
			<span class="vo-num hint">
				{users.length} accounts · {active} active · every registration is in the
				<a href={resolve(`/admin/audit?${AUDIT_QUERY.users}`)}>audit log</a>
			</span>
		{/snippet}
	</Panel>

	<Panel
		title="Register a user"
		icon={UserPlus}
		note="As the Welcome page form: leading and trailing blanks are removed, the password is stored as its SHA-256 digest."
	>
		<form class="register" onsubmit={(e) => (e.preventDefault(), void register())}>
			<label class="field">
				<span>User name</span>
				<input bind:value={username} maxlength="128" autocomplete="off" required />
			</label>
			<label class="field">
				<span>Password</span>
				<input type="password" bind:value={password} maxlength="256" autocomplete="new-password" required />
			</label>
			<label class="field">
				<span>Role</span>
				<select bind:value={role}>
					{#each roles as r (r)}
						<option value={r}>{ROLE_LABEL[r]}</option>
					{/each}
				</select>
			</label>
			<Button variant="primary" type="submit" icon={UserPlus} disabled={saving || !view}>Register user</Button
			>
		</form>
		{#if touched && problem}
			<p class="problem" role="alert">{problem}</p>
		{/if}
	</Panel>
</div>

<style>
	.tab {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.pad {
		padding: var(--vo-space-4) var(--vo-space-5);
	}
	.hint {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.register {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
	}
	.field {
		display: flex;
		flex: 1 1 200px;
		flex-direction: column;
		gap: 6px;
		min-width: min(200px, 100%);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	select,
	input {
		width: 100%;
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	select:focus-visible,
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.problem {
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-critical);
	}
</style>
