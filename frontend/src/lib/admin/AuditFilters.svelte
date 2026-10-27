<!--
@component
Filters of the audit log: user, action and object type from the values in the log, object id, and the IST period.
A change applies at once (a menu choice, a period, the object id on Enter or when the field is left); the URL holds
the result, so navigation back and links to an object's history set the fields.
-->
<script lang="ts">
	import FunnelX from '@lucide/svelte/icons/funnel-x';
	import Button from '../components/Button.svelte';
	import DateRange from '../components/DateRange.svelte';
	import { rangeError, type RangePreset } from '../components/date-range.ts';
	import { istDay } from '../time/format.ts';
	import { hasFilters, menuOptions, type AuditFacets, type AuditView } from './audit.ts';

	interface Props {
		view: AuditView;
		/** Values present in the log; null until the first page has loaded. */
		facets: AuditFacets | null;
		onapply: (next: AuditView) => void;
	}

	let { view, facets, onapply }: Props = $props();

	const PRESETS: RangePreset[] = [
		{ label: 'Today', days: 1 },
		{ label: 'Last 7 days', days: 7 },
		{ label: 'Last 30 days', days: 30 },
		{ label: 'Last 365 days', days: 365 }
	];

	let actor = $state('');
	let action = $state('');
	let objectType = $state('');
	let objectId = $state('');
	let from = $state('');
	let to = $state('');

	// The URL is the source of truth: a navigation sets the fields again.
	$effect.pre(() => {
		actor = view.actor;
		action = view.action;
		objectType = view.objectType;
		objectId = view.objectId;
		from = view.from;
		to = view.to;
	});

	/** Apply the fields with `patch` (the value of the control that changed, read from its event). */
	function apply(patch: Partial<AuditView> = {}): void {
		const next: AuditView = {
			actor,
			action,
			objectType,
			objectId: objectId.trim(),
			from,
			to,
			...patch,
			page: 1
		};
		if (rangeError(next.from, next.to, { max: istDay(new Date()) }) !== null) return;
		onapply(next);
	}

	function clear(): void {
		onapply({ ...view, actor: '', action: '', objectType: '', objectId: '', page: 1 });
	}

	type MenuKey = 'actor' | 'action' | 'objectType';

	function choose(key: MenuKey, value: string): void {
		if (key === 'actor') apply({ actor: value });
		else if (key === 'action') apply({ action: value });
		else apply({ objectType: value });
	}

	function current(key: MenuKey): string {
		return key === 'actor' ? actor : key === 'action' ? action : objectType;
	}

	const menus = $derived([
		{ key: 'actor', label: 'User', all: 'All users', values: facets?.actors ?? [], value: view.actor },
		{ key: 'action', label: 'Action', all: 'All actions', values: facets?.actions ?? [], value: view.action },
		{
			key: 'objectType',
			label: 'Object type',
			all: 'All object types',
			values: facets?.object_types ?? [],
			value: view.objectType
		}
	] as const);
</script>

<form class="filters" role="search" aria-label="Audit log filters" onsubmit={(e) => e.preventDefault()}>
	<div class="fields">
		{#each menus as menu (menu.key)}
			<label class="field">
				<span>{menu.label}</span>
				<select value={current(menu.key)} onchange={(e) => choose(menu.key, e.currentTarget.value)}>
					<option value="">{menu.all}</option>
					{#each menuOptions(menu.values, menu.value) as option (option)}
						<option value={option}>{option}</option>
					{/each}
				</select>
			</label>
		{/each}
		<label class="field">
			<span>Object id</span>
			<input
				type="search"
				bind:value={objectId}
				placeholder="e.g. hopper_04"
				autocomplete="off"
				spellcheck="false"
				onchange={(e) => apply({ objectId: e.currentTarget.value.trim() })}
			/>
		</label>
	</div>
	<div class="period">
		<DateRange
			label="Period"
			bind:start={from}
			bind:end={to}
			presets={PRESETS}
			onchange={(range) => apply({ from: range.start, to: range.end })}
		/>
		{#if hasFilters(view)}
			<Button variant="ghost" size="sm" icon={FunnelX} onclick={clear}>Clear filters</Button>
		{/if}
	</div>
</form>

<style>
	.filters {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.fields {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
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
	input::placeholder {
		color: var(--vo-text-placeholder);
	}
	select:focus-visible,
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.period {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		justify-content: space-between;
		gap: var(--vo-space-3);
	}
</style>
