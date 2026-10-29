<script lang="ts">
	/**
	 * Furnace Status (SV-2.2, #294): the V-Board on the SPA. The schematic with zone callouts and zone status in the
	 * centre between the uptake and hearth strips, the section panels on either side, the operating state and the
	 * telemetry in the header, a zone drawer, and the trend of any reading above the board. Refresh at 10 s follows the
	 * `live.status` stream; 30 s, 1 min and 5 min poll. Rules (stale-after, live-age, refresh choices, intervals) come
	 * from `/api/status/settings`. State and timers live in `FurnaceStatus`; destroying the page stops them.
	 */
	import Activity from '@lucide/svelte/icons/activity';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Freshness from '#lib/components/Freshness.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import FurnaceSchematic from '#lib/furnace/FurnaceSchematic.svelte';
	import { FurnaceStatus } from '#lib/status/furnace-status.svelte.ts';
	import SectionPanel from '#lib/status/SectionPanel.svelte';
	import {
		calloutReadings,
		operatingBadge,
		PANEL_COLUMNS,
		parameterMap,
		SECTION,
		telemetryBadge,
		zoneReadings,
		zoneStatuses
	} from '#lib/status/status.ts';
	import TrendPanel from '#lib/status/TrendPanel.svelte';
	import ZoneDrawer from '#lib/status/ZoneDrawer.svelte';
	import { formatDuration } from '#lib/time/format.ts';

	const page = new FurnaceStatus();

	$effect(() => page.start());

	const snapshot = $derived(page.snapshot);
	const settings = $derived(page.settings);
	const byKey = $derived(snapshot ? parameterMap(snapshot) : new Map());
	const readings = $derived(snapshot ? calloutReadings(snapshot) : {});
	const zones = $derived(snapshot ? zoneReadings(snapshot.zones) : {});
	const zoneStatus = $derived(snapshot ? zoneStatuses(snapshot) : {});
	const linked = $derived(new Set(page.selectedZone ? (zones[page.selectedZone] ?? []) : []));
	const operating = $derived(snapshot ? operatingBadge(snapshot.operating_state) : null);
	const telemetry = $derived(snapshot ? telemetryBadge(snapshot.freshness) : null);
	const dataAt = $derived(
		snapshot?.freshness.data_at ? new Date(Date.parse(snapshot.freshness.data_at)) : null
	);
	const staleAfter = $derived(settings?.stale_after_minutes ?? snapshot?.freshness.stale_after_minutes);
	const liveAge = $derived(settings?.live_max_age_minutes ?? snapshot?.freshness.live_max_age_minutes);
	const refreshChoices = $derived(settings?.refresh ?? [{ label: '10 s', seconds: 10 }]);
	const refreshLabel = $derived(
		refreshChoices.find((c) => c.seconds === page.refreshSeconds)?.label ?? `${page.refreshSeconds} s`
	);
	/** Seconds since the last `live.status` beat, for the source note. */
	const beatAge = $derived(
		page.lastEventAt === null ? null : Math.round((page.clock - page.lastEventAt) / 1000)
	);
	const sourceNote = $derived.by(() => {
		if (!page.live) return `Polling every ${refreshLabel}`;
		if (beatAge !== null && beatAge <= 2.5 * page.livePeriod)
			return `Following the live stream, beat ${beatAge} s ago`;
		return beatAge === null
			? 'Polling until the live stream delivers'
			: `Polling: no beat for ${formatDuration(beatAge, 'seconds')}`;
	});
	const trendTitle = $derived(page.trendKey ? (byKey.get(page.trendKey)?.title ?? page.trendKey) : '');

	let drawerOpen = $state(false);

	function selectZone(name: string | null): void {
		page.selectedZone = name;
		drawerOpen = name !== null;
	}

	function openTrend(key: string): void {
		page.openTrend(key);
		drawerOpen = false;
	}
</script>

<div class="status">
	<div class="head">
		<Panel
			title="BF-02 blast furnace status"
			icon={Activity}
			note="Select a reading to open its trend; pick a zone under the diagram for its readings."
		>
			{#snippet actions()}
				{#if operating}
					<span class="badge">
						<span class="k">Furnace</span>
						<StatusChip kind={operating.kind} label={operating.label} title={operating.title} />
					</span>
				{/if}
				{#if telemetry}
					<span class="badge">
						<span class="k">Telemetry</span>
						<StatusChip kind={telemetry.kind} label={telemetry.label} title={telemetry.title} />
					</span>
				{/if}
				<label class="refresh">
					<span class="k">Refresh</span>
					<select
						value={page.refreshSeconds}
						onchange={(e) => page.setRefresh(Number(e.currentTarget.value))}
					>
						{#each refreshChoices as c (c.seconds)}
							<option value={c.seconds}>{c.label}</option>
						{/each}
					</select>
				</label>
			{/snippet}
			<div class="meta">
				{#if snapshot}
					<Freshness at={dataAt} warnAfterMin={liveAge} staleAfterMin={staleAfter} />
					<span class="muted">{snapshot.freshness.detail}</span>
					{#if settings}
						<span class="muted"
							>Rule: live within {settings.live_max_age_minutes} min, stale after {settings.stale_after_minutes}
							min</span
						>
					{/if}
					{#if snapshot.operating_state.detail}<span class="muted">{snapshot.operating_state.detail}</span
						>{/if}
				{/if}
				<span class="muted" title={snapshot ? `Snapshot source: ${snapshot.freshness.source}` : undefined}
					>{sourceNote}</span
				>
				{#if page.snapshotState.error && snapshot}
					<span class="warn">Refresh failed: {page.snapshotState.error} — showing the last snapshot.</span>
				{/if}
				{#if page.settingsState.error}
					<span class="warn">Display rules not loaded: {page.settingsState.error}</span>
				{/if}
			</div>
			{#if snapshot?.freshness.failed_sources?.length}
				<p class="alert" role="alert">
					Data source unavailable: {snapshot.freshness.failed_sources.join(', ')}. Affected parameters show
					Not available.
				</p>
			{/if}
		</Panel>
	</div>

	{#if page.trendKey}
		<div class="trend">
			<TrendPanel
				parameterKey={page.trendKey}
				title={trendTitle}
				trend={page.trend}
				load={page.trendState}
				{settings}
				interval={page.trendInterval}
				customStart={page.customStart}
				customEnd={page.customEnd}
				customError={page.trendIsCustom ? page.customError : null}
				{refreshLabel}
				oninterval={(id) => page.setTrendInterval(id)}
				oncustom={(s, e) => page.setCustomRange(s, e)}
				onretry={() => void page.loadTrend()}
				onclose={() => page.closeTrend()}
			/>
		</div>
	{/if}

	{#if !snapshot}
		<div class="board">
			{#if page.snapshotState.error}
				<ErrorState
					title="Furnace status not loaded"
					message={page.snapshotState.error}
					onretry={() => void page.loadSnapshot()}
					retrying={page.snapshotState.loading}
				/>
			{:else}
				<Panel title="Loading the furnace status" busy>
					<Skeleton variant="block" height="420px" label={null} />
				</Panel>
			{/if}
		</div>
	{:else}
		<div class="board">
			<div class="col left">
				{#each PANEL_COLUMNS[0] as section (section)}
					<SectionPanel
						{section}
						{snapshot}
						{linked}
						updatedAt={dataAt}
						warnAfterMin={liveAge}
						staleAfterMin={staleAfter}
						onopen={openTrend}
					/>
				{/each}
			</div>
			<div class="col centre">
				<SectionPanel
					section={SECTION.uptake}
					{snapshot}
					{linked}
					updatedAt={dataAt}
					warnAfterMin={liveAge}
					staleAfterMin={staleAfter}
					onopen={openTrend}
				/>
				<Panel
					title="Furnace"
					level={3}
					note="Callouts are related readings, not sensor positions."
					updatedAt={dataAt}
					warnAfterMin={liveAge}
					staleAfterMin={staleAfter}
				>
					<FurnaceSchematic
						{readings}
						zoneReadings={zones}
						{zoneStatus}
						selected={page.selectedZone}
						onselect={selectZone}
					/>
				</Panel>
				<SectionPanel
					section={SECTION.hearth}
					{snapshot}
					{linked}
					updatedAt={dataAt}
					warnAfterMin={liveAge}
					staleAfterMin={staleAfter}
					onopen={openTrend}
				/>
			</div>
			<div class="col right">
				{#each PANEL_COLUMNS[1] as section (section)}
					<SectionPanel
						{section}
						{snapshot}
						{linked}
						updatedAt={dataAt}
						warnAfterMin={liveAge}
						staleAfterMin={staleAfter}
						onopen={openTrend}
					/>
				{/each}
			</div>
		</div>
	{/if}
</div>

<ZoneDrawer
	bind:open={drawerOpen}
	zone={page.selectedZone}
	{snapshot}
	onopen={openTrend}
	onclose={() => (page.selectedZone = null)}
/>

<style>
	.status {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.k {
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.refresh {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.refresh select {
		min-height: 32px;
		padding: 0 8px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-4);
		font-size: var(--vo-fs-label);
	}
	.muted {
		color: var(--vo-text-muted);
	}
	.warn {
		color: var(--vo-warning);
	}
	.alert {
		margin-top: var(--vo-space-3);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-small);
		background: var(--vo-warning-bg);
		color: var(--vo-warning);
		font-size: var(--vo-fs-label);
		font-weight: 500;
	}
	/* One column on a tablet in portrait, two at 1366 × 768 (furnace beside the left panels), three at 1920 × 1080. */
	.board {
		display: grid;
		gap: var(--vo-space-4);
		grid-template-columns: minmax(0, 1fr);
		grid-template-areas: 'centre' 'left' 'right';
	}
	.col {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.left {
		grid-area: left;
	}
	.centre {
		grid-area: centre;
	}
	.right {
		grid-area: right;
	}
	@media (min-width: 1100px) {
		.board {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
			grid-template-areas: 'left centre' 'right centre';
		}
	}
	@media (min-width: 1600px) {
		.board {
			grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.3fr) minmax(0, 0.9fr);
			grid-template-areas: 'left centre right';
		}
	}
</style>
