<script lang="ts">
	/**
	 * Live Twin (SV-1.9, #289): the furnace now, the coke rate by model, shaft efficiency, thermal state, the burden
	 * column, the optimiser plan, alarms and the data age of every source, on one page that stays open through a shift.
	 * State and subscriptions live in `LiveTwin`; destroying the page stops every subscription and timer.
	 */
	import AlarmsPanel from '#lib/twin/AlarmsPanel.svelte';
	import CokePanel from '#lib/twin/CokePanel.svelte';
	import ColumnPanel from '#lib/twin/ColumnPanel.svelte';
	import FreshnessBar from '#lib/twin/FreshnessBar.svelte';
	import OmegaPanel from '#lib/twin/OmegaPanel.svelte';
	import PlanCard from '#lib/twin/PlanCard.svelte';
	import ThermalPanel from '#lib/twin/ThermalPanel.svelte';
	import TwinKpiStrip from '#lib/twin/TwinKpiStrip.svelte';
	import WindowDrawer from '#lib/twin/WindowDrawer.svelte';
	import { nearestWindow, stripState } from '#lib/twin/live-twin.ts';
	import { LiveTwin } from '#lib/twin/live-twin.svelte.ts';
	import { jobErrorText, JobTracker } from '#lib/live/jobs.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';

	const twin = new LiveTwin();
	const tracker = new JobTracker();

	$effect(() => {
		const stop = twin.start();
		return () => {
			stop();
			tracker.stop();
		};
	});

	const role = $derived(session.user?.role ?? null);
	const supervisor = $derived(hasRole(role, 'supervisor'));
	const latest = $derived(twin.twinNow?.snapshot ?? null);
	const latestAt = $derived(latest ? new Date(Date.parse(latest.window_end)) : null);
	const statusStale = $derived(
		stripState(twin.status, twin.statusReceivedAt, twin.startedAt, twin.clock) === 'stale'
	);
	const ends = $derived(twin.rows.map((r) => r.window_end));

	let drawerOpen = $state(false);
	let drawerEnd = $state<string | null>(null);

	function pick(t: number): void {
		const end = nearestWindow(twin.rows, t);
		if (!end) return;
		drawerEnd = end;
		drawerOpen = true;
	}

	async function runNow(): Promise<void> {
		tracker.error = null;
		try {
			const { job } = await twin.startRun();
			tracker.follow(job);
		} catch (e) {
			tracker.error = jobErrorText(e);
		}
	}
</script>

<div class="twin">
	<div class="kpi">
		<TwinKpiStrip
			status={twin.status}
			receivedAt={twin.statusReceivedAt}
			startedAt={twin.startedAt}
			now={twin.clock}
		/>
	</div>
	<div class="fresh">
		<FreshnessBar twinNow={twin.twinNow} status={twin.status} zone={twin.nowState} now={twin.clock} />
	</div>
	<div class="coke">
		<CokePanel
			rows={twin.rows}
			gaps={twin.twinNow?.gaps}
			range={twin.range}
			zone={twin.seriesState}
			updatedAt={latestAt}
			onrange={(r) => void twin.setRange(r)}
			onretry={() => void twin.loadSeries()}
			onpick={pick}
		/>
	</div>
	<div class="omega">
		<OmegaPanel
			rows={twin.rows}
			alarms={twin.alarms}
			range={twin.range}
			zone={twin.seriesState}
			updatedAt={latestAt}
			onretry={() => void twin.loadSeries()}
			onpick={pick}
		/>
	</div>
	<div class="column">
		<ColumnPanel
			transit={twin.transit}
			status={twin.status}
			{statusStale}
			zone={twin.transitState}
			now={twin.clock}
			onretry={() => void twin.loadTransit()}
		/>
	</div>
	<div class="thermal">
		<ThermalPanel
			taps={twin.taps}
			si={twin.si}
			snapshot={latest}
			forecast={twin.siForecast}
			forecastAt={twin.siForecastAt}
			zone={twin.hotMetalState}
			onretry={() => void twin.loadHotMetal()}
		/>
	</div>
	<div class="plan">
		<PlanCard
			run={twin.plan}
			zone={twin.planState}
			{tracker}
			canRun={supervisor}
			canDecide={supervisor}
			onrun={() => void runNow()}
			ondecide={(d) => twin.decide(d)}
			onretry={() => void twin.loadPlan()}
		/>
	</div>
	<div class="alarms">
		<AlarmsPanel alarms={twin.alarms} zone={twin.alarmState} onretry={() => void twin.loadAlarms()} />
	</div>
</div>

<WindowDrawer bind:open={drawerOpen} bind:end={drawerEnd} load={(e) => twin.window(e)} {ends} />

<style>
	/* One column on a tablet in portrait, two at 1366 × 768, three at 1920 × 1080 (furnace column on the right). */
	.twin {
		display: grid;
		gap: var(--vo-space-4);
		grid-template-columns: minmax(0, 1fr);
		grid-template-areas: 'kpi' 'fresh' 'coke' 'omega' 'column' 'thermal' 'plan' 'alarms';
	}
	.twin > div {
		min-width: 0;
	}
	.kpi {
		grid-area: kpi;
	}
	.fresh {
		grid-area: fresh;
	}
	.coke {
		grid-area: coke;
	}
	.omega {
		grid-area: omega;
	}
	.column {
		grid-area: column;
	}
	.thermal {
		grid-area: thermal;
	}
	.plan {
		grid-area: plan;
	}
	.alarms {
		grid-area: alarms;
	}
	@media (min-width: 1100px) {
		.twin {
			grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
			grid-template-areas:
				'kpi kpi'
				'fresh fresh'
				'coke column'
				'omega column'
				'thermal plan'
				'alarms plan';
		}
	}
	@media (min-width: 1600px) {
		.twin {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 0.85fr);
			grid-template-areas:
				'kpi kpi kpi'
				'fresh fresh fresh'
				'coke omega column'
				'thermal plan column'
				'alarms alarms column';
		}
	}
</style>
