/**
 * Navigation of the operator console: page routes, labels, icons and the issue that builds each page.
 * The rail, the page titles and the planned-page placeholders all read this list.
 */
import type { LucideIcon } from '@lucide/svelte';
import type { RouteId } from '$app/types';
import Activity from '@lucide/svelte/icons/activity';
import Blend from '@lucide/svelte/icons/blend';
import BookOpen from '@lucide/svelte/icons/book-open';
import Brain from '@lucide/svelte/icons/brain';
import ChartLine from '@lucide/svelte/icons/chart-line';
import ClipboardList from '@lucide/svelte/icons/clipboard-list';
import Flame from '@lucide/svelte/icons/flame';
import FlaskConical from '@lucide/svelte/icons/flask-conical';
import Gauge from '@lucide/svelte/icons/gauge';
import Layers from '@lucide/svelte/icons/layers';
import Scale from '@lucide/svelte/icons/scale';
import ScrollText from '@lucide/svelte/icons/scroll-text';
import ServerCog from '@lucide/svelte/icons/server-cog';
import Shield from '@lucide/svelte/icons/shield';
import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';
import Thermometer from '@lucide/svelte/icons/thermometer';
import Users from '@lucide/svelte/icons/users';
import Wind from '@lucide/svelte/icons/wind';
import type { LegacyPage } from './legacy.ts';
import { hasRole, type Role } from './session/session.svelte.ts';

export interface NavItem {
	href: RouteId;
	label: string;
	/** Page title in the page header and the browser tab. */
	title: string;
	icon: LucideIcon;
	/** One line under the page title. */
	subtitle: string;
	/** What the page shows, in the operator's terms. */
	summary: string;
	/** GitHub issue that builds the page. */
	issue: number;
	/** Streamlit page that covers the same ground until the Svelte page ships. */
	legacy?: LegacyPage;
	/** Views of a section, listed in its sub-navigation (`SectionNav`) rather than in the rail. */
	children?: NavItem[];
}

export interface NavGroup {
	label: string;
	items: NavItem[];
	/** Minimum role that sees the group. */
	role?: Role;
}

export const ISSUE_URL = 'https://github.com/saikumar-voptimai/evonith_bf_webapp/issues/';

/**
 * Views of the admin space (SV-4.1, #304), in the order of its sub-navigation. Each view has its own route,
 * `routes/admin/<view>/+page.svelte`; a view that is not built yet shows the planned-page placeholder of its issue.
 */
export const ADMIN_VIEWS: NavItem[] = [
	{
		href: '/admin/facts',
		label: 'Fact register',
		title: 'Fact register',
		subtitle: 'Coefficients, plant bands and default limits, changed with a second review.',
		icon: BookOpen,
		summary:
			'The fact register behind every model and limit: propose a change to a coefficient, band or default limit, have a second administrator review it, and compare the versions.',
		issue: 305
	},
	{
		href: '/admin/what-if',
		label: 'What-if',
		title: 'What-if analysis',
		subtitle: 'Coke rate and cost of a changed blend, blast or limit at the calibration.',
		icon: FlaskConical,
		summary:
			'Change the blend, the blast or a limit and see the coke rate, the cost and the certainty the models give at the current calibration.',
		issue: 306
	},
	{
		href: '/admin/burnt-vs-charged',
		label: 'Burnt vs charged',
		title: 'Burnt vs charged coke',
		subtitle: 'Agreement from a shift to a month, what averages out, lag, mass closure and leak water.',
		icon: Scale,
		summary:
			'Burnt coke from the oxygen balance against the coke charged, per averaging window from a shift to a month: concordance and Bland–Altman limits, the variance split, the lag, the raw mass closure and the leak-water check.',
		issue: 307,
		legacy: 'Burden & Fuel Optimiser'
	},
	{
		href: '/admin/validation',
		label: 'Validation',
		title: 'Validation and STRATA-Learn',
		subtitle: 'Out-of-sample error of the coke models and the STRATA-Learn shadow run.',
		icon: ChartLine,
		summary:
			'Out-of-sample error of TRACE, FORGE and STRATA by period, and STRATA-Learn in shadow mode: scale estimates, champion and challenger, drift alarm and register proposals.',
		issue: 308,
		legacy: 'Burden & Fuel Optimiser'
	},
	{
		href: '/admin/calibration',
		label: 'Calibration',
		title: 'Calibration',
		subtitle: 'The 14-day calibration of the balance, its stable shifts and the nightly run.',
		icon: Gauge,
		summary:
			'The 14-day calibration of the two-zone balance: the stable shifts it uses, its parameters, the nightly recalibration and the STRATA-Flow Jacobian of each plan.',
		issue: 309,
		legacy: 'Coke Energy Model'
	},
	{
		href: '/admin/users',
		label: 'Users & hoppers',
		title: 'Users, hoppers and burden',
		subtitle: 'User accounts and roles, hopper materials and the burden distribution.',
		icon: Users,
		summary:
			'User accounts and their roles, the material in each hopper and the burden distribution (rings and angles); every change is written to the audit log.',
		issue: 310,
		legacy: 'Welcome'
	},
	{
		href: '/admin/jobs',
		label: 'Jobs & health',
		title: 'Jobs and health',
		subtitle: 'Job history with cancel and retry, run times, source data age and host health.',
		icon: ServerCog,
		summary:
			'Background jobs and their history with audited cancel and retry, run times (p50 and p95) and missed slots, the data age of every source, the services and the host’s CPU, RAM, disk and temperatures.',
		issue: 311
	},
	{
		href: '/admin/audit',
		label: 'Audit log',
		title: 'Audit log',
		subtitle: 'Who changed what and when, with the values before and after.',
		icon: ScrollText,
		summary:
			'Every admin write and plan decision: who did it, on which object and when, with the values before and after the change.',
		issue: 304
	}
];

export const NAV: NavGroup[] = [
	{
		label: 'Digital twin',
		items: [
			{
				href: '/',
				label: 'Live Twin',
				title: 'Live Twin',
				subtitle: 'Coke rate by model, HMT forecast and the burden in the furnace, every hour.',
				icon: Activity,
				summary:
					'Coke rate from TRACE, FORGE and STRATA side by side, the HMT forecast, implied shaft efficiency and the burden in the furnace, refreshed every hour.',
				issue: 289,
				legacy: 'Coke Energy Model'
			},
			{
				href: '/trace',
				label: 'TRACE',
				title: 'TRACE',
				subtitle: 'Burnt coke from the gas-side oxygen balance.',
				icon: Wind,
				summary:
					'Burnt coke per shift and per hour from the gas-side oxygen balance, with its correction and record.',
				issue: 290,
				legacy: 'Coke Energy Model'
			},
			{
				href: '/forge',
				label: 'FORGE',
				title: 'FORGE',
				subtitle: 'Modelled coke from the two-zone Rist energy balance.',
				icon: Flame,
				summary:
					'Modelled coke from the two-zone Rist energy balance at the calibration, and the terms that drive it.',
				issue: 291,
				legacy: 'Coke Energy Model'
			},
			{
				href: '/strata',
				label: 'STRATA',
				title: 'STRATA',
				subtitle: 'Shaft efficiency from the STRATA-Flow ring model.',
				icon: Layers,
				summary:
					'Shaft efficiency from the STRATA-Flow ring model, its uncertainty, and the certainty tier of each blend.',
				issue: 292,
				legacy: 'Burden & Fuel Optimiser'
			}
		]
	},
	{
		label: 'Furnace',
		items: [
			{
				href: '/status',
				label: 'Furnace status',
				title: 'Furnace status',
				subtitle: 'Process readings, wall temperatures and stave heat loads.',
				icon: Thermometer,
				summary: 'Current process readings, wall temperatures and stave heat loads on the furnace profile.',
				issue: 294,
				legacy: 'V-Board'
			},
			{
				href: '/reports',
				label: 'Shift reports',
				title: 'Shift reports',
				subtitle: 'Shift handover reports: KPI status, burden, temperatures, twin and narrative.',
				icon: ClipboardList,
				summary:
					'Shift handover reports with KPI status against the thresholds, burden, temperatures, twin values and the LLM narrative; print on A4.',
				issue: 296
			},
			{
				href: '/furnacemind',
				label: 'FurnaceMind',
				title: 'FurnaceMind',
				subtitle: 'Ask about the furnace data; answers come with tables and charts.',
				icon: Brain,
				summary: 'Ask questions about furnace data in plain language and get tables and charts back.',
				issue: 313,
				legacy: 'FurnaceMind'
			}
		]
	},
	{
		label: 'Optimisers',
		items: [
			{
				href: '/optimiser',
				label: 'Burden & fuel',
				title: 'Burden & Fuel Optimiser',
				subtitle: 'Lowest-cost ore, flux and fuel blend and production level for today.',
				icon: SlidersHorizontal,
				summary:
					'Lowest-cost ore, flux and fuel blend and production level under the day’s limits, priced against today’s charge.',
				issue: 300,
				legacy: 'Burden & Fuel Optimiser'
			},
			{
				href: '/bmo',
				label: 'Blend mix',
				title: 'Blend Mix Optimiser',
				subtitle: 'Ore and flux blend at a fixed production.',
				icon: Blend,
				summary: 'Ore and flux blend at a fixed production with slag and hot-metal chemistry limits.',
				issue: 303,
				legacy: 'Blend Mix Optimiser'
			}
		]
	},
	{
		label: 'Admin',
		role: 'admin',
		items: [
			{
				href: '/admin',
				label: 'Admin',
				title: 'Admin',
				subtitle: 'Fact register, analyses, calibration, users and jobs.',
				icon: Shield,
				summary:
					'Fact register, what-if analysis, burnt vs charged coke, validation, calibration, users and job health.',
				issue: 304,
				children: ADMIN_VIEWS
			}
		]
	}
];

/** Titles of routes that are not in the navigation. */
const EXTRA_TITLES: Record<string, { title: string; subtitle: string }> = {
	'/dev/components': {
		title: 'Component catalogue',
		subtitle: 'Base components in their states, on sample values.'
	}
};

export function visibleGroups(role: Role | null | undefined): NavGroup[] {
	return NAV.filter((g) => !g.role || hasRole(role, g.role));
}

function normalise(pathname: string): string {
	return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
}

/** Every navigation item: the rail's pages and the views of their sections. */
function allItems(): NavItem[] {
	return NAV.flatMap((g) => g.items.flatMap((i) => [i, ...(i.children ?? [])]));
}

/**
 * Navigation item of a pathname: exact match, or the item whose route contains it (`/trace/2026-10-26`); a section's
 * view wins over the section (`/admin/audit`).
 */
export function navItemFor(pathname: string): NavItem | undefined {
	const path = normalise(pathname);
	const items = allItems();
	return (
		items.find((i) => i.href === path) ??
		items
			.filter((i) => i.href !== '/' && path.startsWith(`${i.href}/`))
			.sort((a, b) => b.href.length - a.href.length)[0]
	);
}

export interface PageHeading {
	/** Navigation group, shown before the title. */
	group?: string;
	title: string;
	subtitle?: string;
}

export function headingFor(pathname: string): PageHeading {
	const path = normalise(pathname);
	const item = navItemFor(path);
	const group = item && NAV.find((g) => g.items.some((i) => i === item || (i.children ?? []).includes(item)));
	if (item && group) return { group: group.label, title: item.title, subtitle: item.subtitle };
	const extra = EXTRA_TITLES[path];
	return extra ? { title: extra.title, subtitle: extra.subtitle } : { title: 'BF-02' };
}
