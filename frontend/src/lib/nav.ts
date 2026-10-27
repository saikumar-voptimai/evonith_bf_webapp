/**
 * Navigation of the operator console: page routes, labels, icons and the issue that builds each page.
 * The rail, the page titles and the planned-page placeholders all read this list.
 */
import type { LucideIcon } from '@lucide/svelte';
import type { RouteId } from '$app/types';
import Activity from '@lucide/svelte/icons/activity';
import Blend from '@lucide/svelte/icons/blend';
import Brain from '@lucide/svelte/icons/brain';
import ClipboardList from '@lucide/svelte/icons/clipboard-list';
import Flame from '@lucide/svelte/icons/flame';
import Layers from '@lucide/svelte/icons/layers';
import Shield from '@lucide/svelte/icons/shield';
import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';
import Thermometer from '@lucide/svelte/icons/thermometer';
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
}

export interface NavGroup {
	label: string;
	items: NavItem[];
	/** Minimum role that sees the group. */
	role?: Role;
}

export const ISSUE_URL = 'https://github.com/saikumar-voptimai/evonith_bf_webapp/issues/';

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
				subtitle: 'Shift, daily and weekly reports with flagged parameters.',
				icon: ClipboardList,
				summary: 'Shift, daily and weekly reports with flagged parameters, stability and operator notes.',
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
				issue: 304
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

/** Navigation item of a pathname: exact match, or the item whose route contains it (`/trace/2026-10-26`). */
export function navItemFor(pathname: string): NavItem | undefined {
	const path = normalise(pathname);
	const items = NAV.flatMap((g) => g.items);
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
	const group = item && NAV.find((g) => g.items.includes(item));
	if (item && group) return { group: group.label, title: item.title, subtitle: item.subtitle };
	const extra = EXTRA_TITLES[path];
	return extra ? { title: extra.title, subtitle: extra.subtitle } : { title: 'BF-02' };
}
