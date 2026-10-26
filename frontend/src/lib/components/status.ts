/** Status vocabulary shared by chips, tiles, freshness labels and notifications. */
import type { LucideIcon } from '@lucide/svelte';
import CircleCheck from '@lucide/svelte/icons/circle-check';
import ClockAlert from '@lucide/svelte/icons/clock-alert';
import Info from '@lucide/svelte/icons/info';
import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
import TriangleAlert from '@lucide/svelte/icons/triangle-alert';

export type StatusKind = 'ok' | 'warning' | 'critical' | 'stale' | 'info';

/** Default word for each status; a status is never shown by colour alone. */
export const STATUS_WORD: Record<StatusKind, string> = {
	ok: 'Normal',
	warning: 'Warning',
	critical: 'Critical',
	stale: 'Stale',
	info: 'Info'
};

export const STATUS_ICON: Record<StatusKind, LucideIcon> = {
	ok: CircleCheck,
	warning: TriangleAlert,
	critical: OctagonAlert,
	stale: ClockAlert,
	info: Info
};

/** Age state of a data point: `ok` below `warnAfterMin`, `warning` below `staleAfterMin`, then `stale`. */
export function ageStatus(
	at: Date | null | undefined,
	now: Date,
	warnAfterMin = 20,
	staleAfterMin = 75
): 'ok' | 'warning' | 'stale' {
	if (!at) return 'stale';
	const minutes = (now.getTime() - at.getTime()) / 60_000;
	if (minutes < warnAfterMin) return 'ok';
	if (minutes < staleAfterMin) return 'warning';
	return 'stale';
}
