/**
 * Version line of the footer (SV-6.4, #318): the release the twin-api runs (`GET /api/version`, from the release
 * record `RELEASE.json` of `deploy/jetson/release.sh`) and the build of the SPA in this tab (`version` of
 * `$app/env`, the tag in a release build).
 */
import type { Schemas } from '../../api/client.ts';
import { istDateTime } from '../../time/format.ts';

export type ApiVersion = Schemas['Version'];

/** The first 7 characters of a commit; null when unknown. */
export function shortCommit(sha: string | null | undefined): string | null {
	if (!sha || sha === 'unknown') return null;
	return sha.slice(0, 7);
}

/** The API's side: `v1.2.0 (abc1234)` for a release, `0.1.0 dev` in a checkout. */
export function apiLabel(v: ApiVersion): string {
	const commit = shortCommit(v.git_sha);
	const name = v.release ?? `${v.version} ${v.environment}`;
	return commit ? `${name} (${commit})` : name;
}

/**
 * This tab runs an older SPA build than the one installed: SvelteKit's check of `_app/version.json` found a new
 * build (`polled`), or the production API reports another build for its release.
 */
export function buildOutdated(v: ApiVersion | null, spa: string, polled: boolean): boolean {
	if (polled) return true;
	return v !== null && v.environment === 'prod' && v.frontend_build != null && v.frontend_build !== spa;
}

/** Hover text: what each part names and when the release was installed (IST). */
export function versionTitle(v: ApiVersion | null, spa: string): string {
	const parts = [`SPA build ${spa}`];
	if (v) {
		parts.push(`twin-api ${v.version}, commit ${v.git_sha}`);
		if (v.release) {
			const at = v.installed_at ? `, installed ${istDateTime(new Date(v.installed_at))} IST` : '';
			parts.push(`release ${v.release}${at}`);
		}
	}
	return parts.join('; ');
}
