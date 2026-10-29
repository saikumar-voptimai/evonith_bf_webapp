import { describe, expect, it } from 'vitest';
import { apiLabel, buildOutdated, shortCommit, versionTitle, type ApiVersion } from './version.ts';

const release: ApiVersion = {
	version: '0.1.0',
	git_sha: '0123456789abcdef0123456789abcdef01234567',
	environment: 'prod',
	python: '3.12.10',
	started_at: '2026-10-29T11:30:00Z',
	release: 'v1.2.0',
	installed_at: '2026-10-29T11:29:00Z',
	frontend_build: 'v1.2.0'
};

const checkout: ApiVersion = {
	version: '0.1.0',
	git_sha: 'unknown',
	environment: 'dev',
	python: '3.12.10',
	started_at: '2026-10-29T11:30:00Z',
	release: null,
	installed_at: null,
	frontend_build: null
};

describe('footer version', () => {
	it('shortens a commit and drops an unknown one', () => {
		expect(shortCommit(release.git_sha)).toBe('0123456');
		expect(shortCommit('unknown')).toBeNull();
		expect(shortCommit(null)).toBeNull();
	});

	it('names the release and its commit, or the API version in a checkout', () => {
		expect(apiLabel(release)).toBe('v1.2.0 (0123456)');
		expect(apiLabel(checkout)).toBe('0.1.0 dev');
		expect(apiLabel({ ...checkout, git_sha: 'abcdef0123' })).toBe('0.1.0 dev (abcdef0)');
	});

	it('flags a tab whose build is older than the installed one', () => {
		expect(buildOutdated(release, 'v1.2.0', false)).toBe(false);
		expect(buildOutdated(release, 'v1.1.0', false)).toBe(true);
		expect(buildOutdated(release, 'v1.2.0', true)).toBe(true); // SvelteKit's poll found a new build
		expect(buildOutdated(null, 'v1.2.0', false)).toBe(false);
		// in development the API's build is the last local build, not what the dev server serves
		expect(buildOutdated({ ...checkout, frontend_build: 'abc1234' }, 'def5678-dirty', false)).toBe(false);
	});

	it('explains the parts on hover, installation time in IST', () => {
		expect(versionTitle(release, 'v1.2.0')).toBe(
			'SPA build v1.2.0; twin-api 0.1.0, commit 0123456789abcdef0123456789abcdef01234567; ' +
				'release v1.2.0, installed 29 Oct 16:59 IST'
		);
		expect(versionTitle(null, 'dev')).toBe('SPA build dev');
	});
});
