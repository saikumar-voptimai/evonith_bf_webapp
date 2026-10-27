#!/usr/bin/env node
/**
 * UI audit of the built SPA (SV-0.5, #274; sign-in SV-0.4, #273): axe-core on every page at 1920x1080, 1366x768 and
 * 768x1024, horizontal overflow, keyboard and dialog behaviour, sign-in, role guard and sign-out, and screenshots.
 *
 *   pnpm build && pnpm audit:ui                 # serves build/ with `vite preview`
 *   pnpm audit:ui --base http://127.0.0.1:5173  # audits a running server instead
 *   pnpm audit:ui --out ../audit-shots
 *
 * Needs Playwright with Chromium (`pnpm dlx playwright install chromium` on a dev machine). The script answers
 * /api/health, /api/stream and /api/auth/* itself (account `auditor`, password `correct-horse`), so the audit does
 * not need the twin-api. Exit code 1 on a critical or serious axe violation, a horizontal overflow or a failed
 * behaviour check.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const AXE = require.resolve('axe-core/axe.min.js');
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
	const i = argv.indexOf(`--${name}`);
	return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};
const OUT = resolve(arg('out', join(ROOT, '.audit')));
const PORT = 4179;
let base = arg('base', null);

const VIEWPORTS = [
	{ name: '1920x1080', width: 1920, height: 1080 },
	{ name: '1366x768', width: 1366, height: 768 },
	{ name: '768x1024', width: 768, height: 1024 }
];
const ROUTES = [
	{ name: 'live-twin', path: '/' },
	{ name: 'trace', path: '/trace' },
	{ name: 'catalogue', path: '/dev/components' },
	{ name: 'admin', path: '/admin' },
	{ name: 'not-found', path: '/no-such-page' }
];
const AXE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'];

const failures = [];
const report = { pages: [], checks: [] };

function fail(message) {
	failures.push(message);
	console.error(`FAIL ${message}`);
}

async function startPreview() {
	// Bound to the address the audit opens: `localhost` resolves to ::1 first on hosts such as the GitHub runners, and
	// a server on [::1] refuses the connection to 127.0.0.1.
	const args = ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', String(PORT), '--strictPort'];
	const proc = spawn('pnpm', args, {
		cwd: ROOT,
		stdio: ['ignore', 'pipe', 'pipe']
	});
	await new Promise((ok, bad) => {
		const timer = setTimeout(() => bad(new Error('vite preview did not start in 30 s')), 30_000);
		proc.stdout.on('data', (chunk) => {
			if (String(chunk).includes(String(PORT))) {
				clearTimeout(timer);
				ok();
			}
		});
		proc.on('exit', (code) => bad(new Error(`vite preview exited with ${code}`)));
	});
	return proc;
}

async function axe(page, label) {
	await page.addScriptTag({ path: AXE });
	const result = await page.evaluate(
		async (tags) =>
			await window.axe.run(document, { runOnly: { type: 'tag', values: tags }, resultTypes: ['violations'] }),
		AXE_TAGS
	);
	const violations = result.violations.map((v) => ({
		id: v.id,
		impact: v.impact,
		help: v.help,
		nodes: v.nodes.length,
		targets: v.nodes.slice(0, 4).map((n) => n.target.join(' '))
	}));
	for (const v of violations) {
		const line = `${label}: axe ${v.impact} ${v.id} (${v.nodes}) ${v.help} -> ${v.targets.join(' | ')}`;
		if (v.impact === 'critical' || v.impact === 'serious') fail(line);
		else console.warn(`WARN ${line}`);
	}
	return violations;
}

async function overflow(page, label) {
	const o = await page.evaluate(() => ({
		scroll: document.documentElement.scrollWidth,
		inner: window.innerWidth
	}));
	if (o.scroll > o.inner) fail(`${label}: horizontal overflow, scrollWidth ${o.scroll} > ${o.inner}`);
	return o;
}

async function settle(page) {
	await page.evaluate(() => document.fonts.ready);
	await page.waitForTimeout(250);
}

async function check(name, fn) {
	try {
		await fn();
		report.checks.push({ name, ok: true });
		console.log(`ok   ${name}`);
	} catch (e) {
		report.checks.push({ name, ok: false, error: String(e) });
		fail(`${name}: ${e instanceof Error ? e.message : e}`);
	}
}

function expect(cond, message) {
	if (!cond) throw new Error(message);
}

const preview = base ? null : await startPreview();
base ??= `http://127.0.0.1:${PORT}`;
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
try {
	const context = await browser.newContext({ deviceScaleFactor: 1, reducedMotion: 'reduce' });
	await context.route('**/api/health', (route) =>
		route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: '{"status":"ok","service":"twin-api"}'
		})
	);
	// Event stream mock: one heartbeat per connection; the EventSource reconnects after `retry` (SV-0.7, #276).
	await context.route(/\/api\/stream(\?.*)?$/, (route) => {
		const at = new Date().toISOString();
		const beat = JSON.stringify({ id: '0-0', type: 'heartbeat', at, data: { at } });
		return route.fulfill({
			status: 200,
			headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' },
			body: `retry: 3000\n\nevent: heartbeat\ndata: ${beat}\n\n`
		});
	});
	// Session mock: `auth.role` is the signed-in role, or null when signed out.
	const auth = { role: 'admin' };
	const problem = (route, status, detail) =>
		route.fulfill({
			status,
			contentType: 'application/problem+json',
			body: JSON.stringify({ type: 'about:blank', title: String(status), status, detail })
		});
	const sessionBody = () =>
		JSON.stringify({
			user: { username: 'auditor', role: auth.role, user_id: 'u-audit' },
			expires_at: new Date(Date.now() + 12 * 3600_000).toISOString()
		});
	await context.route('**/api/auth/me', (route) =>
		auth.role
			? route.fulfill({ status: 200, contentType: 'application/json', body: sessionBody() })
			: problem(route, 401, 'Sign in to continue.')
	);
	await context.route('**/api/auth/login', (route) => {
		const body = route.request().postDataJSON();
		if (body?.username !== 'auditor' || body?.password !== 'correct-horse')
			return problem(route, 401, 'Unknown user name or wrong password.');
		auth.role = 'admin';
		return route.fulfill({ status: 200, contentType: 'application/json', body: sessionBody() });
	});
	await context.route('**/api/auth/logout', (route) => {
		auth.role = null;
		return route.fulfill({ status: 204, body: '' });
	});
	const page = await context.newPage();
	const consoleErrors = [];
	page.on('console', (m) => {
		// Expected HTTP statuses: the 404 of the not-found route's document and the 401s of the sign-in checks.
		const url = m.location()?.url ?? '';
		const expected = url.endsWith('/no-such-page') || url.includes('/api/auth/');
		if (m.type() === 'error' && !expected) consoleErrors.push(`${m.text()} ${url}`);
	});
	page.on('pageerror', (e) => consoleErrors.push(String(e)));

	for (const vp of VIEWPORTS) {
		await page.setViewportSize({ width: vp.width, height: vp.height });
		for (const route of ROUTES) {
			const label = `${route.name}@${vp.name}`;
			await page.goto(base + route.path, { waitUntil: 'networkidle' });
			await settle(page);
			const violations = await axe(page, label);
			const o = await overflow(page, label);
			await page.screenshot({
				path: join(OUT, `${route.name}-${vp.name}.png`),
				fullPage: route.name === 'catalogue'
			});
			if (new URL(page.url()).pathname !== route.path) fail(`${label}: landed on ${page.url()}`);
			report.pages.push({ label, violations, overflow: o });
		}
	}

	// Behaviour at 1366x768 on the catalogue.
	await page.setViewportSize({ width: 1366, height: 768 });
	await page.goto(`${base}/dev/components`, { waitUntil: 'networkidle' });
	await settle(page);

	await check('page header shows the shift, the IST clock and the connection', async () => {
		const text = await page.locator('header.header').innerText();
		expect(/Shift [ABC]/.test(text), `no shift in "${text}"`);
		expect(/\d{2}:\d{2}\s+IST/.test(text), `no IST clock in "${text}"`);
		expect(text.includes('Connected'), `no connection state in "${text}"`);
	});

	await check('skip link moves focus to the page', async () => {
		await page.keyboard.press('Tab');
		const skip = await page.evaluate(() => document.activeElement?.textContent?.trim());
		expect(skip === 'Skip to content', `first tab stop is "${skip}"`);
		await page.keyboard.press('Enter');
		expect((await page.evaluate(() => document.activeElement?.id)) === 'main', 'focus did not reach main');
	});

	await check('rail marks the current page', async () => {
		const current = await page.locator('nav[aria-label="Main"] [aria-current="page"]').count();
		expect(current === 0, 'catalogue is not a navigation item, nothing should be current');
		await page.goto(`${base}/forge`, { waitUntil: 'networkidle' });
		const label = await page.locator('nav[aria-label="Main"] [aria-current="page"]').innerText();
		expect(label.trim() === 'FORGE', `current item is "${label}"`);
		await page.goto(`${base}/dev/components`, { waitUntil: 'networkidle' });
		await settle(page);
	});

	await check('tabs follow the arrow keys', async () => {
		await page.getByRole('tab', { name: 'Cost' }).focus();
		await page.keyboard.press('ArrowRight');
		expect(
			(await page.getByRole('tab', { name: 'Fuel' }).getAttribute('aria-selected')) === 'true',
			'ArrowRight did not select Fuel'
		);
		await page.keyboard.press('End');
		const last = await page.locator('[role="tab"][aria-selected="true"]').innerText();
		expect(last.trim().startsWith('Charging'), `End selected "${last}" (CO₂ is disabled)`);
		await page.keyboard.press('Home');
	});

	await check('table sorts by a header and reports aria-sort', async () => {
		const header = page.getByRole('columnheader', { name: /Fe/ }).first();
		await header.getByRole('button').click();
		expect((await header.getAttribute('aria-sort')) === 'descending', 'Fe is not sorted descending');
		const first = await page.locator('table').first().locator('tbody tr').first().innerText();
		expect(first.startsWith('Lloyds pellet'), `first row after sorting by Fe is "${first.slice(0, 30)}"`);
	});

	await check('number input refuses an entry above the maximum', async () => {
		const input = page.getByRole('spinbutton', { name: /Coke rate/ });
		await input.fill('472');
		const describedBy = await input.getAttribute('aria-describedby');
		const msg = await page.locator(`#${describedBy}`).innerText();
		// Value and unit are joined by a no-break space (formatQuantity, UNIT_SPACE).
		expect(msg.includes('The maximum is 450.0\u00a0kg/tHM.'), `message is "${msg}"`);
		expect((await input.getAttribute('aria-invalid')) === 'true', 'aria-invalid not set');
		await page.locator('.inputs').screenshot({ path: join(OUT, 'inputs-error.png') });
		await input.fill('372.5');
		await input.press('ArrowUp');
		expect((await input.inputValue()) === '373.0', `ArrowUp gave ${await input.inputValue()}`);
	});

	await check('modal traps focus, closes on Escape and returns focus', async () => {
		const opener = page.getByRole('button', { name: 'Open confirmation' });
		await opener.click();
		const dialog = page.getByRole('dialog', { name: 'Apply the plan to shift C?' });
		await dialog.waitFor();
		await settle(page);
		await axe(page, 'modal@1366x768');
		await page.screenshot({ path: join(OUT, 'modal-1366x768.png') });
		for (let i = 0; i < 6; i++) await page.keyboard.press('Tab');
		const inside = await page.evaluate(() => !!document.activeElement?.closest('dialog'));
		expect(inside, 'focus left the modal');
		await page.keyboard.press('Escape');
		await dialog.waitFor({ state: 'hidden' });
		const back = await page.evaluate(() => document.activeElement?.textContent?.trim());
		expect(back === 'Open confirmation', `focus returned to "${back}"`);
	});

	await check('drawer opens from the right and closes on the backdrop', async () => {
		await page.getByRole('button', { name: 'Open side panel' }).click();
		const dialog = page.getByRole('dialog', { name: 'Plan assumptions' });
		await dialog.waitFor();
		await settle(page);
		await axe(page, 'drawer@1366x768');
		await page.screenshot({ path: join(OUT, 'drawer-1366x768.png') });
		const box = await dialog.boundingBox();
		expect(box && Math.round(box.x + box.width) === 1366, 'drawer is not at the right edge');
		await page.mouse.click(40, 400);
		await dialog.waitFor({ state: 'hidden' });
	});

	await check('notifications appear and close', async () => {
		await page.getByRole('button', { name: 'Critical', exact: true }).click();
		await page.getByRole('button', { name: 'Success', exact: true }).click();
		const region = page.getByRole('region', { name: 'Notifications', exact: true });
		await region.waitFor();
		await settle(page);
		await page.screenshot({ path: join(OUT, 'toasts-1366x768.png') });
		await axe(page, 'toasts@1366x768');
		await page.getByRole('button', { name: 'Close notification: Optimiser failed' }).click();
		await page.getByRole('button', { name: 'Close notification: Plan saved' }).click();
		await region.waitFor({ state: 'detached' });
	});

	await check('job button runs through queued and running', async () => {
		const run = page.locator('.job').first().locator('button.run');
		await run.click();
		await page.waitForTimeout(400);
		expect((await run.innerText()).includes('Queued'), 'not queued');
		await page.waitForTimeout(3000);
		const text = await run.innerText();
		expect(/Running \d+ %/.test(text), `running label is "${text}"`);
		await page
			.locator('.job')
			.first()
			.screenshot({ path: join(OUT, 'job-running.png') });
	});

	// Charts and schematic (SV-0.6, #275).
	await check('72 h coke chart renders within 100 ms', async () => {
		const host = page.locator('[data-chart="coke-72h"][data-render-ms]');
		await host.waitFor({ timeout: 15_000 });
		const ms = Number(await host.getAttribute('data-render-ms'));
		report.renderMs = ms;
		expect(ms < 100, `rendered in ${ms} ms`);
	});

	await check('chart legend toggles a series and the table view lists every value', async () => {
		const panel = page.locator('[data-chart="coke-72h"]');
		const forge = panel.getByRole('button', { name: 'FORGE, modelled' });
		await forge.click();
		expect((await forge.getAttribute('aria-pressed')) === 'false', 'FORGE still shown');
		await forge.click();
		await panel.getByRole('button', { name: 'Table' }).click();
		const rows = await panel.locator('table tbody tr').count();
		expect(rows === 72, `table view has ${rows} rows, expected 72 hours`);
		await settle(page);
		await axe(page, 'chart-table@1366x768');
		await panel.getByRole('button', { name: 'Chart' }).click();
		await panel.locator('[data-render-ms]').first().waitFor();
	});

	await check('schematic zone highlights its readings', async () => {
		const tuyere = page.getByRole('button', { name: 'Tuyere', exact: true });
		await tuyere.click();
		expect((await tuyere.getAttribute('aria-pressed')) === 'true', 'zone not selected');
		const on = await page.locator('.schematic .callout.on').count();
		expect(on === 3, `${on} callouts highlighted, expected 3 (blast temperature, RAFT, blast pressure)`);
		await page
			.locator('section.panel', { has: page.getByRole('heading', { name: 'Furnace schematic' }) })
			.screenshot({ path: join(OUT, 'schematic-tuyere-1366x768.png') });
		await tuyere.click();
	});

	await check('catalogue fits a 390 px phone; the schematic lists its readings', async () => {
		await page.setViewportSize({ width: 390, height: 844 });
		await page.goto(`${base}/dev/components`, { waitUntil: 'networkidle' });
		await settle(page);
		const label = 'catalogue@390x844';
		report.pages.push({ label, violations: await axe(page, label), overflow: await overflow(page, label) });
		await page.screenshot({ path: join(OUT, 'catalogue-390x844.png'), fullPage: true });
		const readings = await page.locator('.schematic.compact .readings li').count();
		expect(readings === 8, `${readings} readings in the compact schematic, expected 8`);
		const endLabels = await page
			.locator('[data-chart="coke-72h"] svg text')
			.filter({ hasText: /^TRACE / })
			.count();
		expect(endLabels === 0, 'end labels drawn on a 390 px chart');
		await page.setViewportSize({ width: 1366, height: 768 });
		await page.goto(`${base}/dev/components`, { waitUntil: 'networkidle' });
		await settle(page);
	});

	await check('rail collapses to icons and keeps accessible names', async () => {
		await page.getByRole('button', { name: 'Collapse navigation' }).click();
		await settle(page);
		const width = await page.locator('.rail-slot').evaluate((el) => el.getBoundingClientRect().width);
		expect(width <= 70, `collapsed rail is ${width}px wide`);
		expect(await page.getByRole('link', { name: 'STRATA' }).isVisible(), 'STRATA link not visible');
		await page.screenshot({ path: join(OUT, 'rail-collapsed-1366x768.png') });
		await axe(page, 'rail-collapsed@1366x768');
		await page.getByRole('button', { name: 'Expand navigation' }).click();
	});

	await check('navigation drawer replaces the rail below 1024 px', async () => {
		await page.setViewportSize({ width: 768, height: 1024 });
		await page.goto(`${base}/trace`, { waitUntil: 'networkidle' });
		await settle(page);
		expect(!(await page.locator('.rail-slot').isVisible()), 'rail visible at 768 px');
		await page.getByRole('button', { name: 'Open navigation' }).click();
		const dialog = page.getByRole('dialog', { name: 'V-OptimAIse' });
		await dialog.waitFor();
		await settle(page);
		await axe(page, 'nav-drawer@768x1024');
		await page.screenshot({ path: join(OUT, 'nav-drawer-768x1024.png') });
		await dialog.getByRole('link', { name: 'STRATA' }).click();
		await page.waitForURL('**/strata');
		await dialog.waitFor({ state: 'hidden' });
	});

	await check('signed-out visit goes to the sign-in page and back', async () => {
		auth.role = null;
		await page.setViewportSize({ width: 1366, height: 768 });
		await page.goto(`${base}/strata`, { waitUntil: 'networkidle' });
		await page.waitForURL('**/login?next=%2Fstrata');
		await settle(page);
		for (const vp of VIEWPORTS) {
			await page.setViewportSize({ width: vp.width, height: vp.height });
			await settle(page);
			const label = `login@${vp.name}`;
			report.pages.push({ label, violations: await axe(page, label), overflow: await overflow(page, label) });
			await page.screenshot({ path: join(OUT, `login-${vp.name}.png`) });
		}
		await page.setViewportSize({ width: 1366, height: 768 });
		await page.getByLabel('User name').fill('auditor');
		await page.getByLabel('Password', { exact: true }).fill('wrong');
		await page.getByRole('button', { name: 'Sign in' }).click();
		const alert = page.getByRole('alert');
		await alert.getByText('Unknown user name or wrong password.').waitFor();
		await page.screenshot({ path: join(OUT, 'login-refused-1366x768.png') });
		expect(
			(await page.getByLabel('Password', { exact: true }).inputValue()) === '',
			'password field kept the refused password'
		);
		await page.getByLabel('Password', { exact: true }).fill('correct-horse');
		await page.getByRole('button', { name: 'Sign in' }).click();
		await page.waitForURL('**/strata');
		const current = await page.locator('nav[aria-label="Main"] [aria-current="page"]').innerText();
		expect(current.trim() === 'STRATA', `after sign-in the current page is "${current}"`);
	});

	await check('the Admin pages refuse a supervisor', async () => {
		auth.role = 'supervisor';
		await page.goto(`${base}/trace`, { waitUntil: 'networkidle' });
		expect(
			(await page.getByRole('link', { name: 'Admin' }).count()) === 0,
			'Admin link shown to a supervisor'
		);
		await page.goto(`${base}/admin`, { waitUntil: 'networkidle' });
		await settle(page);
		const text = await page.locator('main').innerText();
		expect(
			text.includes('The Admin pages are for administrators.'),
			`admin page shows "${text.slice(0, 80)}"`
		);
		await axe(page, 'admin-403@1366x768');
		await page.screenshot({ path: join(OUT, 'admin-403-1366x768.png') });
	});

	await check('sign-out ends the session and shows the sign-in page', async () => {
		auth.role = 'admin';
		await page.goto(`${base}/forge`, { waitUntil: 'networkidle' });
		await page.getByRole('button', { name: /account menu/ }).click();
		await page.getByRole('button', { name: 'Sign out' }).click();
		await page.waitForURL('**/login');
		expect(auth.role === null, 'logout was not called');
	});

	await check('no console errors', async () => {
		expect(consoleErrors.length === 0, consoleErrors.join(' | '));
	});
} finally {
	await browser.close();
	preview?.kill('SIGTERM');
}

writeFileSync(join(OUT, 'report.json'), JSON.stringify({ ...report, failures }, null, 2));
const total = report.pages.reduce((n, p) => n + p.violations.length, 0);
console.log(
	`\n${report.pages.length} page views, ${total} axe findings, ${report.checks.length} checks, ${failures.length} failures`
);
console.log(`Screenshots and report: ${OUT}`);
process.exit(failures.length ? 1 : 0);
