import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// Development proxy: the SPA calls relative paths (/api, /legacy) exactly as it does behind Caddy in production.
const api = process.env.BF_API_URL ?? 'http://127.0.0.1:8100';
const legacy = process.env.BF_LEGACY_URL ?? 'http://127.0.0.1:8501';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Runes mode for project files; libraries decide for themselves.
				runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
			},
			// Static single-page application. Caddy (or the API) serves `build/`; every path that is not a file
			// falls back to `200.html`, so client-side routes survive a reload.
			adapter: adapter({
				pages: 'build',
				assets: 'build',
				fallback: '200.html',
				precompress: false,
				strict: true
			})
		})
	],
	server: {
		port: 5173,
		strictPort: true,
		proxy: {
			'/api': { target: api, changeOrigin: false },
			'/legacy': { target: legacy, changeOrigin: false, ws: true }
		}
	},
	test: {
		expect: { requireAssertions: true },
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'unit',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
});
