<!--
@component
Side panel over the page for detail and settings that keep the page in view: a blend's assumptions, a shift's
detail, run options. Opens from the right by default; the navigation drawer opens from the left.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Dialog from './Dialog.svelte';

	interface Props {
		open: boolean;
		title: string;
		description?: string;
		side?: 'left' | 'right';
		size?: 'sm' | 'md' | 'lg';
		dismissible?: boolean;
		/** Black background of the navigation chrome. */
		chrome?: boolean;
		onclose?: () => void;
		children: Snippet;
		footer?: Snippet;
	}

	let { open = $bindable(false), ...rest }: Props = $props();
</script>

<Dialog bind:open variant="drawer" {...rest} />
