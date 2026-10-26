<!--
@component
Centred modal dialog for confirmations and short forms. Put the actions in `footer`; the confirming action names
what happens ("Apply plan", "Delete fact"), never "OK".
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Dialog from './Dialog.svelte';

	interface Props {
		open: boolean;
		title: string;
		description?: string;
		size?: 'sm' | 'md' | 'lg';
		dismissible?: boolean;
		onclose?: () => void;
		children: Snippet;
		footer?: Snippet;
	}

	let { open = $bindable(false), ...rest }: Props = $props();
</script>

<Dialog bind:open variant="modal" {...rest} />
