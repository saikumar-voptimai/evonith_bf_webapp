<!--
@component
Label text with chemical subscripts: `<Formula text="SiO₂" />` renders SiO<sub>2</sub> in Montserrat digits.
-->
<script lang="ts">
	import { formulaParts } from '../format/formula.ts';

	interface Props {
		text: string;
	}

	let { text }: Props = $props();

	const parts = $derived(formulaParts(text));
</script>

<span class="formula"
	>{#each parts as part, i (i)}{#if part.sub}<sub>{part.text}</sub>{:else}{part.text}{/if}{/each}</span
>
