<!--
@component
One run parameter that replaces a page-11 default: empty keeps the default, an entry outside the API's bounds is
refused (never clamped). Reports its validity to the form, which blocks the run while an entry is refused.
-->
<script lang="ts">
	import NumberInput from '../components/NumberInput.svelte';
	import type { FieldSpec } from './optimiser.ts';

	interface Props {
		spec: FieldSpec;
		value: number | null;
		hint: string;
		disabled?: boolean;
		onchange: (key: string, value: number | null) => void;
		onvalid: (key: string, valid: boolean) => void;
	}

	let { spec, value, hint, disabled = false, onchange, onvalid }: Props = $props();
	let valid = $state(true);

	$effect(() => {
		onvalid(spec.key, valid);
		return () => onvalid(spec.key, true);
	});
</script>

<NumberInput
	label={spec.label}
	unit={spec.unit || undefined}
	min={spec.min}
	max={spec.max}
	step={spec.step}
	digits={spec.digits}
	{hint}
	{value}
	{disabled}
	width="100%"
	bind:valid
	onchange={(v) => onchange(spec.key, v)}
/>
