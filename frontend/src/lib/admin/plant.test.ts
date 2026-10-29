import { describe, expect, it } from 'vitest';
import {
	burdenChanges,
	burdenForm,
	burdenLayout,
	burdenPayload,
	burdenValueText,
	changedMaterials,
	changedText,
	effectiveIso,
	historyRows,
	hopperChanges,
	hopperForm,
	plantTab,
	registrationProblem,
	roleLabel,
	snapshotCount,
	writerText,
	type BurdenField
} from './plant.ts';

const NBSP = ' ';

function position(charge: 'coke' | 'non_coke', p: number, quantity: 'rings' | 'angles'): BurdenField {
	const prefix = charge === 'coke' ? 'coke' : 'noncoke';
	return {
		name: `${prefix}_p${String(p).padStart(2, '0')}_${quantity}`,
		label: `${charge === 'coke' ? 'Coke' : 'Non-coke'} P${String(p).padStart(2, '0')} ${quantity}`,
		kind: 'number',
		unit: quantity === 'angles' ? '°' : 'rotations',
		charge,
		position: p,
		quantity
	};
}

const FIELDS: BurdenField[] = [
	position('coke', 2, 'rings'),
	position('coke', 1, 'rings'),
	position('coke', 1, 'angles'),
	{
		name: 'coke_discharge_time',
		label: 'Coke discharge time',
		kind: 'number',
		unit: 's',
		charge: 'coke',
		position: null,
		quantity: null
	},
	{
		name: 'coke_charge_pattern',
		label: 'Coke charge pattern',
		kind: 'text',
		unit: null,
		charge: 'coke',
		position: null,
		quantity: null
	},
	position('non_coke', 1, 'rings'),
	{
		name: 'burden_changing_purpose',
		label: 'Burden changing purpose',
		kind: 'text',
		unit: null,
		charge: null,
		position: null,
		quantity: null
	}
];

describe('accounts', () => {
	it('requires a user name and a password after trimming, as the Streamlit form', () => {
		expect(registrationProblem(' qa ', ' pw ')).toBeNull();
		expect(registrationProblem('  ', 'pw')).toBe('Enter a user name and a password.');
		expect(registrationProblem('qa', ' ')).toBe('Enter a user name and a password.');
		expect(registrationProblem('x'.repeat(129), 'pw')).toBe('A user name has at most 128 characters.');
	});

	it('labels roles and keeps an unknown stored role', () => {
		expect(roleLabel('admin')).toBe('Administrator');
		expect(roleLabel('operator')).toBe('operator');
	});

	it('reads the tab from the URL', () => {
		expect(plantTab('burden')).toBe('burden');
		expect(plantTab(null)).toBe('users');
		expect(plantTab('nope')).toBe('users');
	});
});

describe('effective time', () => {
	it('reads the entry as IST and sends it as UTC', () => {
		expect(effectiveIso('2025-07-05T06:00')).toBe('2025-07-05T00:30:00.000Z');
		expect(effectiveIso('2025-07-05T00:15')).toBe('2025-07-04T18:45:00.000Z'); // C shift, the IST day before in UTC
		expect(effectiveIso('2025-07-05')).toBeNull();
		expect(effectiveIso('')).toBeNull();
	});
});

describe('hoppers', () => {
	const hoppers = [
		{ code: 'hopper_01', display_name: 'Bin 1' },
		{ code: 'hopper_02', display_name: 'Bin 2' },
		{ code: 'hopper_03', display_name: 'Bin 3' }
	];

	it('fills the form from the mapping in force, unassigned where it has none', () => {
		const current = {
			id: 1,
			date_time: '2025-06-28T00:30:00Z',
			source_type: 'import',
			ip_address: null,
			user_id: null,
			user_name: null,
			materials: { hopper_01: 'Sinter' }
		};
		expect(hopperForm(hoppers, current)).toEqual({
			hopper_01: 'Sinter',
			hopper_02: 'UNASSIGNED',
			hopper_03: 'UNASSIGNED'
		});
		expect(hopperForm(hoppers, null)).toEqual({
			hopper_01: 'UNASSIGNED',
			hopper_02: 'UNASSIGNED',
			hopper_03: 'UNASSIGNED'
		});
	});

	it('changes are the hoppers that differ from the mapping in force (domain.plant_config.hopper_changes)', () => {
		const current = { hopper_01: 'Sinter', hopper_02: 'UNASSIGNED' };
		const form = { hopper_01: 'Sinter', hopper_02: 'Quartz', hopper_03: 'UNASSIGNED', hopper_04: 'Dolomite' };
		const changed = hopperChanges(form, current);
		expect(changed).toEqual(['hopper_02', 'hopper_04']);
		expect(changedMaterials(form, changed)).toEqual({ hopper_02: 'Quartz', hopper_04: 'Dolomite' });
		expect(hopperChanges(form, null)).toEqual(['hopper_01', 'hopper_02', 'hopper_04']);
	});
});

describe('burden', () => {
	it('arranges the fields as the charge matrix', () => {
		const layout = burdenLayout(FIELDS);
		expect(layout.charges.map((c) => c.label)).toEqual(['Coke', 'Non-coke']);
		const [coke, nonCoke] = layout.charges;
		expect(coke!.rings.map((f) => f.position)).toEqual([1, 2]);
		expect(coke!.angles.map((f) => f.name)).toEqual(['coke_p01_angles']);
		expect(coke!.other.map((f) => f.name)).toEqual(['coke_discharge_time', 'coke_charge_pattern']);
		expect(nonCoke!.rings.map((f) => f.name)).toEqual(['noncoke_p01_rings']);
		expect(layout.general.map((f) => f.name)).toEqual(['burden_changing_purpose']);
	});

	const current = {
		coke_p01_rings: 3,
		coke_p02_rings: null,
		coke_p01_angles: 38,
		coke_discharge_time: 92,
		coke_charge_pattern: 'C 3 2 2',
		noncoke_p01_rings: 2,
		burden_changing_purpose: null
	};

	it('fills the form with numbers or null and text or an empty string', () => {
		expect(burdenForm(FIELDS, current)).toEqual({ ...current, burden_changing_purpose: '' });
		expect(burdenForm(FIELDS, { coke_p01_rings: Number.NaN })['coke_p01_rings']).toBeNull();
	});

	it('changes compare the values as the snapshot stores them (domain.plant_config.burden_changes)', () => {
		const form = burdenForm(FIELDS, current);
		expect(burdenChanges(FIELDS, form, current)).toEqual([]);
		form['coke_p01_angles'] = 31.5;
		form['coke_charge_pattern'] = '';
		form['burden_changing_purpose'] = 'Open the centre';
		expect(burdenChanges(FIELDS, form, current)).toEqual([
			'coke_p01_angles',
			'coke_charge_pattern',
			'burden_changing_purpose'
		]);
		expect(burdenPayload(FIELDS, form)).toEqual({
			...current,
			coke_p01_angles: 31.5,
			coke_charge_pattern: null,
			burden_changing_purpose: 'Open the centre'
		});
	});

	it('shows values with their units', () => {
		const [rings, , angle, time, pattern] = FIELDS;
		expect(burdenValueText(rings!, 3)).toBe(`3${NBSP}rotations`);
		expect(burdenValueText(angle!, 31.5)).toBe(`31.5${NBSP}°`);
		expect(burdenValueText(time!, 92)).toBe(`92.0${NBSP}s`);
		expect(burdenValueText(time!, null)).toBe('—');
		expect(burdenValueText(pattern!, 'C 3 2 2')).toBe('C 3 2 2');
		expect(burdenValueText(pattern!, '')).toBe('—');
	});
});

describe('history', () => {
	const row = (
		id: number,
		date_time: string,
		materials: Record<string, string>,
		user_name: string | null
	) => ({
		id,
		date_time,
		source_type: user_name ? 'webapp' : 'import',
		ip_address: null,
		user_id: user_name ? `uuid-${user_name}` : null,
		user_name,
		materials
	});

	it('lists what each snapshot changed against the one before it', () => {
		const rows = historyRows(
			[
				row(3, '2025-07-05T00:30:00Z', { hopper_01: 'Sinter', hopper_02: 'Quartz' }, 'adm'),
				row(2, '2025-07-01T08:30:00Z', { hopper_01: 'Sinter', hopper_02: 'Quartz' }, 'adm'),
				row(1, '2025-06-28T00:30:00Z', { hopper_01: 'Pellet', hopper_02: 'UNASSIGNED' }, null)
			],
			(r) => r.materials
		);
		expect(rows.map((r) => r.changed)).toEqual([[], ['hopper_01', 'hopper_02'], null]);
		const labels = (k: string) => k.replace('hopper_', 'Bin ');
		expect(rows.map((r) => changedText(r.changed, labels))).toEqual([
			'No change',
			'Bin 01, Bin 02',
			'First snapshot'
		]);
		expect(changedText(['a', 'b', 'c', 'd', 'e'], (k) => k)).toBe('a, b, c and 2 more');
	});

	it('names the writer, or the account id, or the source', () => {
		expect(writerText({ user_name: 'adm', user_id: 'x', source_type: 'webapp' })).toBe('adm');
		expect(writerText({ user_name: null, user_id: '0123456789ab', source_type: 'webapp' })).toBe('01234567…');
		expect(writerText({ user_name: null, user_id: null, source_type: 'import' })).toBe('import');
		expect(snapshotCount(1)).toBe('1 snapshot');
		expect(snapshotCount(3)).toBe('3 snapshots');
	});
});
