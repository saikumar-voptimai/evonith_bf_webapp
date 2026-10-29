// Streamlit pages under /legacy: title in config/page_registry.py → URL name given by st.navigation.
// Written by scripts/export_legacy_pages.py (make fixtures), checked by tests/test_legacy_pages.py.
export const LEGACY_PAGES = {
	Welcome: '',
	'Data Explorer': 'Data_Explorer',
	'V-Board': 'Data_Visualisation',
	'V-Sense': 'Recommendations',
	CoPilot: 'AI_Copilot',
	'Material Balance': 'Material_Balance',
	FurnaceMind: 'FurnaceMind',
	'Blend Mix Optimiser': 'Blend_Optimizer',
	'Coke Energy Model': 'Coke_Energy',
	'Burden & Fuel Optimiser': 'Burden_Fuel_Optimiser',
	Feedback: 'Feedback'
} as const;

// Sections the SPA links to (PageSection): via 'query' = ?section=<key> and the params; 'anchor' = #<key>.
export const LEGACY_SECTIONS = {
	'Blend Mix Optimiser': {
		'de-search': { title: 'Intensive Optimizer (DE search)', via: 'anchor', params: [] },
		replay: { title: 'Snapshot replay', via: 'query', params: ['snapshot'] },
		sandbox: { title: 'Sandbox', via: 'query', params: [] }
	}
} as const;
