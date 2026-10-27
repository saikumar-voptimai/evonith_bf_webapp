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
