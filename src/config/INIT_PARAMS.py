# File PARAMS
folder = f'./src/'

IMG_VOPTIMAI = "images/VOPTIMAI.png"
IMG_HOSPET = "images/HOSPET.jpeg"
IMG_MBF = "images/MBF.jpg"
DATA = folder + f'data/data.pkl'

MODEL = folder + f'models/model.pkl'

# VARIABLES:     
SINTER_PARAMS = ['SINTER SP02_Fe(T)%', 'SINTER SP02_FeO%', 'SINTER SP02_SiO2%', 'SINTER SP02_Al2O3%', 'SINTER SP02_CaO%', 'SINTER SP02_MgO%', 'SINTER SP02_Na2O%', 'SINTER SP02_K2O%', 'SINTER SP02_P%', 'SINTER SP02_MnO%', 'SINTER SP02_TiO2%', 'SINTER SP02_Basicity', 
                ]

PELLET_PARAMS = ['PELLET_%Fe(T)', 'PELLET_%SiO2', 'PELLET_%Al2O3', 'PELLET_%P', 'PELLET_%TiO2', 'PELLET_%Na2O', 'PELLET_%K2O', 'PELLET_%MnO', 'PELLET_%CaO', 'PELLET_%MgO', 'PELLET_%LOI', 'PELLET_%TM', 
                ]

ORE_PARAMS = ['ORE_%Fe(T)', 'ORE_%SiO2', 'ORE_%Al2O3', 'ORE_%P', 'ORE_%TiO2', 'ORE_%Na2O', 'ORE_%K2O', 'ORE_%MnO', 'ORE_%CaO', 'ORE_%MgO', 'ORE_%LOI', 'ORE_%TM']
                
COKE_PARAMS = ['COKE_MOIST%', 'COKE_IM%', 'COKE_ASH%', 'COKE_VM% ', 'COKE_FC%', 
                'NUT COKE(12&19)_MOIST%', 'NUT COKE(12&19)_IM%', 'NUT COKE(12&19)_ASH%', 'NUT COKE(12&19)_VM% ', 'NUT COKE(12&19)_FC%', 
                'PCI COAL_%TM', 'PCI COAL_%IM', 'PCI COAL_%Ash', 'PCI COAL_%VM', 'PCI COAL_%FC']

FLUX_PARAMS = ['Limestone_MT', 'Dolomite_MT', 'Quartzite_MT']

OTHER_PARAMS = ['Total Pellet (Wet)_MT', 'Total Pellet (Wet)_%', 'Total Gross CLO (Wet)_MT', 'Total Gross CLO (Wet)_%', 'Total Sinter Consumption (Wet)_MT', 'Total Sinter Consumption (Wet)_%', 'Total IBRM Wet_MT']

CONTROL_PARAMS = ['UTILITY _Steam (Blast Furnace)_Tons', 'UTILITY _Steam (Turbo Blower)_Tons', 
                'CBV From BlowerNm3/Hr.', 'Hot Blast VolumeNm3/Hr.', 
                'ActualKg/Thm.', 'Hot Blast PressureBar', 'Hot Blast Temp.oC', 'Oxygen\nFlowNm3/Hr.', 'SteamKgs/Hr.', 'Tuyere\nVelocitym/s', 'RAFToC']

INPUT_PARAMS = ['COKE_MOIST%', 'COKE_IM%', 'COKE_ASH%', 'COKE_VM% ', 'COKE_FC%',
                'NUT COKE(12&19)_MOIST%', 'NUT COKE(12&19)_IM%', 'NUT COKE(12&19)_ASH%', 'NUT COKE(12&19)_VM% ', 'NUT COKE(12&19)_FC%',
                'SINTER SP02_Fe(T)%', 'SINTER SP02_FeO%', 'SINTER SP02_SiO2%', 'SINTER SP02_Al2O3%', 'SINTER SP02_CaO%', 'SINTER SP02_MgO%', 'SINTER SP02_Na2O%', 'SINTER SP02_K2O%', 'SINTER SP02_P%', 'SINTER SP02_MnO%', 'SINTER SP02_TiO2%', 'SINTER SP02_Basicity',
                'Total Pellet (Wet)_MT', 'Total Pellet (Wet)_%', 'Total Gross CLO (Wet)_MT', 'Total Gross CLO (Wet)_%', 'Total Sinter Consumption (Wet)_MT', 'Total Sinter Consumption (Wet)_%', 'Total IBRM Wet_MT',
                'Limestone_MT', 'Dolomite_MT', 'Quartzite_MT',
                'PCI COAL_%TM', 'PCI COAL_%IM', 'PCI COAL_%Ash', 'PCI COAL_%VM', 'PCI COAL_%FC',
                'PELLET_%Fe(T)', 'PELLET_%SiO2', 'PELLET_%Al2O3', 'PELLET_%P', 'PELLET_%TiO2', 'PELLET_%Na2O', 'PELLET_%K2O', 'PELLET_%MnO', 'PELLET_%CaO', 'PELLET_%MgO', 'PELLET_%LOI', 'PELLET_%TM',
                'ORE_%Fe(T)', 'ORE_%SiO2', 'ORE_%Al2O3', 'ORE_%P', 'ORE_%TiO2', 'ORE_%Na2O', 'ORE_%K2O', 'ORE_%MnO', 'ORE_%CaO', 'ORE_%MgO', 'ORE_%LOI', 'ORE_%TM',
                ]

INPUT_PARAMS_MODEL = ['COKE_MOIST%', 'COKE_IM%', 'COKE_ASH%', 'COKE_VM% ', 'COKE_FC%', 
                'NUT COKE(12&19)_MOIST%', 'NUT COKE(12&19)_IM%', 'NUT COKE(12&19)_ASH%', 'NUT COKE(12&19)_VM% ', 'NUT COKE(12&19)_FC%', 
                'SINTER SP02_Fe(T)%', 'SINTER SP02_FeO%', 'SINTER SP02_SiO2%', 'SINTER SP02_Al2O3%', 'SINTER SP02_CaO%', 'SINTER SP02_MgO%', 'SINTER SP02_Na2O%', 'SINTER SP02_K2O%', 'SINTER SP02_P%', 'SINTER SP02_MnO%', 'SINTER SP02_TiO2%', 'SINTER SP02_Basicity', 
                'Total Pellet (Wet)_MT', 'Total Pellet (Wet)_%', 'Total Gross CLO (Wet)_MT', 'Total Gross CLO (Wet)_%', 'Total Sinter Consumption (Wet)_MT', 'Total Sinter Consumption (Wet)_%', 'Total IBRM Wet_MT', 
                'Limestone_MT', 'Dolomite_MT', 'Quartzite_MT', 
                'PCI COAL_%TM', 'PCI COAL_%IM', 'PCI COAL_%Ash', 'PCI COAL_%VM', 'PCI COAL_%FC', 
                'PELLET_%Fe(T)', 'PELLET_%SiO2', 'PELLET_%Al2O3', 'PELLET_%P', 'PELLET_%TiO2', 'PELLET_%Na2O', 'PELLET_%K2O', 'PELLET_%MnO', 'PELLET_%CaO', 'PELLET_%MgO', 'PELLET_%LOI', 'PELLET_%TM', 
                'ORE_%Fe(T)', 'ORE_%SiO2', 'ORE_%Al2O3', 'ORE_%P', 'ORE_%TiO2', 'ORE_%Na2O', 'ORE_%K2O', 'ORE_%MnO', 'ORE_%CaO', 'ORE_%MgO', 'ORE_%LOI', 'ORE_%TM',
                'UTILITY _Steam (Blast Furnace)_Tons', 'UTILITY _Steam (Turbo Blower)_Tons', 
                'CBV From BlowerNm3/Hr.', 'Hot Blast VolumeNm3/Hr.', 
                'ActualKg/Thm.', 'Hot Blast PressureBar', 'Hot Blast Temp.oC', 'Oxygen\nFlowNm3/Hr.', 'SteamKgs/Hr.', 'Tuyere\nVelocitym/s', 'RAFToC', 
                ]

OUTPUT_PARAMS = ['PRODUCTION PRODUCT _Total Hot Metal Production_MT', 'Furnace Top Gas AnalysisCO2\n(CO+CO2)ŋ CO', 'Act. Fuel RateKg/Thm.']

# Optimiser Params:
OPTIM_STEPS = 10
