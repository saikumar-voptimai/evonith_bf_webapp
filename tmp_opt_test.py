import sys, joblib
sys.path.insert(0, 'src')
from utils.recommendations.optimiser import run_optimiser
from utils.recommendations.data import DataframesProcessor
from config.config_loader import load_config

config = load_config('setting_ds_dv.yml')
config_vsense = load_config('setting_vsense.yml')
models_dict = config_vsense['Optimisation']
optimisation_type = 'Eta CO'

for model in models_dict.keys():
    models_dict[model]['input_params_flat'] = [v for group in models_dict[model]['input_params'].values() for v in group]
    models_dict[model]['Optimised'] = False
    models_dict[model]['LoadedMLModel'] = joblib.load(models_dict[model]['model'])
    if model == optimisation_type:
        models_dict[model]['Optimised'] = True

processor = DataframesProcessor(config=config, config_vsense=config_vsense, debug_on=False)
feat_vec_target = processor.process_dataframe(scaler_path=models_dict[optimisation_type]['scaling'])
result = run_optimiser(feat_vec_target, models_dict, {}, {}, processor, lambda_reg=0.0, impute_lags=False)

target = config_vsense['Optimisation'][optimisation_type]['output_param']
print('target prev/curr', result[target + '_previous'], result[target + '_current'])
print('production prev/curr', result['PRODUCTIONTONNESPERHR_previous'], result['PRODUCTIONTONNESPERHR_current'])
print('unitcost prev/curr', result['UNITCOST LAKHS/THM_previous'], result['UNITCOST LAKHS/THM_current'])
print('free cp result snippet', {k: result[k] for k in list(result.keys())[:8]})
