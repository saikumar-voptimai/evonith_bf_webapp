import json
import sys

import joblib
import pandas as pd

from apps.frontend_streamlit.utils.recommendations.optimiser import run_optimiser

if __name__ == "__main__":
    payload_path = sys.argv[1]
    with open(payload_path, "r") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["df"])
    models_dict = payload["models_dict"]

    # 🔥 Re-load models here (instead of JSON-passing them)
    for name, cfg in models_dict.items():
        model_path = cfg["model"]  # or cfg["model_path"]
        models_dict[name]["LoadedMLModel"] = joblib.load(model_path)

    user_input = payload["user_input"]
    fixed_cp = payload["fixed_cp"]
    lambda_reg = payload["lambda_reg"]

    result = run_optimiser(df, models_dict, user_input, fixed_cp, lambda_reg)
    print(json.dumps(result))
