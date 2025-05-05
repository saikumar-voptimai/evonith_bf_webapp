import itertools
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict

from config import INIT_PARAMS as IPS


def run_optimiser(data_path: str,
                  model_path: str,
                  user_input: Dict,
                  prev_control_input: Dict,
                  control_params_list: List[str],
                  include_control: Dict,
                  num: int):
    """
    A wrapper function to run the BruteForceOptimiser. This is executed on pressuing "Run Optimiser" button.
    :param user_input: Input parameters entered in the number_input boxes.
    :param prev_control_input: Previous days control inputs. Used to calculate distance to optimal solutions.
    :param control_params_list: List of control parameters as strings.
    :param include_control:
    :return:
    """
    # Perform optimization to find the optimal set of control parameters
    optimiser = BruteForceOptim(data_path=data_path,
                                model_path=model_path,
                                control_params_list=control_params_list,
                                include_control=include_control)
    df_optim_comb, prev_y_val = optimiser.top_combinations(user_input, num=num,
                                               prev_control_params=list(prev_control_input.values()))
    disp_cols = control_params_list.copy()
    disp_cols.append("Efficiency")
    disp_cols.append("Distance to prev")
    df_disp = df_optim_comb[disp_cols].round(5).reset_index(drop=True)
    return df_disp, prev_y_val


def control_param_comb(df_bf: pd.DataFrame,
                       control_params: List,
                       include_control: Dict,
                       num: int = 10) -> List:
    """
    Generates the possible combination of control parameters (control_params) at steps prescribed by
    number of points (num).
    :param df_bf: Input dataframe to get the minimum and maximum values of the contro_params columns
    :param control_params: List of control_params that are adjusted to maximise efficiency
    :param num: Number of discrete to which each control parameter is discretised into.
    :return: List of control parameter combinations
    """
    # Define parameter ranges
    control_param_ranges = {}
    for i in range(len(control_params)):
        if control_params[i] not in df_bf.columns:
            raise Exception(f'Column {control_params[i]} not in DataFrame')
        # Find the min and max values of each "control_param"
        min_val = df_bf[control_params[i]].min()
        max_val = df_bf[control_params[i]].max()

        # Divide this range into "num" points when Override is not selected (value = nan if not selected):
        if include_control[control_params[i]] is np.nan:
            control_param_ranges[control_params[i]] = np.linspace(min_val, max_val, num=num)
        else:
            control_param_ranges[control_params[i]] = np.array([include_control[control_params[i]]])

    # Generate all parameter combinations as a list of tuples.
    param_combinations = list(itertools.product(*control_param_ranges.values()))
    return param_combinations


class BruteForceOptim:
    def __init__(self, data_path, model_path, control_params_list, include_control):
        self.control_params_list = control_params_list
        self.historical_data = pd.read_pickle(data_path)  # Load historical data
        self.model = joblib.load(model_path)   # Initialize your pre-trained XGBoost model here
        self.include_control = include_control # Has the values of Overriding control parameters

    def top_combinations(self, input_params: Dict,
                         num: int = 10,
                         prev_control_params: List = None) -> pd.DataFrame:
        """
        Top_combinations gets the user-specified input parameters Dict from streamlit, combines them with the
        various combinations of the control parameters, estimates the efficiency on all the points. The eucledian
        distance to the previous operating points is also specified. Note the prev_control_params can be a
        moving_average.
        :param input_params:
        :param num:
        :param prev_control_params:
        :return:
        """
        # TODO: Make the prev_control_params a moving average of the data points.
        # Generate all possible combinations of control parameters.
        control_param_combinations = control_param_comb(self.historical_data,
                                                        self.control_params_list,
                                                        self.include_control,
                                                        num=num)
        combinations = {key: None for key in self.control_params_list}

        # Make a dictionary of the "control params" combinations, keys as "parameter names" and values as combinations.
        # Ex: {"Sinter_usage": [60,62,...]..}.
        for i, key in enumerate(self.control_params_list):
            combinations[key] = [control_param_combinations[j][i] for j in range(len(control_param_combinations))]
        # Create an index key to the dictionary. This will be the index for dataframe
        combinations["index"] = [j for j in range(len(control_param_combinations))]

        # Convert the top_combinations into a dataframe.
        df_comb = pd.DataFrame.from_dict(combinations).set_index("index")

        # Similarly, create a dataframe for the user specified "input params". Note they are only one point vector.
        df = pd.DataFrame(input_params, index=[0])
        # Repeat the input params as many times as size of "control_params" combinations. Merge "input" & "control" dfs.
        df_input_params = df.loc[df.index.repeat(len(df_comb.index))].reset_index(drop=True)
        df_params = pd.concat([df_input_params, df_comb], axis=1)
        df_params = df_params[IPS.INPUT_PARAMS_MODEL]

        if df_params.columns.tolist() != IPS.INPUT_PARAMS_MODEL:
            raise Exception(f'DF column names {df_params.columns }\n dont match model requirements '
                            f'{IPS.INPUT_PARAMS_MODEL}')
        # Predict the efficiency by supplying the "full" (input + control) params to the model
        efficiency = self.model.predict(df_params)

        # Create a copy of the params dataframe and add a column "Efficiency" and attach the values.
        df_result = df_params.copy()
        df_result["Efficiency"] = efficiency
        df_result = df_result.sort_values(by="Efficiency", ascending=False)
        df_top_5 = df_result.head(10)

        # Find the Euclidean distance of each full param combination to the previous operating point
        if prev_control_params:
            prev_vector = np.array(prev_control_params)
            col_temp = df_top_5[IPS.CONTROL_PARAMS].apply(lambda row: np.linalg.norm(row - prev_vector), axis=1)
            df_top_5["Distance to prev"] = col_temp

        # Return the top combinations as a DataFrame
        df_top_5 = df_top_5.sort_values(by="Distance to prev", ascending=True)
        return df_top_5, efficiency[-1]


# Example usage:
if __name__ == "__main__":
    df = pd.DataFrame(data={"Param 1": [1, 5, 3, 2],
                            "Param 2": [-3, 4, 2, 8],
                            "Param 3": [-2, 35, 34, 68],
                            "Param 4": [745, 22, 45, 84]})
    control_pars = ["Param 1", "Param 2", "Param 4"]
    include_control = {"Param 1": np.nan, "Param 2": -3, "Param 4": np.nan}
    comb = control_param_comb(df, control_pars,include_control, num=5)
    top_comb = {key: None for key in control_pars}
    for i, key in enumerate(control_pars):
        top_comb[key] = [comb[j][i] for j in range(len(comb))]
    top_comb["index"] = [i for i in range(len(comb))]
    df_top = pd.DataFrame.from_dict(top_comb)
    df_top = df_top.set_index("index")
    effi = []
    for i in range(len(comb)):
        effi.append(np.random.rand())
    df_top["Efficiency"] = effi
    df_top = df_top.sort_values(by="Efficiency", ascending=False)

    # Return the top 5 combinations as a DataFrame
    df_top_five = df_top.head(5)
    print("Range")
    # historical_data_path = "historical_data.csv"  # Replace with your data file
    # optimizer = Optimizer(historical_data_path)
    # input_params = {"Input_Param1": 0.5, "Input_Param2": 0.3}  # Get input parameters from Streamlit
    # top_combinations = optimizer.optimize(input_params)
    # print(top_combinations)
