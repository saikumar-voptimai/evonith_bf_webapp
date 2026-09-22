import sys

sys.path[:0] = ["src", "furnace_data"]

from dotenv import load_dotenv

from data.ml.static_dataset_manager import StaticDatasetManager

load_dotenv()
dataset = StaticDatasetManager("src/assets/data/furnace_dataset.csv").update_static()
print("static_refresh_shape", dataset.shape)
print("static_refresh_end", dataset.index.max())
