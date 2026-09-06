import numpy as np
import pandas as pd

from src.data_load import load_full_data
from src.model_2 import prepare_model2_data
from src.model_1 import fit_model1, predict_model1, save_model1

full_data = load_full_data()
long_data = prepare_model2_data(full_data, LF_number=1000)

fit_model1(long_data, HF_only=True)
predict_model1(long_data, HF_only=True)
save_model1("results/model2/full_fit/model2_SF.pkl")

fit_model1(long_data, HF_only=False)
predict_model1(long_data, HF_only=False)
save_model1("results/model2/full_fit/model2_MF.pkl")
