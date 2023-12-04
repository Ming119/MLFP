from typing import List, Tuple, Dict
from utils import _read_json
import os
import json
import nvidia_smi
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

total_stops: np.array = None

def get_total_stops() -> np.array:
  res = []
  date = sorted(os.listdir("./traindata3"))[0]
  for station in sorted(os.listdir(f"./traindata3/{date}")):
    if not station.endswith(".json"): continue
    with open(f"./traindata3/{date}/{station}", "r") as f:
      res.append(list(json.load(f).values())[0]["tot"])
  return np.array(res)

def xgb_eval_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  y_true = y_true.reshape(y_pred.shape)
  tds = y_true / total_stops
  return np.mean(3 * np.abs((y_true - y_pred) / total_stops) * (np.abs(tds - 1/3) + np.abs(tds - 2/3)))

def xgb_objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  y_true = y_true.reshape(y_pred.shape)
  tds = y_true / total_stops
  uc_term = np.abs(tds - 1/3) + np.abs(tds - 2/3)

  sign = np.sign((y_true - y_pred) / total_stops)

  grad = -3 / total_stops * sign * uc_term
  hess = 3 / np.power(total_stops, 2) * uc_term
  grad = grad.flatten()
  hess = hess.flatten()

  return grad, hess

def daterange(start_date: str, end_date: str, step_mins: int = 1) -> List[str]:
  return pd.date_range(start_date, end_date, freq=f"{step_mins}T", tz="Asia/Taipei")[:-1]

def read_data(root: str, step_mins: int = 1) -> pd.DataFrame:
  df = pd.DataFrame()
  for date in sorted(os.listdir(root)):
    if date == "20231203": continue
    if date == "20231011": continue

    df_date = pd.DataFrame()
    indices = pd.date_range(date, periods=1440/step_mins, freq=f"{step_mins}T", tz="Asia/Taipei")

    for station in sorted(os.listdir(f"./{root}/{date}")):
      if not station.endswith(".json"): continue

      data = _read_json(f"./{root}/{date}/{station}", step_mins)
      df_date = pd.concat([df_date, pd.DataFrame(data, index=pd.DatetimeIndex(indices), columns=[station.split('.')[0]])], axis=1)

    df = pd.concat([df, df_date], axis=0)
  return df

def create_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
  weekday = [x.dayofweek for x in df.index]
  minutes = [x.minute for x in df.index]
  hours   = [x.hour   for x in df.index]
  days    = [x.day    for x in df.index]
  months  = [x.month  for x in df.index]
  years   = [x.year   for x in df.index]

  return np.array([weekday, minutes, hours, days, months, years]).T, df.values

def split_train_test(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
  sample_size = int(len(df) * ratio)
  train = df.sample(n=sample_size)
  test = df.drop(train.index)
  return train, test

def plot_hist(hist: list) -> None:
  import matplotlib.pyplot as plt
  plt.plot(hist["validation_0"]["xgb_eval_metric"], label="train loss")
  plt.plot(hist["validation_1"]["xgb_eval_metric"], label="test loss")
  plt.legend()
  plt.savefig("hist.png")

total_stops = get_total_stops()

data = read_data("data", 20)
train, test = split_train_test(data, 0.8)

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

model_params = {
  # "device": "cuda",
  "n_jobs": -1,
  "eval_metric": xgb_eval_metric,
  # "objective": xgb_objective,
  "objective": "reg:absoluteerror",

  "n_estimators": 512,
  "max_depth": 8,
  "min_child_weight": 5,
  "gamma": 0.1,
  "eta": 0.1,

  "subsample": 1,
  "colsample_bytree": 1,
  "colsample_bylevel": 1,
  "reg_alpha": 0,
  "reg_lambda": 1,
}

cv_params = {
  # "objective": [xgb_objective, "reg:absoluteerror", "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror"]
  # "n_estimators": np.linspace(256, 768, 9, dtype=int),
  # "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  # "min_child_weight": np.linspace(1, 10, 10, dtype=int),
  # "gamma": np.linspace(0, 1, 11),
  # "subsample": np.linspace(0, 1, 11),
  # "colsample_bytree": np.linspace(0.1, 1, 10),
  # "colsample_bylevel": np.linspace(0.1, 1, 10),
  # "reg_alpha": np.linspace(0, 1, 11),
  # "reg_lambda": np.linspace(0, 1, 11),
  # "eta": np.logspace(-2, 0, 10),
}

model = xgb.XGBRegressor(**model_params)
model.fit(trainX, trainY, eval_set=[(trainX, trainY), (testX, testY)])
print(model.evals_result())

# scorer = make_scorer(xgb_eval_metric, greater_is_better=False)
# gs = GridSearchCV(estimator=model, param_grid=cv_params, scoring=scorer, n_jobs=-1, verbose=2, cv=3)
# gs.fit(trainX, trainY)
# print(gs.best_params_)
# print(gs.best_score_)

hist = model.evals_result()
plot_hist(hist)