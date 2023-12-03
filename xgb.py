from typing import List, Tuple
from utils import _read_json
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

total_stops: List[int] = []

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
  predt = predt.reshape(-1, dtrain.shape[1])
  tmp = (dtrain - predt)
  tds = dtrain / total_stops
  return -3 * (tmp / total_stops * np.abs(tmp)) * (np.abs(tds - 1/3) + np.abs(tds - 2/3))

def xgb_loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
  grad = gradient(predt, dtrain)
  hess = np.zeros_like(grad)
  return grad, hess

def daterange(start_date: str, end_date: str, step_mins: int = 1) -> List[str]:
  return pd.date_range(start_date, end_date, freq=f"{step_mins}T", tz="Asia/Taipei")[:-1]

def get_total_stops() -> List[int]:
  res = []
  date = sorted(os.listdir("./traindata3"))[0]
  for station in sorted(os.listdir(f"./traindata3/{date}")):
    if not station.endswith(".json"): continue
    with open(f"./traindata3/{date}/{station}", "r") as f:
      res.append(list(json.load(f).values())[0]["tot"])
  return res

def read_data(root: str, step_mins: int = 1) -> pd.DataFrame:
  df = pd.DataFrame()
  for date in sorted(os.listdir(root)):
    if date == "20231202": continue
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

  return np.array([weekday, minutes, hours, days, months]).T, df.values

total_stops =  get_total_stops()

train_df = read_data("traindata3", 20)
trainX, trainY = create_dataset(train_df)

model = xgb.XGBRegressor()
model.fit(trainX, trainY)
print(model.score(trainX, trainY))

model.save_model("xgb.model")

dr1 = daterange("20231021", "20231025", 20)
dr2 = daterange("20231204", "20231211", 20)
print(dr1.shape, dr2.shape)

test_date = pd.concat([pd.DataFrame(dr1), pd.DataFrame(dr2)], axis=0)

test_weekday = [x.dayofweek for x in test_date[0]]
test_minutes = [x.minute    for x in test_date[0]]
test_hours   = [x.hour      for x in test_date[0]]
test_days    = [x.day       for x in test_date[0]]
test_months  = [x.month     for x in test_date[0]]
test_years   = [x.year      for x in test_date[0]]

testX = np.array([test_weekday, test_minutes, test_hours, test_days, test_months]).T
predTest = model.predict(testX)

test_df = pd.DataFrame(predTest, index=test_date[0], columns=train_df.columns)
test_df[test_df < 0] = 0

stacked_df = test_df.stack()

flattened_df = pd.DataFrame(stacked_df.index.tolist(), columns=["datetime", "sno"])

flattened_df["sbi"] = stacked_df.values
flattened_df["id"] = flattened_df["datetime"].dt.strftime("%Y%m%d") + "_" + flattened_df["sno"].astype(str) + "_" + flattened_df["datetime"].dt.strftime("%H:%M")
flattened_df.set_index("id", inplace=True)
flattened_df.drop(["datetime", "sno"], axis=1, inplace=True)
flattened_df.to_csv("submission.csv")
print("Submission file saved!")
