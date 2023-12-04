from typing import List
import os
import json
import requests
import numpy as np
import pandas as pd
import xgboost as xgb

def get_sno() -> List[int]:
  snoURL:  str = "https://raw.githubusercontent.com/hyusterr/html.2023.final.data/release/sno_test_set.txt"
  return requests.get(snoURL).text.split("\n")[:-1]

def get_total_stops() -> List[int]:
  res = []
  date = sorted(os.listdir("./traindata3"))[0]
  for station in sorted(os.listdir(f"./traindata3/{date}")):
    if not station.endswith(".json"): continue
    with open(f"./traindata3/{date}/{station}", "r") as f:
      res.append(list(json.load(f).values())[0]["tot"])
  return res

def daterange(start_date: str, end_date: str, step_mins: int = 1) -> pd.DataFrame:
  dr = pd.date_range(start_date, end_date, freq=f"{step_mins}T", tz="Asia/Taipei")[:-1]
  return pd.DataFrame(dr, columns=["datetime"])

def handle_extreme(df: pd.DataFrame) -> pd.DataFrame:
  total_stops = get_total_stops()
  df[df < 0] = 0
  df[df > total_stops] = total_stops
  return df

def create_testset(df: pd.DataFrame) -> np.ndarray:
  weekday = [x.dayofweek for x in df['datetime']]
  minutes = [x.minute for x in df['datetime']]
  hours   = [x.hour   for x in df['datetime']]
  days    = [x.day    for x in df['datetime']]
  months  = [x.month  for x in df['datetime']]
  years   = [x.year   for x in df['datetime']]

  return np.array([weekday, minutes, hours, days, months]).T

model = xgb.XGBRegressor()
model.load_model("xgb.json")

dr1 = daterange("20231021", "20231025", 20)
dr2 = daterange("20231204", "20231211", 20)
dr = pd.concat([dr1, dr2], axis=0)

test_date = create_testset(dr)
pred_test = model.predict(test_date)
pred_test = pd.DataFrame(pred_test, index=dr['datetime'], columns=get_sno())
pred_test = handle_extreme(pred_test)

stacked_df = pred_test.stack()

flattened_df = pd.DataFrame(stacked_df.index.tolist(), columns=["datetime", "sno"])

flattened_df["sbi"] = stacked_df.values
flattened_df["id"] = flattened_df["datetime"].dt.strftime("%Y%m%d") + "_" + flattened_df["sno"].astype(str) + "_" + flattened_df["datetime"].dt.strftime("%H:%M")
flattened_df.set_index("id", inplace=True)
flattened_df.drop(["datetime", "sno"], axis=1, inplace=True)
flattened_df.to_csv("submission.csv")
print("Submission file saved!")


