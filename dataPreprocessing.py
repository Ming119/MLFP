# read data from url
# output the csv file

from typing import List
import os
import json
import datetime
import requests
import asyncio
import pandas as pd

dateURL: str = "https://github.com/hyusterr/html.2023.final.data/tree/release/release/_date_"
rawURL:  str = "https://raw.githubusercontent.com/hyusterr/html.2023.final.data/release"
snoURL:  str = f"{rawURL}/sno_test_set.txt"
dataURL: str = f"{rawURL}/release/_date_/_sno_.json"

snos = requests.get(snoURL).text.split("\n")[:-1]

def create_empty_dataset() -> None:
  with open("./dataset.csv", "w") as f:
    f.write("datetime,"+",".join(snos))

def dateRange(startDay: str, endDay: str = None) -> List[str]:
  d_start: datetime.date = datetime.datetime.strptime(startDay, "%Y%m%d")
  d_end:   datetime.date = datetime.datetime.strptime(endDay, "%Y%m%d") if endDay else datetime.datetime.today()

  return [(d_end - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range((d_end - d_start).days, 0, -1)]

async def read_total_stop() -> None:
  async def async_get_data(url: str, loop: asyncio.AbstractEventLoop):
    response = await loop.run_in_executor(None, requests.get, url)
    data = json.loads(response.text)

    for i, value in enumerate(data.values()):
      if value != {}:
        return value["tot"]

  if os.path.exists("./total_stop.csv"):
    print("Total stop data already exists, skiping...")
    return
  
  total_stop = pd.DataFrame(index=snos, columns=["total_stop"])
  date = datetime.datetime.today().strftime("%Y%m%d")

  tasks = []
  loop = asyncio.get_event_loop()
  for sno in snos:
    print(f"Reading {date}/{sno}...")
    tasks.append(async_get_data(dataURL.replace("_date_", date).replace("_sno_", sno), loop))
  
  for sno, ts in zip(snos, await asyncio.gather(*tasks)):
    total_stop.loc[sno] = ts

  total_stop.to_csv("./total_stop.csv")

async def read_ubike_data(dataframe: pd.DataFrame) -> None:
  def _check_data_completeness(dataframe: pd.DataFrame, date: str) -> bool:
    return dataframe.index.astype(str).str.contains(datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")).any()

  async def async_get_data(url: str, loop: asyncio.AbstractEventLoop) -> List[int]:
    response = await loop.run_in_executor(None, requests.get, url)
    data = json.loads(response.text)
    missing = []
    for time, value in data.items():
      if value == {}:
        missing.append(time)
      elif len(missing) > 0:
        for ms in missing: data[ms] = data[time]
        missing = []
    
    return [value["sbi"] if value != {} else None for value in data.values()]

  ubike_data = pd.DataFrame()
  
  for date in dateRange("20231002"):
    if _check_data_completeness(dataframe, date):
      print(f"Data {date} already exists, skiping...")
      continue

    if requests.get(dateURL.replace("_date_", date)).status_code != 200:
      print(f"Date {date} not found!")
      continue 
    
    indices = pd.DatetimeIndex(pd.date_range(date, periods=1440, freq="1T", tz="Asia/Taipei"))
    date_data = pd.DataFrame(index=indices, columns=snos)
    loop = asyncio.get_event_loop()
    tasks = []
    for sno in snos:
      print(f"Reading {date}/{sno}...")
      tasks.append(async_get_data(dataURL.replace("_date_", date).replace("_sno_", sno), loop))

    for sno, data in zip(snos, await asyncio.gather(*tasks)):
      date_data[sno] = data

    ubike_data = pd.concat([ubike_data, date_data], axis=0)
    dataframe = pd.concat([dataframe, date_data], axis=0)
    dataframe.to_csv("./dataset.csv")

  dataframe.bfill(inplace=True)
  dataframe.to_csv("./dataset.csv")

async def read_weather_data(dataframe: pd.DataFrame) -> None:
  # TODO: read weather data and append to dataframe
  pass

async def main() -> None:
  originalData = pd.read_csv("./dataset.csv", index_col=0)

  await read_total_stop()
  await read_ubike_data(originalData)
  await read_weather_data(originalData)

if __name__ == "__main__":
  if not os.path.exists("dataset.csv"):
    create_empty_dataset()
  
  asyncio.run(main())
