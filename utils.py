import os
import json
import numpy as np
import torch

def _time2int(time: str) -> int:
  return int(time[:2]) * 60 + int(time[2:])

def _read_json(path: str, step_minute: int = 1) -> list:
  with open(path, 'r') as f:
    print(f'Reading {path}...')
    return [value['sbi'] for key, value in json.load(f).items() if value and _time2int(key) % step_minute == 0]

def read_data(head_folder: str):
  total_stops = []
  first_folder = sorted(os.listdir(f'./{head_folder}'))[0]

  for station in sorted(os.listdir(f'./{head_folder}/{first_folder}')):
    with open(f'./{head_folder}/{first_folder}/{station}', 'r') as f:
      total_stops.append(list(json.load(f).values())[0]['tot'])
  
  data = np.array([])
  for folder in sorted(os.listdir(f'./{head_folder}')):
    stationData = []
    for file in sorted(os.listdir(f'./{head_folder}/{folder}')):
      if not file.endswith('.json'): continue
      stationData.append(_read_json(f'{head_folder}/{folder}/{file}'))
    stationData = np.transpose(np.array(stationData))
    data = np.append(data, stationData)
  data = np.reshape(data, (-1, 112))
  return data, total_stops

def create_dataset(data: np.array, lookback: int):
  dataX = np.array([data[i:(i + lookback)] for i in range(len(data) - lookback)])
  dataY = data[lookback:]
  return torch.tensor(dataX), torch.tensor(dataY)
