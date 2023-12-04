import json
import requests
import datetime
import os

dateURL: str = "https://github.com/hyusterr/html.2023.final.data/tree/release/release/_date_"
rawURL:  str = "https://raw.githubusercontent.com/hyusterr/html.2023.final.data/release"
snoURL:  str = f"{rawURL}/sno_test_set.txt"
dataURL: str = f"{rawURL}/release/_date_/_sno_.json"

d_start: datetime.date = datetime.date(2023, 10, 2)
d_end:   datetime.date = datetime.date.today()

date: list = [(d_end - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range((d_end - d_start).days, 0, -1)]
sno:  list = requests.get(snoURL).text.split("\n")[:-1]

for d in date:
	if requests.get(dateURL.replace("_date_", d)).status_code != 200: continue
	if not os.path.exists(f"./data/{d}"): os.mkdir(f"./data/{d}")
	for s in sno:
		if os.path.exists(f"./data/{d}/{s}.json"): continue
		url = dataURL.replace("_date_", d).replace("_sno_", s)
		data = json.loads(requests.get(url).text)
		missing = []
		for i in range(1440):
			h = str(i // 60).zfill(2)
			m = str(i % 60).zfill(2)
			if data[f"{h}:{m}"] == {}:
				missing.append(f"{h}:{m}")
			elif len(missing) > 0:
				for ms in missing: data[ms] = data[f"{h}:{m}"]
				missing = []
		with open(f"./data/{d}/{s}.json", "w") as f:
			json.dump(data, f, indent=2)
			print(f"Saved {d}/{s}.json")

# validate
for date in os.listdir("./data"):
	for sno in os.listdir(f"./data/{date}"):
		currentFile = open(f"./data/{date}/{sno}", "r")
		data = json.load(currentFile)
		currentFile.close()

		change = False

		for i in range(1440):
			h = str(i // 60).zfill(2)
			m = str(i % 60).zfill(2)
			if data[f"{h}:{m}"] != {}: continue

			next_day = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:])) + datetime.timedelta(days=1)
			next_day = next_day.strftime("%Y%m%d")
			if os.path.exists(f"./data/{next_day}/{sno}"):
				with open(f"./data/{next_day}/{sno}", "r") as f:
					data[f"{h}:{m}"] = json.load(f)["00:00"]
					change = True
				
			else: print(f"Missing {date}/{sno} {h}:{m}")

		if change:
			with open(f"./data/{date}/{sno}", "w") as f:
				json.dump(data, f, indent=2)
				print(f"Saved {date}/{sno}")