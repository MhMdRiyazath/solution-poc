"""Download the raw Electricity file (321 clients, hourly kW) if missing."""
import sys, pathlib, urllib.request; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config as C
URL = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
if C.RAW_FILE.exists(): print("already there")
else: urllib.request.urlretrieve(URL, C.RAW_FILE); print("downloaded", C.RAW_FILE.stat().st_size // 10**6, "MB")
