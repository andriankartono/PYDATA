import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.error import HTTPError
import json
from pandas.core.reshape.concat import concat

with open('stations.json', encoding='utf-8') as f:
    d=json.load(f)
stations=pd.DataFrame.from_dict(d, orient='index')
stations.sort_index()
print(stations)