import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("class3/lunch_box.csv", delimiter=",",
                   index_col='datetime', parse_dates=True)

print(data.head())

plt.bar(data.index, data["y"])

# data["y"].hist(bins=30)

plt.show()
