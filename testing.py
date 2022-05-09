import pandas as pd
import numpy as np
img_1 = pd.read_csv('feaSubEImg_1.csv', header=None, dtype=float)
img_2 = pd.read_csv('feaSubEImg_2.csv', header=None, dtype=float)

xImg_1 = img_1.transpose().to_numpy()
print(xImg_1)