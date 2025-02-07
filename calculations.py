import numpy as np
from scipy.signal import butter
from scipy.stats import pearsonr


b, a = butter(3, 15, "lp", fs=250)
print(b)
print(a)

