import numpy as np
from scipy.signal import butter
from scipy.stats import pearsonr


ref_t = np.array([162, 105, 215, 162])
t = np.array([162, 105, 215, 162-30])

coef_cor = pearsonr(ref_t, t)[0]
print(coef_cor)  

