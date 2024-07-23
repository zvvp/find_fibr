import numpy as np
from scipy.signal import butter


t1 = 101
t2 = 97
t3 = 104

print(f"t1 {t1}")
print(f"t2 {t2}")
print(f"t3 {t3}")

kv = (t3 - t2) / (t3 + t2) * 100

mean_d = np.mean([np.abs(t1 - t2), np.abs(t1 - t3), np.abs(t2 - t3)])

print(f"kv {kv}")
print(f"mean_d {mean_d}")

# b, a = butter(2, 11, 'low', fs=250)
# bh, ah = butter(1, 0.08, 'high', fs=250)
# print(f"b {b}")
# print(f"a {a}")
# print(f"bh {bh}")
# print(f"ah {ah}")


