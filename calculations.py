import numpy as np
from scipy.signal import butter


t1 = 104
t2 = 85
t3 = 88

t = np.array([t1, t2, t3])
print(t)
prev_kv = (t1 - t2) / (t2 + t1) * 100
kv = (t3 - t2) / (t3 + t2) * 100
sum_kv = prev_kv + kv

std_t = np.std(t)

print(f"prev_kv {prev_kv}")
print(f"kv {kv}")
print(f"sum_kv {sum_kv}")
print(f"std_t {std_t}")

# b, a = butter(2, 12, 'low', fs=250)
# bh, ah = butter(1, 0.08, 'high', fs=250)
# print(f"b {b}")
# print(f"a {a}")
# print(f"bh {bh}")
# print(f"ah {ah}")


