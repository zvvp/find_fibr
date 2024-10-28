from functions import get_fname, get_start_time, parse_B_txt, get_coef_fibr, get_ranges_fibr, get_time_qrs, get_diff_time
from scipy.signal import medfilt
from datetime import datetime, time
import numpy as np


fname = get_fname()
start_time = get_start_time(fname)
print(fname)
# print(start_time)

r_pos, intervals, chars, forms = parse_B_txt()

n = 111
fintervals = medfilt(intervals, n)
coef_fibr = get_coef_fibr(intervals, chars)
coef_fibr[coef_fibr > 3000] = 3000
fcoef_fibr = medfilt(coef_fibr, n)

ranges_fibr = get_ranges_fibr(fintervals, fcoef_fibr, r_pos)

len_f = len(ranges_fibr[0])
sum_time = np.array([0, 0, 0])
for i in range(len_f):
    start = get_time_qrs(ranges_fibr[0][i], start_time)
    stop = get_time_qrs(ranges_fibr[1][i], start_time)
    diff = np.array(get_diff_time(start, stop))
    sum_time += diff
    time_start = np.array(start)
    time_stop = np.array(stop)
    if time_start[0] >= 24:
        time_start[0] = time_start[0] - 24
    if time_stop[0] >= 24:
        time_stop[0] = time_stop[0] - 24
    print(f"{time_start[0]}:{time_start[1]}:{time_start[2]}  {time_stop[0]}:{time_stop[1]}:{time_stop[2]}")
if sum_time[2] > 59:
    overflow_s = sum_time[2] // 60
    sum_time[1] += overflow_s
    sum_time[2] %= 60
if sum_time[1] > 59:
    overflow_m = sum_time[1] // 60
    sum_time[0] += overflow_m
    sum_time[1] %= 60
sum_fibr_time = f"{sum_time[0]}:{sum_time[1]}:{sum_time[2]}"
print(f"Всего эпизодов {len_f}, суммарное время фибриляции {sum_fibr_time}")