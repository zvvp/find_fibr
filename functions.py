from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelmax, argrelmin
from scipy.interpolate import interp1d
import glob
import os
from numba import njit
# from time import time
from functools import wraps
from timeit import default_timer


k = 8078
bl1, al1 = butter(2, 10.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
bl2, al2 = butter(2, 25.0, 'lp', fs=250)

b, a = butter(2, 18.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
bi, ai = butter(1, 3.0, 'lp', fs=250)

bh1, ah1 = butter(2, 1.0, 'hp', fs=250)  # 2, 1.0, 'hp', fs=250
bh2, ah2 = butter(2, 0.1, 'hp', fs=250)

def get_Q(fdir):
    try:
        os.remove(fdir + "/B1.txt")
    except FileNotFoundError:
        pass
    try:
        os.remove(fdir + "/F.txt")
    except FileNotFoundError:
        pass
    s = 0
    with open(fdir + "/B.txt", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if (i >= 14) and (i < len(lines) - 2):  # i > 13   i < len(lines) - 1
            if (';N' in lines[i]) or (';S' in lines[i]):  # and (not ';V' in lines[i - 1]) and (not ';S' in lines[i - 1]):
                periods = get_periods(lines[i - 2:i + 3])
                tf = np.array(periods)
                diff_tf = np.diff(tf)
                diff21 = diff_tf[2] - diff_tf[1]
                ref_t = np.array([tf[1], tf[1] * 0.6, tf[1] * 1.3, tf[1]])
                ref_t0 = np.array([tf[1], tf[1] * 0.6, tf[1], tf[1] * 0.6])
                ref_t1 = np.array([tf[1], tf[1] * 1.35, tf[1] , tf[1] * 1.35])
                # ref_t3 = np.array([tf[1], tf[1] * 1.6, tf[1] * 0.65, tf[1]])
                # ref_t4 = np.array([tf[1], tf[1] * 1.6, tf[1] * 0.5, tf[1]])
                coef_cor = get_coef_cor(ref_t, tf[1:])
                coef_cor0 = get_coef_cor(ref_t0, tf[1:])
                coef_cor1 = get_coef_cor(ref_t1, tf[1:])
                # coef_cor3 = get_coef_cor(ref_t3, tf[1:])
                # coef_cor4 = get_coef_cor(ref_t4, tf[1:])
                arr_cor = np.array([coef_cor, coef_cor0, coef_cor1])
                trs = 0.99  # 0.985
                if (np.max(arr_cor) > trs) or (diff21 > 160):
                    lines[i] = lines[i].replace(';N', ';Q')
                    s += 1
                # if ((coef_cor > trs) or (coef_cor0 > trs) or (coef_cor1 > trs) or (
                #         coef_cor3 > trs)) and (np.std(tf) > 50):  # or (coef_cor4 > trs):
                #     lines[i] = lines[i].replace(';N', ';Q')
                #     s += 1
    lines[6] = lines[6] + f"НЖ: {s}"
    with open(fdir + "/B1.txt", "w") as f:
        for i, line in enumerate(lines):
            f.write(line)

def parse_B1_txt(fdir):
    r_pos = []
    intervals = []
    chars = []
    forms = []
    with open(fdir + "/B1.txt", "r") as f:
        for line in f:
            if ';' in line:
                line_split = line.split(';')
                r_pos.append(int(line_split[0]))
                intervals.append(int(line_split[1]))
                chars.append(line_split[2][0])
                forms.append(int(line.split(':')[1]))
        r_pos = np.array(r_pos)
        intervals = np.array(intervals)
        chars = np.array(chars)
        forms = np.array(forms)
        intervals[intervals >= 500] = 500
    return r_pos, intervals, chars, forms

def del_V_S(intervals, chars):
    len_in = len(intervals)
    out = intervals.copy()
    diff_intervals = np.abs(intervals - np.roll(intervals, 1))[1:]
    for i in np.arange(5, len_in - 5):
        max_diff = np.max(diff_intervals[i:i + 2])
        mean_diff = np.mean(abs(diff_intervals[i - 1:i + 2])) # i - 2:i + 3
        mean_interval = np.mean(intervals[[i - 3,i - 2,i - 1, i + 2, i + 3]])
        if 'V' in chars[i] and 'V' in chars[i + 1] and (max_diff > 100.0):
            out[i] = mean_interval + (intervals[i] - mean_interval) * 0.2
        elif 'V' in chars[i] and not 'V' in chars[i + 1] and (max_diff > 40.0):
            out[i:i + 2] = mean_interval + (intervals[i:i + 2] - mean_interval) * 0.2
        elif 'Q' in chars[i] and (mean_diff > 30.0):
            out[i:i + 2] = mean_interval + (intervals[i:i + 2] - mean_interval) * 0.3
    return out

def del_artifacts(fintervals, intervals):
    out = fintervals.copy()
    for i in np.arange(1, len(intervals) - 3):
        if ((out[i] / ((out[i-1] + out[i+1]) / 2.0) > 1.9) and
                (abs(out[i-1] - out[i+1]) < 10)):
            out[i] = (fintervals[i-1] + fintervals[i+1]) / 2.0
        elif ((2.2 > (out[i-1] + out[i+2]) / (out[i] + out[i+1]) > 1.8) and
              (abs(out[i-1] - out[i+2]) < 15)):
            out[i] = fintervals[i+2]
            out[i+1] = fintervals[i-1]
        elif ((2.2 > (out[i-1] * 2 + out[i+3]) / (out[i] + out[i+1] + out[i+2]) > 1.8) and
              (abs(out[i-1] - out[i+3]) < 15)):
            out[i] = fintervals[i+3]
            out[i+1] = fintervals[i-1]
            out[i+2] = fintervals[i+3]

    return out



def time_fun(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure execution time using default_timer
        start_time = default_timer()
        result = func(*args, **kwargs)
        execution_time = default_timer() - start_time

        # Print elapsed time
        print(f"Время выполнения {func.__name__}: {execution_time:.9f} секунд")

        # Return the result of the decorated function
        return result

    return wrapper

def get_time_from_addr(line):
    fname = glob.glob("d:/Kp_01/*.ecg")[0]
    with open(fname, "rb") as f:
        f.seek(151)
        dlmt = f.read(1)
        if dlmt == b":":
            f.seek(150)
            start_h = int(f.read(1))
            f.seek(152)
            start_m = int(f.read(2))
            f.seek(155)
            start_s = int(f.read(2))
        else:
            f.seek(150)
            start_h = int(f.read(2))
            f.seek(153)
            start_m = int(f.read(2))
            f.seek(156)
            start_s = int(f.read(2))
    addr = int(line.split(';')[0])
    s = addr * 4 // 1000
    m = s // 60
    s = s % 60
    s = s + start_s
    if s >= 60:
        s = s - 60
        m = m + 1
    h = m // 60
    m = m % 60
    m = m + start_m
    if m >= 60:
        m = m - 60
        h = h + 1
    d = h // 24
    h = h % 24
    h = h + start_h
    if h >= 24:
        h = h - 24
        d = d + 1
    return f": {d + 1} день {h}:{m}:{s}\n"

def get_periods(lines):
    period_2 = int(lines[0].split(';')[1])
    period_1 = int(lines[1].split(';')[1])
    period = int(lines[2].split(';')[1])
    period1 = int(lines[3].split(';')[1])
    period2 = int(lines[4].split(';')[1])
    return period_2, period_1, period, period1, period2

# @time_fun
# def get_S():
#     s = 0
#     with open("C:/EcgVar/B.txt", "r") as f:
#         lines = f.readlines()
#     ref_t = np.array([200, 200, 100, 300, 200])  # 200, 160, 240, 200
#     ref_t1 = np.array([100, 300, 100, 300, 200])
#     ref_t2 = np.array([300, 200, 100, 300, 200])
#     ref_t3 = np.array([300, 100, 300, 100, 300])
#     ref_t4 = np.array([100, 300, 100, 300, 100])
#     for i, line in enumerate(lines):
#         if (i > 14) and (i < len(lines) - 2):  # i > 13   i < len(lines) - 1
#             form_1 = int(lines[i - 1].split(':')[1])
#             form = int(lines[i].split(':')[1])
#             if (';N' in line) and (not ';V' in lines[i - 1]) and (not ';S' in lines[i - 1]):
#                 periods = get_periods(lines[i - 2:i + 3])
#                 tf = np.array(periods)
#                 # t = np.array(periods[1:])
#                 max_t = np.max(tf)
#                 min_t = np.min(tf)
#                 mean_t = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
#                 if (min_t > 50) and (max_t < mean_t * 2) and ((max_t - min_t) > mean_t * 0.03):
#                     coef_cor = get_coef_cor(ref_t, tf)
#                     coef_cor1 = get_coef_cor(ref_t1, tf)
#                     coef_cor2 = get_coef_cor(ref_t2, tf)
#                     coef_cor3 = get_coef_cor(ref_t3, tf)
#                     coef_cor4 = get_coef_cor(ref_t4, tf)
#                     trs = 0.9787
#                     if (coef_cor > trs) or (coef_cor1 > trs) or (coef_cor2 > trs) or (coef_cor4 > trs):
#                         if (form == 0):
#                             lines[i] = lines[i].replace(';N', ';A')
#                             lines[i + 1] = lines[i + 1].replace(';N', ';A')
#                         elif (form_1 == 0):
#                             lines[i] = lines[i].replace(';N', ';A')
#                             lines[i - 1] = lines[i - 1].replace(';N', ';A')
#                         else:
#                             lines[i] = lines[i].replace(';N', ';S')
#                             s += 1
#                     elif coef_cor3 > trs:
#                         if (form == 0):
#                             lines[i] = lines[i].replace(';N', ';A')
#                             lines[i + 1] = lines[i + 1].replace(';N', ';A')
#                         elif (form_1 == 0):
#                             lines[i] = lines[i].replace(';N', ';A')
#                             lines[i - 1] = lines[i - 1].replace(';N', ';A')
#                         else:
#                             lines[i-1] = lines[i-1].replace(';N', ';S')
#                             s += 1
#                             lines[i + 1] = lines[i + 1].replace(';N', ';S')
#                             s += 1
#     lines[6] = lines[6] + f"НЖ: {s}"
#     with open("C:/EcgVar/B1.txt", "w") as f:
#         for i, line in enumerate(lines):
#             f.write(line)

def get_coef_cor(x: np.ndarray, y: np.ndarray) -> float:
    mean_x: float = np.mean(x)
    mean_y: float = np.mean(y)
    mean_xy: float = np.mean(x * y)
    std_x: float = np.std(x)
    std_y: float = np.std(y)
    if std_x * std_y:
        return (mean_xy - mean_x * mean_y) / (std_x * std_y)
    else:
        return 0

def get_S():
    try:
        os.remove("C:/EcgVar/B1.txt")
    except FileNotFoundError:
        pass
    try:
        os.remove("C:/EcgVar/F.txt")
    except FileNotFoundError:
        pass
    s = 0
    with open("C:/EcgVar/B.txt", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if (i >= 14) and (i < len(lines) - 2):  # i > 13   i < len(lines) - 1
            if (';N' in lines[i]):  # and (not ';V' in lines[i - 1]) and (not ';S' in lines[i - 1]):
                periods = get_periods(lines[i - 2:i + 3])
                tf = np.array(periods)
                ref_t = np.array([tf[1], tf[1] * 0.8, tf[1] * 1.2, tf[1]])
                ref_t0 = np.array([tf[1], tf[1] * 0.65, tf[1] * 1.15, tf[1]])
                ref_t1 = np.array([tf[1], tf[1] * 0.65, tf[1] * 1.1, tf[1] * 0.65])
                ref_t3 = np.array([tf[1], tf[1] * 1.6, tf[1], tf[1] * 1.6])
                ref_tA = np.array([tf[1], tf[1] * 0.5, tf[1] * 0.5, tf[1]])
                ref_tA1 = np.array([tf[1], tf[1] * 0.65, tf[1] * 0.35, tf[1]])
                ref_tA2 = np.array([tf[1], tf[1] * 0.35, tf[1] * 0.65, tf[1]])
                mean_t = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
                # if (tf[2] > 50) and (tf[3] > 50) and (tf[2] < mean_t * 2) and (tf[3] < mean_t * 2):
                if True:
                    coef_cor = get_coef_cor(ref_t, tf[1:])
                    coef_cor0 = get_coef_cor(ref_t0, tf[1:])
                    coef_cor1 = get_coef_cor(ref_t1, tf[1:])
                    coef_cor3 = get_coef_cor(ref_t3, tf[1:])
                    coef_corA = get_coef_cor(ref_tA, tf[1:])
                    coef_corA1 = get_coef_cor(ref_tA1, tf[1:])
                    coef_corA2 = get_coef_cor(ref_tA2, tf[1:])
                    trs = 0.975  # trs = 0.985
                    if (coef_corA > 0.95) or (coef_corA1 > 0.95) or (coef_corA2 > 0.95):
                        if (tf[3] + tf[2]) < tf[1] * 1.1:
                            lines[i] = lines[i].replace(';N', ';A')
                            lines[i + 1] = lines[i + 1].replace(';N', ';A')
                    elif (coef_cor > trs) or (coef_cor0 > trs) or (coef_cor1 > trs):
                        if (tf[3] - tf[2]) > tf[1] * 0.06:
                            lines[i] = lines[i].replace(';N', ';S')
                            s += 1
                    elif coef_cor3 > trs:
                        if (tf[3] - tf[2]) > tf[1] * 0.06:
                            lines[i + 1] = lines[i + 1].replace(';N', ';S')
                            s += 1
                else:
                    lines[i] = lines[i].replace(';N', ';A')
                    lines[i + 1] = lines[i + 1].replace(';N', ';A')
    lines[6] = lines[6] + f"НЖ: {s}"
    with open("C:/EcgVar/B1.txt", "w") as f:
        for i, line in enumerate(lines):
            f.write(line)

def get_max_p(fragment):
    diff_loc = get_diff_loc(fragment)
    if (diff_loc.size == 0) or (diff_loc.size > 2):
        return 0
    else:
        return np.max(diff_loc)
        # return 0.1

def get_number_of_peaks1(fragment):
    diff_loc = get_diff_loc(fragment)
    if (diff_loc.size == 0) or (diff_loc.size > 1):
        return 0
    else:
        return 1
    # max_diff = np.max(diff_loc)
    # if max_diff > 0.001:
    #     return 1
    # else:
    #     return 0

def get_diff_loc(fragment):
    num_max = argrelmax(fragment)[0]
    if num_max.size == 0:
        return np.array([])
    else:
        num_min = argrelmin(fragment)[0]
        ind_max_min = np.sort(np.concatenate((num_max, num_min)))
        loc_extrem = fragment[ind_max_min]
        diff_loc = (loc_extrem - np.roll(loc_extrem, 1))[1:]
        diff_loc = diff_loc[diff_loc > 0.001]
        if diff_loc.size == 0:
            return np.array([])
        else:
            return diff_loc

def del_V_A(intervals, chars):
    len_in = len(intervals)
    out = intervals.copy()
    out1 = del_artifacts(out)
    diff_out1 = np.abs(out1 - np.roll(out1, 1))[1:]
    for i in np.arange(5, len_in - 5):
        max_diff = np.max(diff_out1[i:i + 2])
        mean_diff = np.mean(abs(diff_out1[i - 1:i + 2]))
        mean_interval = np.mean(out1[i - 5:i + 5])
        if 'V' in chars[i] and 'V' in chars[i + 1] and (max_diff > 100.0):
            out1[i] = mean_interval + (out1[i] - mean_interval) * 0.2
        elif 'V' in chars[i] and not 'V' in chars[i + 1] and (max_diff > 40.0):
            out1[i:i + 2] = mean_interval + (out1[i:i + 2] - mean_interval) * 0.2
        elif 'Q' in chars[i] and (max_diff > 60.0):
            out1[i:i + 2] = mean_interval + (out1[i:i + 2] - mean_interval) * 0.3
    return out1

def get_diff_intervals(intervals, step):
    diff_intervals = intervals.copy()
    diff_intervals[step:] = np.abs(diff_intervals[step:] - diff_intervals[:-step])
    diff_intervals[:step] = diff_intervals[step]
    diff_intervals[-step:] = diff_intervals[-(step + 1)]
    return diff_intervals

def get_disp_coef1(intervals):
    len_in = len(intervals)
    out = np.zeros(len_in)
    diff1 = get_diff_intervals(intervals, 1)
    diff2 = get_diff_intervals(intervals, 2)
    diff3 = get_diff_intervals(intervals, 3)
    diff4 = get_diff_intervals(intervals, 4)
    for i in np.arange(30, len_in - 30):
        win_diff1 = diff1[i - 30:i + 31]
        win_diff2 = diff2[i - 30:i + 31]
        win_diff3 = diff3[i - 30:i + 31]
        win_diff4 = diff4[i - 30:i + 31]
        mul_diff = (win_diff1 * win_diff2 + win_diff1 * win_diff3 + win_diff1 * win_diff4 + win_diff2 * win_diff3 +
                    win_diff2 * win_diff4 + win_diff3 * win_diff4) ** 0.5

        # mul_diff = (win_diff1 ** 2 + win_diff2 ** 2 + win_diff3 ** 2 + win_diff4 ** 2) ** 0.5  # ** 0.25 win_diff2 * win_diff3 * win_diff4) ** 0.5
        # mul_diff = (win_diff1 * win_diff2 * win_diff3 * win_diff4) ** 0.5  # ** 0.25
        sort_mul_diff = np.sort(mul_diff)#[:-15]
        # out[i] = sort_mul_diff[30] * 0.28
        out[i] = np.mean(sort_mul_diff[5:-5]) * 0.35
        bp = 0
        # out[i] = np.mean(sort_mul_diff[16:-24]) * 1.5
        # out[i] = np.mean(sort_mul_diff[30:-10]) * 1.0
    out[:30] = out[30]
    out[-30:] = out[-31]
    return out

def count_trans(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    count = 0
    for i in range(len(vec1) - 1):
        if vec1[i] < vec2[i] and vec1[i + 1] > vec2[i + 1]:
            count += 1
    if vec1[0] > vec2[0]:
        count += 1
    return count

def get_fibr_num_samples(e_coef_p, intervals, mask_pseudo_fibr):
    start_ind = 0
    stop_ind = 0
    start_ind_arr = np.array([])
    stop_ind_arr = np.array([])
    if e_coef_p[0] > intervals[0]:
        start_ind_arr = np.append(start_ind_arr, start_ind)
        flag = True
    else:
        flag = False
    min_size = 22
    for i in np.arange(e_coef_p.size - 1):
        if flag == False and e_coef_p[i] < intervals[i] and e_coef_p[i + 1] > intervals[i + 1] and mask_pseudo_fibr[i] == 1:
            if i - stop_ind > min_size:
                start_ind = i
                start_ind_arr = np.append(start_ind_arr, start_ind)
            else:
                if stop_ind_arr.size > 0:
                    stop_ind_arr = np.delete(stop_ind_arr, -1)
            flag = True
        elif flag == True and e_coef_p[i] > intervals[i] and e_coef_p[i + 1] < intervals[i + 1] and mask_pseudo_fibr[i] == 1:
            if i - start_ind > min_size:
                stop_ind = i
                stop_ind_arr = np.append(stop_ind_arr, stop_ind)
            else:
                if start_ind_arr.size > 0:
                    start_ind_arr = np.delete(start_ind_arr, -1)
            flag = False
        elif flag == True and i == e_coef_p.size - 2:
            if i - start_ind > min_size:
                stop_ind = i + 1
                stop_ind_arr = np.append(stop_ind_arr, stop_ind)
            else:
                if start_ind_arr.size > 0:
                    start_ind_arr = np.delete(start_ind_arr, -1)
    diff_stop_start = stop_ind_arr - start_ind_arr
    return start_ind_arr.astype(int), stop_ind_arr.astype(int), diff_stop_start.astype(int)


def get_scatter_coef(intervals):
    # win_t = intervals[:50].copy()
    # mean_win = np.mean(win_t)
    # diff_t = np.abs(win_t - np.roll(win_t, 1))[1:]
    # diff_t2 = np.abs(diff_t - np.roll(diff_t, 1))[1:]
    # diff_t2 = np.sort(diff_t2)[:-12]
    # start_value = np.mean(diff_t2) * (0.8 + 350 / mean_win)  #  np.mean(diff_t2) * (1.8 + 350 / mean_win)
    # out[i-5] = np.mean(out[i-10:i+1])
    len_in = len(intervals)
    out = np.zeros(len_in)
    # out = np.ones(len_in) * start_value
    for i in np.arange(30, len_in - 31):  # np.arange(25, len_in - 26) 30 31
        win_t = intervals[i - 30:i + 31].copy()
        # sort_win_t = np.sort(win_t)[:-10]   # [:-12]
        # mean_win = np.median(sort_win_t)
        diff_t = np.abs(win_t - np.roll(win_t, 1))
        diff_t2 = np.abs(diff_t - np.roll(diff_t, 1))
        diff_t2 = np.abs(diff_t2 - np.roll(diff_t2, 1))
        start_ind = int(diff_t2.size // 2.2)
        stop_ind = int(diff_t2.size - start_ind)
        diff_t2 = np.sort(diff_t2)[start_ind:stop_ind]
        # diff_t2 = np.sort(diff_t2)[:-12]   # [:-12]
        dt2_mean = np.mean(diff_t2)
        # if diff_t2[diff_t2 > dt2_mean].size > 0:
        #     threshold = np.mean(diff_t2[diff_t2 > dt2_mean])
        #     diff_t2 = diff_t2[diff_t2 < threshold]
        # mean_diff_t2 = diff_t2.std() * (1.5 + 200 / mean_win)  # (1.9 + 210 / mean_win)
        out[i] = dt2_mean #* (1 + 100.0 / mean_win) # dt2_mean * 175.0 / mean_win
        tempvar = 0
        # out[i-2] = np.mean(out[i-5:i+1])
    # out[:35] = np.mean(out[30:50])
    # out[-36:] = np.mean(out[-50:-30])
    out[:30] = out[30]
    out[-30:] = out[-30]
    # out[out < 0] = 0
    return out

def get_coef_fibr(intervals):
    len_in = len(intervals)
    # mean_interval = np.mean([intervals])
    out = np.zeros(len_in)
    for i in np.arange(15, len_in - 16):
        win_t = intervals[i - 15:i + 16].copy()
        win_t = np.sort(win_t)
        mean_win = np.median(win_t)
        diff_t = np.abs(win_t - np.roll(win_t, 1))
        diff_t = np.sort(diff_t)
        # mean_diff = np.mean(diff_t[1:-7])
        out[i] = np.mean(diff_t[10:-10])
        # out[i] = np.mean(diff_t[:-5])
        # out[i] = mean_diff * (1.0 + 150 / mean_win * 1.0)  #  np.mean(diff_t[10:-2]) * (1 + 150 / mean_win)
    # out = medfilt(out, 51)
    # out = truncate_win1(out, 0.2, 20)  #  0.2, 30
    out = moving_average(out, 20) * 10.0

    return out


def get_ranges_fibr(fintervals, fcoef_fibr, r_pos):
    start = []
    stop = []
    temp = 0
    w = 7500  # 2150
    for i in range(1, len(fintervals)):
        if (fcoef_fibr[i - 1] <= fintervals[i - 1]) and (fcoef_fibr[i] > fintervals[i]):
            if len(start) == 0:
                start.append(r_pos[i])
                temp = r_pos[i]
                continue
            if (r_pos[i] - temp) >= w:
                start.append(r_pos[i])
                temp = r_pos[i]
            elif len(stop) > 0:
                stop.pop(-1)
                if len(stop) > 0:
                    temp = stop[-1]
        elif (fcoef_fibr[i - 1] >= fintervals[i - 1]) and (fcoef_fibr[i] < fintervals[i]):
            if len(stop) == 0:
                stop.append(r_pos[i])
                temp = r_pos[i]
                continue
            if (r_pos[i] - temp) >= w:
                stop.append(r_pos[i])
                temp = r_pos[i]
            elif len(start) > 0:
                start.pop(-1)
                if len(start) > 0:
                    temp = start[-1]

    return start, stop


# @time_fun
@njit
def moving_average(data, window_size):
    half_win = window_size // 2
    mean_data = np.mean(data[:window_size])
    out = np.ones(len(data)) * mean_data
    for i in range(half_win, len(data) - half_win):
        out[i] = np.mean(data[i - half_win:i + half_win])
    out[:half_win] = out[half_win]
    out[-half_win:] = out[-half_win]
    return out

def step_moving_average(data, window_size):
    out = np.ones(len(data)) * np.mean(data)
    hf_w_size = window_size // 2
    for i in range(hf_w_size, len(data) - hf_w_size, window_size):
        buff = data[i - hf_w_size:i + hf_w_size]
        mean_buff = np.mean(buff)
        out[i - hf_w_size:i + hf_w_size] = mean_buff
    return out

# @time_fun
@njit
def truncate_win2(ch, k, win_size):
    in_ch = ch.copy()
    out = ch.copy()
    half_win = win_size // 2
    for i in range(half_win, len(in_ch) - half_win, half_win // 2):
        buff = out[i - half_win:i + half_win]
        # buff = in_ch[i - half_win:i + half_win]
        mean_buff = np.mean(buff)
        if len(buff[buff >= mean_buff]) > 0:
            over_mean = np.mean(buff[buff >= mean_buff])
            if len(buff[buff > over_mean]) > 0:
                buff[buff > over_mean] = (buff[buff > over_mean] - over_mean) * k + over_mean
        if len(buff[buff < mean_buff]) > 0:
            under_mean = np.mean(buff[buff < mean_buff])
            if len(buff[buff < under_mean]) > 0:
                buff[buff < under_mean] = (buff[buff < under_mean] - under_mean) * k + under_mean
        out[i - half_win:i + half_win] = buff
    out[:half_win] = out[half_win]
    out[-half_win:] = out[-half_win]
    return out


def truncate_ch(ch, k):
    mean_ch = np.mean(ch)
    return mean_ch + (ch - mean_ch) * k

def magnific_ch(ch, k):
    mean_ch = np.mean(ch)
    over_mean = np.mean(ch[ch > mean_ch])
    # over_mean = np.mean(ch[ch > over_mean])
    under_mean = np.mean(ch[ch < mean_ch])
    # under_mean = np.mean(ch[ch < under_mean])
    half_ch = over_mean - under_mean
    # print(f"half_ch = {half_ch}")
    out = half_ch + (ch - half_ch) * k
    out[out < 0.0] = 0.0
    return out


def get_under_threshold(ch):
    mean_ch = np.mean(ch)
    return np.mean(ch[ch < mean_ch])


def get_over_threshold(ch):
    mean_ch = np.mean(ch)
    return np.mean(ch[ch > mean_ch])


def interp_pr(ch):
    out = ch.copy()
    non_zero_indices = np.nonzero(out)[0]
    if len(non_zero_indices) == 0:
        return out
    interp_func = interp1d(non_zero_indices, out[non_zero_indices], kind='linear', fill_value='extrapolate')
    out[out == 0] = interp_func(np.where(out == 0)[0])
    return out

def get_attr(ch, len_win):
    # ch = ch.copy()
    len_ch = len(ch)
    # out = np.zeros(len_ch)
    mean_ch = np.zeros(len_ch)
    over_mean = np.zeros(len_ch)
    under_mean = np.zeros(len_ch)
    for i in range(len_win, len_ch - len_win):
        win = ch[i - len_win:i + len_win].copy()
        mean_ch[i] = np.mean(win)
        if len(win[win >= mean_ch[i]]) > 0:
            over_mean[i] = np.mean(win[win >= mean_ch[i]])
        if len(win[win < mean_ch[i]]) > 0:
            under_mean[i] = np.mean(win[win < mean_ch[i]])
    return mean_ch, over_mean, under_mean

def get_win_std(ch, len_win):
    out = np.zeros(len(ch))
    len_ch = len(ch)
    half_win = len_win // 2
    for i in range(half_win, len_ch - half_win):
        win = ch[i - half_win:i + half_win].copy()
        out[i] = np.std(win)
    return out

def interp_min_loc(min_loc, ind_min_loc):
    out = np.array([])
    for i in range(0, min_loc.size - 1):
        diff_val = min_loc[i + 1] - min_loc[i]
        count_step = int(ind_min_loc[i + 1] - ind_min_loc[i])
        step_val = diff_val / count_step
        for j in range(count_step):
            out = np.append(out, min_loc[i] + step_val * j)
    out = np.append(out, min_loc[-1])
    return out
# bi, ai = butter(1, 0.1, 'low', fs=250)
def get_p_amp1(fragment, max_amp_p, min_amp_p):
    pzub = 0.0
    amp_pzub = 0.0
    ind_p = 0
    ind_max_loc = argrelmax(fragment)[0]
    if (ind_max_loc.size > 0) and (ind_max_loc.size <= 3):
        # ind_min_loc = argrelmin(fragment)[0]
        # ind_min_loc = np.append(ind_min_loc, fragment.size-1)
        # ind_min_loc = np.append(0, ind_min_loc)
        # min_loc = fragment[ind_min_loc]
        # isoline = interp_min_loc(min_loc, ind_min_loc)
        # isoline = filtfilt(bi, ai, fragment)
        # fragment = fragment - isoline
        # plotter.plot_fragment(fragment, k)
        amp_pzub = np.max(fragment[ind_max_loc])
        ind_max = np.argmax(fragment[ind_max_loc])
        ind_p = ind_max_loc[ind_max]
    if 5.0 * max_amp_p > amp_pzub > 1.35 * min_amp_p:
        pzub = 1.0
    # else:
    #     amp_pzub = 0.0
    #     ind_p = 0
    return pzub, amp_pzub, ind_p

def get_amp_pos_p(fragment):
    isoline = filtfilt(bi, ai, fragment)
    fragment = filtfilt(b,a, fragment)
    fragment = fragment - isoline
    over_fragment = fragment.copy()
    over_fragment[over_fragment <= 0.0] = 0.0
    under_fragment = fragment.copy()
    under_fragment[under_fragment > 0.0] = 0.0
    amp_pzub = 0.0
    ind_p = 0
    ind_max_loc = argrelmax(over_fragment)[0]
    if len(ind_max_loc) == 0:
        return amp_pzub, ind_p
    ind_min_loc = argrelmin(under_fragment)[0]
    # if (len(ind_max_loc) == 0) and (len(ind_min_loc) == 0):
    #     return amp_pzub, ind_p
    arr_max_loc = fragment[ind_max_loc]
    if 3 >= arr_max_loc.size > 0:
        ind_max = np.argmax(arr_max_loc)
        ind_p = ind_max_loc[ind_max]
        if ind_min_loc.size > 0:
            ind_arr_loc = np.append(ind_min_loc, ind_p)
            ind_arr_loc = np.sort(ind_arr_loc)
            arr_loc = fragment[ind_arr_loc]
            max_index = np.argmax(arr_loc)
            if max_index == 0:
                # amp_pzub = arr_loc[0] - arr_loc[1]
                amp_pzub = arr_loc[0]
            elif max_index == arr_loc.size - 1:
                # amp_pzub = arr_loc[-1] - arr_loc[-2]
                amp_pzub = arr_loc[-1]
            else:
                side1 = arr_loc[max_index] - arr_loc[max_index - 1]
                side2 = arr_loc[max_index] - arr_loc[max_index + 1]
                if side1 < side2:
                    amp_pzub = side1
                else:
                    amp_pzub = side2
        else:
            amp_pzub = fragment[ind_p]
            # side1 = fragment[ind_p] - fragment[0]
            # side2 = fragment[ind_p] - fragment[-1]
            # if side1 < side2:
            #     amp_pzub = side1
            # else:
            #     amp_pzub = side2

    return amp_pzub, ind_p

def get_p_amp(fragment, max_amp_p, min_amp_p):
    pzub = 0.0
    amp_pzub = 0.0
    amp_pzub1 = 0.0
    amp_pzub2 = 0.0
    ind_p = 0
    ind_max_loc = argrelmax(fragment)[0]
    if ind_max_loc.size == 0:
        return pzub, amp_pzub, ind_p
    ind_min_loc = argrelmin(fragment)[0]
    if (ind_max_loc.size == 0) and (ind_min_loc.size == 0):
        return pzub, amp_pzub, ind_p
    ind_loc = np.concatenate((ind_max_loc, ind_min_loc))
    ind_loc = np.sort(ind_loc)
    loc_arr = fragment[ind_loc]
    loc_arr = np.append(loc_arr, fragment[-1])
    loc_arr = np.append(fragment[0], loc_arr)
    diff_loc = loc_arr[1:] - loc_arr[:-1]
    abs_diff_loc = np.abs(diff_loc)
    if ind_max_loc.size == 1:
        ind_p = ind_max_loc[0]
        if ind_min_loc.size == 0:
            amp_pzub = np.min(abs_diff_loc)
        elif ind_min_loc.size == 1:
            if ind_max_loc[0] < ind_min_loc[0]:
                amp_pzub = np.min(abs_diff_loc[:2])
            elif ind_max_loc[0] > ind_min_loc[0]:
                amp_pzub = np.min(abs_diff_loc[1:])
        elif ind_min_loc.size == 2:
            amp_pzub = np.min(abs_diff_loc[1:3])
    elif ind_max_loc.size == 2:
        if ind_min_loc.size == 1:
            amp_pzub1 = np.min(abs_diff_loc[:2])
            amp_pzub2 = np.min(abs_diff_loc[2:])
        elif ind_min_loc.size == 2:
            if ind_max_loc[0] < ind_min_loc[0]:
                amp_pzub1 = np.min(abs_diff_loc[:2])
                amp_pzub2 = np.min(abs_diff_loc[2:4])
            elif ind_max_loc[0] > ind_min_loc[0]:
                amp_pzub1 = np.min(abs_diff_loc[1:3])
                amp_pzub2 = np.min(abs_diff_loc[3:])
        elif ind_min_loc.size == 3:
            amp_pzub1 = np.min(abs_diff_loc[1:3])
            amp_pzub2 = np.min(abs_diff_loc[3:5])
        if amp_pzub1 > amp_pzub2:
            amp_pzub = amp_pzub1
            ind_p = ind_max_loc[0]
        else:
            amp_pzub = amp_pzub2
            ind_p = ind_max_loc[1]
        # max_amp = np.max([amp_pzub1, amp_pzub2])
        # amp_pzub = max_amp
    # elif ind_max_loc.size == 3:
    #     pzub = 0.5
    #     amp_pzub = 0.05
    #     ind_p = int(fragment.size * 0.6)
    #     return pzub, amp_pzub, ind_p
    # pos = k * 3 - 3
    # num = int(4 * 12)
    # printer.print_var(mean_amp_p * 0.1, pos, num)
    # printer.print_var(amp_pzub, pos, num)
    if 5.0 * max_amp_p > amp_pzub > 0.1 * min_amp_p:
        pzub = 1.0
    # printer.print_var(ind_max_loc.size, pos, num)
    # printer.print_var(ind_min_loc.size, pos, num)
    return pzub, amp_pzub, ind_p

def win_sum_coef(arr_coef, clean_intervals, len_win):
    out = np.zeros(len(arr_coef))
    len_ch = len(arr_coef)
    half_win = len_win // 2
    diff_c_i = arr_coef - clean_intervals
    diff_c_i[diff_c_i < 0.0] = 0.0
    for i in range(half_win, len_ch - half_win):
        win = diff_c_i[i - half_win:i + half_win].copy()
        out[i] = np.sum(win)
    out[:half_win] = out[half_win]
    out[-half_win:] = out[-half_win - 1]
    return out

def get_inds_min_diff(intervals, chars, forms):
    """
    Возвращает индексы интервалов, где абсолютная разница между
    последовательными интервалами меньше или равна 3.

    Параметры:
    intervals (numpy.ndarray): 1D numpy массив интервалов.

    Возвращает:
    numpy.ndarray: 1D numpy массив индексов, где абсолютная разница
    между последовательными интервалами меньше или равна 3.
    """
    inds_min_diff = np.array([], dtype=int)
    diff_intervals = np.abs(intervals - np.roll(intervals, -1))
    for i in range(1, diff_intervals.size):
        min_diff = intervals[i] * 0.1 # 0.05
        if (diff_intervals[i] <= min_diff) and (chars[i] == "N") and (chars[i-1] == "N") and (forms[i] == forms[i - 1]):
            inds_min_diff = np.append(inds_min_diff, i)
    return inds_min_diff

def plot_select_p(lead1, lead2, lead3, intervals, r_pos, mean_PR1, mean_PR2, mean_PR3, ind):
    """
    Строит графики выбранных фрагментов трех каналов и их отфильтрованные версии.

    Параметры:
    lead1 (numpy.ndarray): 1D numpy массив данных lead1.
    lead2 (numpy.ndarray): 1D numpy массив данных lead2.
    lead3 (numpy.ndarray): 1D numpy массив данных lead3.
    intervals (numpy.ndarray): 1D numpy массив интервалов.
    r_pos (numpy.ndarray): 1D numpy массив позиций r.
    ind (int): Индекс интервала для построения.

    Возвращает:
    None
    """
    # len_pr = int(intervals[ind] ** 0.5 * 3.7)
    # start = r_pos[ind] - len_pr
    # stop = r_pos[ind] - int(intervals[ind] ** 0.5 * 0.66 + 3.0)
    stop = r_pos[ind] - 10
    start1 = r_pos[ind] - mean_PR1[ind] - 15
    start2 = r_pos[ind] - mean_PR2[ind] - 15
    start3 = r_pos[ind] - mean_PR3[ind] - 15
    # stop1 = r_pos[ind] - mean_PR1[ind] + int(mean_PR1[ind] ** 0.5 * 3.0)
    # stop2 = r_pos[ind] - mean_PR2[ind] + int(mean_PR2[ind] ** 0.5 * 3.0)
    # stop3 = r_pos[ind] - mean_PR3[ind] + int(mean_PR3[ind] ** 0.5 * 3.0)
    fragment1 = lead1[start1:stop]
    fragment2 = lead2[start2:stop]
    fragment3 = lead3[start3:stop]

    p = pg.plot()
    p.showGrid(x=True, y=True)
    p.setTitle('p')
    p.plot(fragment1, pen='g')
    p.plot(fragment2-0.3, pen='g')
    p.plot(fragment3-0.6, pen='g')

    ffragment1 = filtfilt(b, a, fragment1)
    ffragment2 = filtfilt(b, a, fragment2)
    ffragment3 = filtfilt(b, a, fragment3)
    isoline1 = filtfilt(bi, ai, fragment1)
    isoline2 = filtfilt(bi, ai, fragment2)
    isoline3 = filtfilt(bi, ai, fragment3)
    fragment1 = ffragment1 - isoline1
    fragment2 = ffragment2 - isoline2
    fragment3 = ffragment3 - isoline3
    fragment1[fragment1 < 0.0] = 0.0
    fragment2[fragment2 < 0.0] = 0.0
    fragment3[fragment3 < 0.0] = 0.0

    p.plot(fragment1, pen='r')
    p.plot(fragment2-0.3, pen='r')
    p.plot(fragment3-0.6, pen='r')
    # p02.plot(lead1[r_pos[ind] - 1600:r_pos[ind] + 1600], pen='g')
    # p02.plot(lead2[r_pos[ind] - 1600:r_pos[ind] + 1600] - 2, pen='y')
    # p02.plot(lead3[r_pos[ind] - 1600:r_pos[ind] + 1600] - 4, pen='c')

@time_fun
def get_p_pos(lead1, lead2, lead3, intervals, r_pos, chars, inds_min):
    # global bl, al, k
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    # pos_p1 = 0
    # pos_p2 = 0
    # pos_p3 = 0
    presence_PR1 = np.zeros(r_pos.size, dtype=int)
    presence_PR2 = np.zeros(r_pos.size, dtype=int)
    presence_PR3 = np.zeros(r_pos.size, dtype=int)
    intervals_PR1 = np.array([], dtype=int)
    intervals_PR2 = np.array([], dtype=int)
    intervals_PR3 = np.array([], dtype=int)
    inds_PR1 = np.array([], dtype=int)
    inds_PR2 = np.array([], dtype=int)
    inds_PR3 = np.array([], dtype=int)
    # arr_amp_p1 = np.array([])
    # arr_amp_p2 = np.array([])
    # arr_amp_p3 = np.array([])
    arr_amp_p1 = np.array([])
    arr_amp_p2 = np.array([])
    arr_amp_p3 = np.array([])

    for i in inds_min:
        # if (chars[i] == 'N') and (chars[i - 1] == 'N'):  # and (chars[i + 1] == 'N'):
        # len_pr = int(intervals[i] * 0.4)  # len_pr = int(intervals[i] * 0.36 + 5)
        len_pr = int(intervals[i]**0.5 * 3.7)#int(intervals[i] * 0.36 + 5)
        start = r_pos[i] - len_pr
        stop = r_pos[i] - 10  # r_pos[i] - 7
        # stop = r_pos[i] - int(len_pr * 0.05 + 7)
        fragment1 = lead1[start:stop]
        fragment2 = lead2[start:stop]
        fragment3 = lead3[start:stop]
        # if (k + 15) > i >= k:
        #     p04.plot(fragment1 - 0.1, pen='g', title='Fragment1')
        #     p04.plot(fragment2 - 0.3, pen='y', title='Fragment2')
        #     p04.plot(fragment3 - 0.5, pen='c', title='Fragment3')
        # fragment1 = filtfilt(bl1, al1, fragment1)
        # fragment2 = filtfilt(bl1, al1, fragment2)
        # fragment3 = filtfilt(bl1, al1, fragment3)
        # fragment1 = filtfilt(bh1, ah1, fragment1)
        # fragment2 = filtfilt(bh1, ah1, fragment2)
        # fragment3 = filtfilt(bh1, ah1, fragment3)
        # fragment1 = fragment1 - np.mean(fragment1)
        # fragment2 = fragment2 - np.mean(fragment2)
        # fragment3 = fragment3 - np.mean(fragment3)
        # fragment1[fragment1 < 0.0] = 0.0
        # fragment2[fragment2 < 0.0] = 0.0
        # fragment3[fragment3 < 0.0] = 0.0
        # pzub1, amp_p1, ind_p1 = get_p_amp(fragment1, 1.0, 0.001)
        # pzub2, amp_p2, ind_p2 = get_p_amp(fragment2, 1.0, 0.001)
        # pzub3, amp_p3, ind_p3 = get_p_amp(fragment3, 1.0, 0.001)
        amp_p1, ind_p1 = get_amp_pos_p(fragment1)
        amp_p2, ind_p2 = get_amp_pos_p(fragment2)
        amp_p3, ind_p3 = get_amp_pos_p(fragment3)
        if 1.0 > amp_p1 > 0.001:  # if 1.0 > amp_p1 > 0.001
            intervals_PR1 = np.append(intervals_PR1, len_pr - ind_p1)
            inds_PR1 = np.append(inds_PR1, i)
            arr_amp_p1 = np.append(arr_amp_p1, amp_p1)
        else:
            if intervals_PR1.size > 0:
                intervals_PR1 = np.append(intervals_PR1, np.mean(intervals_PR1))
                inds_PR1 = np.append(inds_PR1, i)
        if 1.0 > amp_p2 > 0.001:  # 0.05
            intervals_PR2 = np.append(intervals_PR2, len_pr - ind_p2)
            inds_PR2 = np.append(inds_PR2, i)
            arr_amp_p2 = np.append(arr_amp_p2, amp_p2)
        else:
            if intervals_PR2.size > 0:
                intervals_PR2 = np.append(intervals_PR2, np.mean(intervals_PR2))
                inds_PR2 = np.append(inds_PR2, i)
        if 1.0 > amp_p3 > 0.001:
            intervals_PR3 = np.append(intervals_PR3, len_pr - ind_p3)
            inds_PR3 = np.append(inds_PR3, i)
            arr_amp_p3 = np.append(arr_amp_p3, amp_p3)
        else:
            if intervals_PR3.size > 0:
                intervals_PR3 = np.append(intervals_PR3, np.mean(intervals_PR3))
                inds_PR3 = np.append(inds_PR3, i)
    mean_amp_p1 = np.median(arr_amp_p1)
    mean_amp_p2 = np.median(arr_amp_p2)
    mean_amp_p3 = np.median(arr_amp_p3)
    max_amp_p1 = np.mean(arr_amp_p1[arr_amp_p1 >= mean_amp_p1])
    max_amp_p1 = np.mean(arr_amp_p1[arr_amp_p1 >= max_amp_p1])
    max_amp_p2 = np.mean(arr_amp_p2[arr_amp_p2 >= mean_amp_p2])
    max_amp_p2 = np.mean(arr_amp_p2[arr_amp_p2 >= max_amp_p2])
    max_amp_p3 = np.mean(arr_amp_p3[arr_amp_p3 >= mean_amp_p3])
    max_amp_p3 = np.mean(arr_amp_p3[arr_amp_p3 >= max_amp_p3])
    min_amp_p1 = np.mean(arr_amp_p1[arr_amp_p1 <= mean_amp_p1])
    # min_amp_p1 = np.mean(arr_amp_p1[arr_amp_p1 <= min_amp_p1])
    min_amp_p2 = np.mean(arr_amp_p2[arr_amp_p2 <= mean_amp_p2])
    # min_amp_p2 = np.mean(arr_amp_p2[arr_amp_p2 <= min_amp_p2])
    min_amp_p3 = np.mean(arr_amp_p3[arr_amp_p3 <= mean_amp_p3])
    # min_amp_p3 = np.mean(arr_amp_p3[arr_amp_p3 <= min_amp_p3])

    print(f"max_amp_p1: {max_amp_p1}")
    print(f"mean_amp_p1: {mean_amp_p1}")
    print(f"min_amp_p1: {min_amp_p1}")

    print(f"max_amp_p2: {max_amp_p2}")
    print(f"mean_amp_p2: {mean_amp_p2}")
    print(f"min_amp_p2: {min_amp_p2}")

    print(f"max_amp_p3: {max_amp_p3}")
    print(f"mean_amp_p3: {mean_amp_p3}")
    print(f"min_amp_p3: {min_amp_p3}")

    mean_PR1 = int(np.median(intervals_PR1))
    mean_PR2 = int(np.median(intervals_PR2))
    mean_PR3 = int(np.median(intervals_PR3))
    print(f"mean_PR1: {mean_PR1}     {intervals_PR1.size}")
    print(f"mean_PR2: {mean_PR2}     {intervals_PR2.size}")
    print(f"mean_PR3: {mean_PR3}     {intervals_PR3.size}")
    # PR = int((mean_PR1 + mean_PR2 + mean_PR3) / 3)
    # mean_PR1 = PR
    # mean_PR2 = PR
    # mean_PR3 = PR
    # mean_PR1 = int(intervals_PR1.mean())
    # mean_PR2 = int(intervals_PR2.mean())
    # mean_PR3 = int(intervals_PR3.mean())

    # marr_amp_p1 = interp_pr(arr_amp_p1)
    # marr_amp_p2 = interp_pr(arr_amp_p2)
    # marr_amp_p3 = interp_pr(arr_amp_p3)
    # marr_amp_p1 = moving_average(marr_amp_p1, 15)
    # marr_amp_p2 = moving_average(marr_amp_p2, 15)
    # marr_amp_p3 = moving_average(marr_amp_p3, 15)
    win_len = 151
    half_win = win_len // 2
    if intervals_PR1.size > win_len:  # 21
        intervals_PR1 = medfilt(intervals_PR1, win_len)
        intervals_PR1[:half_win] = intervals_PR1[half_win]
        intervals_PR1[-half_win:] = intervals_PR1[-half_win - 1]
    if intervals_PR2.size > win_len:
        intervals_PR2 = medfilt(intervals_PR2, win_len)
        intervals_PR2[:half_win] = intervals_PR2[half_win]
        intervals_PR2[-half_win:] = intervals_PR2[-half_win - 1]
    if intervals_PR3.size > win_len:
        intervals_PR3 = medfilt(intervals_PR3, win_len)
        intervals_PR3[:half_win] = intervals_PR3[half_win]
        intervals_PR3[-half_win:] = intervals_PR3[-half_win - 1]

    # presence_PR1 = np.ones(r_pos.size) * mean_PR1
    # presence_PR2 = np.ones(r_pos.size) * mean_PR2
    # presence_PR3 = np.ones(r_pos.size) * mean_PR3
    presence_PR1[inds_PR1] = intervals_PR1
    presence_PR2[inds_PR2] = intervals_PR2
    presence_PR3[inds_PR3] = intervals_PR3
    presence_PR1 = interp_pr(presence_PR1)
    presence_PR2 = interp_pr(presence_PR2)
    presence_PR3 = interp_pr(presence_PR3)
    presence_PR = med_pr(presence_PR1, presence_PR2, presence_PR3)
    # presence_PR1 = truncate_win2(presence_PR1, 0.5, 2000)
    # presence_PR2 = truncate_win2(presence_PR2, 0.5, 2000)
    # presence_PR3 = truncate_win2(presence_PR3, 0.5, 2000)
    # presence_PR1 = moving_average(presence_PR1, 2000)
    # presence_PR2 = moving_average(presence_PR2, 2000)
    # presence_PR3 = moving_average(presence_PR3, 2000)
    # presence_PR1 = medfilt(presence_PR1, 21)
    # presence_PR2 = medfilt(presence_PR2, 21)
    # presence_PR3 = medfilt(presence_PR3, 21)
    np.save("presence_PR1.npy", presence_PR1)
    np.save("presence_PR2.npy", presence_PR2)
    np.save("presence_PR3.npy", presence_PR3)
    np.save("presence_PR.npy", presence_PR)
    # presence_PR = np.array([presence_PR1], [presence_PR2], [presence_PR3])
    # presence_PR = np.mean(presence_PR)
    # presence_PR = presence_PR.astype(int)
    # p03.plot(presence_PR1, pen='g')
    # p03.plot(presence_PR2, pen='y')
    # p03.plot(presence_PR3, pen='c')
    # presence_PR1 = moving_average(presence_PR1, 151)
    # presence_PR2 = moving_average(presence_PR2, 151)
    # presence_PR3 = moving_average(presence_PR3, 151)
    # presence_PR1 = presence_PR1.astype(int)
    # presence_PR2 = presence_PR2.astype(int)
    # presence_PR3 = presence_PR3.astype(int)
    # mean_PR = np.mean([presence_PR1, presence_PR2, presence_PR3], axis=0)
    # mean_PR = moving_average(mean_PR, 15)
    # mean_PR = mean_PR.astype(int)
    # p03.plot(presence_PR1, pen='g')
    # p03.plot(presence_PR2, pen='y')
    # p03.plot(presence_PR3, pen='c')
    # p03.plot(mean_PR, pen='y')
    # p03.plot(arr_amp_p1, pen='g')
    # p03.plot(arr_amp_p2-1, pen='g')
    # p03.plot(arr_amp_p3-2, pen='g')
    # p03.plot(marr_amp_p1, pen='g')
    # p03.plot(marr_amp_p2 - 1, pen='g')
    # p03.plot(marr_amp_p3 - 2, pen='g')
    presence_PR1 = presence_PR.astype(int)
    presence_PR2 = presence_PR.astype(int)
    presence_PR3 = presence_PR.astype(int)

    return max_amp_p1, max_amp_p2, max_amp_p3, min_amp_p1, min_amp_p2, min_amp_p3, presence_PR1, presence_PR2, presence_PR3

def med_pr(presence_PR1, presence_PR2, presence_PR3):
    out = np.zeros(presence_PR1.size)
    for i in range(presence_PR1.size):
        out[i] = np.median([presence_PR1[i], presence_PR2[i], presence_PR3[i]])
    return out

@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars, max_amp_p1, max_amp_p2, max_amp_p3, min_amp_p1, min_amp_p2, min_amp_p3, mean_PR1, mean_PR2,
          mean_PR3):
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    p1 = np.zeros(len(r_pos), dtype=np.float32)
    p1[:5] = 1.0
    p2 = np.zeros(len(r_pos), dtype=np.float32)
    p2[:5] = 1.0
    p3 = np.zeros(len(r_pos), dtype=np.float32)
    p3[:5] = 1.0
    # out = np.ones(len(r_pos), dtype=np.float32) * 7.0
    out = np.zeros(len(r_pos), dtype=np.float32)
    for i in range(4, len(r_pos)):  # range(4, len(r_pos))
        start_range_interval = int(intervals[i]**0.5 * 3.7)#int(intervals[i] * 0.5)  # int(intervals[i]**0.5 * 3.7)
        # stop_range_interval = int(intervals[i] ** 0.5 * 0.66 + 3.0)
        stop_range_interval = 10
        start_slice = r_pos[i] - start_range_interval
        end_slice = r_pos[i] - stop_range_interval
        start_slice1 = r_pos[i] - mean_PR1[i] - 15  # -15
        start_slice2 = r_pos[i] - mean_PR2[i] - 15
        start_slice3 = r_pos[i] - mean_PR3[i] - 15
        # fragment_l1 = lead1[start_slice:end_slice]
        # fragment_l2 = lead2[start_slice:end_slice]
        # fragment_l3 = lead3[start_slice:end_slice]
        # if len_pr * 0.2 > 10:
        #     end_slice = r_pos[i] - 10
        # end_slice = start_slice + 30
        end_slice1 = r_pos[i] - stop_range_interval
        end_slice2 = r_pos[i] - stop_range_interval
        end_slice3 = r_pos[i] - stop_range_interval
        # end_slice1 = r_pos[i] - mean_PR1[i] + int(mean_PR1[i] ** 0.5 * 3.0)  # +15
        # end_slice2 = r_pos[i] - mean_PR2[i] + int(mean_PR2[i] ** 0.5 * 3.0)  # +15
        # end_slice3 = r_pos[i] - mean_PR3[i] + int(mean_PR3[i] ** 0.5 * 3.0)  # +15
        if mean_PR1[i] > intervals[i] * 0.36:  # // 3
            # if len_pr < mean_PR1:
            # fragment_l1 = lead1[start_slice1:end_slice1]
            fragment_l1 = lead1[start_slice:end_slice]
        else:
            fragment_l1 = lead1[start_slice:end_slice1]
        if mean_PR2[i] > intervals[i] * 0.36:
            # if len_pr < mean_PR2:
            # fragment_l2 = lead2[start_slice2:end_slice2]
            fragment_l2 = lead2[start_slice:end_slice]
        else:
            fragment_l2 = lead2[start_slice:end_slice2]
        if mean_PR3[i] > intervals[i] * 0.36:
            # if len_pr < mean_PR3:
            # fragment_l3 = lead3[start_slice3:end_slice3]
            fragment_l3 = lead3[start_slice:end_slice]
        else:
            fragment_l3 = lead3[start_slice:end_slice3]
        # fragment_l1 = lead1[r_pos[i] - mean_PR1_copy - 15:r_pos[i] - mean_PR1_copy + 15]  # -+15
        # fragment_l2 = lead2[r_pos[i] - mean_PR2_copy - 15:r_pos[i] - mean_PR2_copy + 15]
        # fragment_l3 = lead3[r_pos[i] - mean_PR3_copy - 15:r_pos[i] - mean_PR3_copy + 15]
        # bp = 0.0
        # amp_p1 = 0.0
        # amp_p2 = 0.0
        # amp_p3 = 0.0
        if not (chars[i] == 'N'):
            p1[i] = 1.0
            p2[i] = 1.0
            p3[i] = 1.0
            # p1[i] = 0.6
            # p2[i] = 0.6
            # p3[i] = 0.6
        else:
            amp_pzub1, ind_p1 = get_amp_pos_p(fragment_l1)
            if (5.0 * max_amp_p1 > amp_pzub1 > 0.15 * min_amp_p1) and (amp_pzub1 > 0.013):
                p1[i] = 1.0
            amp_pzub2, ind_p2 = get_amp_pos_p(fragment_l2)
            if (5.0 * max_amp_p2 > amp_pzub2 > 0.15 * min_amp_p2) and (amp_pzub2 > 0.013):
                p2[i] = 1.0
            amp_pzub3, ind_p3 = get_amp_pos_p(fragment_l3)
            if (5.0 * max_amp_p3 > amp_pzub3 > 0.15 * min_amp_p3) and (amp_pzub3 > 0.013):
                p3[i] = 1.0
            # fragment_l1_f = filtfilt(bl2, al2, fragment_l1)
            # fragment_l1_f = filtfilt(bh2, ah2, fragment_l1_f)
            # fragment_l1_f = fragment_l1_f - np.mean(fragment_l1_f)
            # fragment_l1_f[fragment_l1_f < 0.0] = 0.0
            # pzub1, amp_pzub1, _ = get_p_amp(fragment_l1_f, max_amp_p1, min_amp_p1)
            # p1[i] = pzub1
            # fragment_l2_f = filtfilt(bl2, al2, fragment_l2)
            # fragment_l2_f = filtfilt(bh2, ah2, fragment_l2_f)
            # fragment_l2_f = fragment_l2_f - np.mean(fragment_l2_f)
            # fragment_l2_f[fragment_l2_f < 0.0] = 0.0
            # pzub2, amp_pzub2, _ = get_p_amp(fragment_l2_f, max_amp_p2, min_amp_p2)
            # p2[i] = pzub2
            # fragment_l3_f = filtfilt(bl2, al2, fragment_l3)
            # fragment_l3_f = filtfilt(bh2, ah2, fragment_l3_f)
            # fragment_l3_f = fragment_l3_f - np.mean(fragment_l3_f)
            # fragment_l3_f[fragment_l3_f < 0.0] = 0.0
            # pzub3, amp_pzub3, _ = get_p_amp(fragment_l3_f, max_amp_p3, min_amp_p3)
            # p3[i] = pzub3
        if i == k:
            # app = QApplication(sys.argv)
            # p01 = pg.plot()
            # p01.showGrid(x=True, y=True)
            # p01.setTitle('p01')
            # p02 = pg.plot()
            # p02.showGrid(x=True, y=True)
            # p02.setTitle('p02')
            # p01.plot(fragment_l1, pen='g')
            # p01.plot(fragment_l2, pen='y')
            # p01.plot(fragment_l3, pen='c')
            # p01.plot(fragment_l1_f - 0.2, pen='g')
            # p01.plot(fragment_l2_f - 0.2, pen='y')
            # p01.plot(fragment_l3_f - 0.2, pen='c')
            # p02.plot(lead1[r_pos[i] - 5000:r_pos[i] + 5000], pen='g')
            # p02.plot(lead2[r_pos[i] - 5000:r_pos[i] + 5000] - 2, pen='y')
            # p02.plot(lead3[r_pos[i] - 5000:r_pos[i] + 5000] - 4, pen='c')
            # sys.exit(app.exec())
            print(f"i = {i}")
            print(p1[i - 4:i + 1])
            print(p2[i - 4:i + 1])
            print(p3[i - 4:i + 1])
            print(f"{amp_pzub1:.5f}, {amp_pzub2:.5f}, {amp_pzub3:.5f}")
            # np.save("fragment_l1_f.npy", fragment_l1_f)
            # np.save("fragment_l2_f.npy", fragment_l2_f)
            # np.save("fragment_l3_f.npy", fragment_l3_f)
        # sum_p1 = np.sum(p1[i - 4:i - 1]) + p1[i-1] * 2.0 + p1[i] * 3.0
        # sum_p2 = np.sum(p2[i - 4:i - 1]) + p2[i-1] * 2.0 + p2[i] * 3.0
        # sum_p3 = np.sum(p3[i - 4:i - 1]) + p3[i-1] * 2.0 + p3[i] * 3.0
        # sum_p1 = p1[i - 4] + p1[i - 3] * 2.0 + p1[i - 2] * 3.0 + p1[i - 1] * 2.0 + p1[i]
        # sum_p2 = p2[i - 4] + p2[i - 3] * 2.0 + p2[i - 2] * 3.0 + p2[i - 1] * 2.0 + p2[i]
        # sum_p3 = p3[i - 4] + p3[i - 3] * 2.0 + p3[i - 2] * 3.0 + p3[i - 1] * 2.0 + p3[i]
        sum_p1 = p1[i - 4] + p1[i - 3] + p1[i - 2] + p1[i - 1] + p1[i]
        sum_p2 = p2[i - 4] + p2[i - 3] + p2[i - 2] + p2[i - 1] + p2[i]
        sum_p3 = p3[i - 4] + p3[i - 3] + p3[i - 2] + p3[i - 1] + p3[i]
        # arr_p = np.array([[p1[i - 4] + p1[i - 3] + p1[i - 2] + p1[i - 1] + p1[i]],
        #                   [p2[i - 4] + p2[i - 3] + p2[i - 2] + p2[i - 1] + p2[i]],
        #                   [p3[i - 4] + p3[i - 3] + p3[i - 2] + p3[i - 1] + p3[i]]])
        # if np.sum(fragment_l1) == 0.0:
        #     arr_p[0] = (arr_p[1] + arr_p[2]) / 2.0
        # elif np.sum(fragment_l2) == 0.0:
        #     arr_p[1] = (arr_p[0] + arr_p[2]) / 2.0
        # elif np.sum(fragment_l3) == 0.0:
        #     arr_p[2] = (arr_p[0] + arr_p[1]) / 2.0
        # sum_buff = arr_p.sum()
        # if arr_p[:][2].sum() == 3.0:
        #     sum_buff = 15.0
        # elif arr_p[:][2].sum() == 0.0:
        #     sum_buff = 0.0
        # if np.sum(fragment_l1) == 0.0:
        #     sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif np.sum(fragment_l2) == 0.0:
        #     sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif np.sum(fragment_l3) == 0.0:
        #     sum_p3 = (sum_p1 + sum_p2) / 2.0
        # if ((sum_p1 < 4.5) and (sum_p2 > 4.5) and (sum_p3 > 4.5)) or (
        #         (sum_p1 > 4.5) and (sum_p2 < 4.5) and (sum_p3 < 4.5)):
        #     sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif ((sum_p2 < 4.5) and (sum_p1 > 4.5) and (sum_p3 > 4.5)) or (
        #         (sum_p2 > 4.5) and (sum_p1 < 4.5) and (sum_p3 < 4.5)):
        #     sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif ((sum_p3 < 4.5) and (sum_p1 > 4.5) and (sum_p2 > 4.5)) or (
        #         (sum_p3 > 4.5) and (sum_p1 < 4.5) and (sum_p2 < 4.5)):
        #     sum_p3 = (sum_p1 + sum_p2) / 2.0
        #     print(p1[i - 4:i + 1])
        #     print(p2[i - 4:i + 1])
        #     print(p3[i - 4:i + 1])
        # sum_p1 = np.sum(p1[i - 4:i + 1])
        # sum_p2 = np.sum(p2[i - 4:i + 1])
        # sum_p3 = np.sum(p3[i - 4:i + 1])
        # if np.sum(fragment_l1) == 0.0:
        #     sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif np.sum(fragment_l2) == 0.0:
        #     sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif np.sum(fragment_l3) == 0.0:
        #     sum_p3 = (sum_p1 + sum_p2) / 2.0
        if ((sum_p1 < 2.5) and (sum_p2 > 2.5) and (sum_p3 > 2.5)) or (
                (sum_p1 > 2.5) and (sum_p2 < 2.5) and (sum_p3 < 2.5)):
            if sum_p1 < 5.0:
                sum_p1 = (sum_p2 + sum_p3) / 2.0
        elif ((sum_p2 < 2.5) and (sum_p1 > 2.5) and (sum_p3 > 2.5)) or (
                (sum_p2 > 2.5) and (sum_p1 < 2.5) and (sum_p3 < 2.5)):
            if sum_p2 < 5.0:
                sum_p2 = (sum_p1 + sum_p3) / 2.0
        elif ((sum_p3 < 2.5) and (sum_p1 > 2.5) and (sum_p2 > 2.5)) or (
                (sum_p3 > 2.5) and (sum_p1 < 2.5) and (sum_p2 < 2.5)):
            if sum_p3 < 5.0:
                sum_p3 = (sum_p1 + sum_p2) / 2.0

        sum_buff = sum_p1 * sum_p2 * sum_p3 * 0.34  # 0.38
        out[i - 2] = sum_buff
        # out[i] = sum_buff
    out = -(out - np.max(out))
    # out = -(out - 125.0 * 0.34)
    out[:2] = out[2]
    out[-2:] = out[-3]
    return out

def save_F_txt():
    file_ecg_name = QFileDialog.getOpenFileName()[0]
    print(file_ecg_name)
    size = (os.path.getsize(file_ecg_name) - 1023) // 6
    time = get_time_from_samples(size)
    h = time[0] * 24 + time[1]
    total_time = f"Длина записи {h}:{time[2]}:{time[3]}\n"
    start_time = get_start_time(file_ecg_name)
    text = "\n"
    text += f"{file_ecg_name}\n\n"
    text += f"Начало записи {start_time[0]}:{start_time[1]}:{start_time[2]}\n\n"
    fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
    try:
        start_ind_arr = np.load(fdir + "/start_ind_arr.npy")
        stop_ind_arr = np.load(fdir + "/stop_ind_arr.npy")
        diff_stop_start = np.load(fdir + "/diff_stop_start.npy")
        r_pos = np.load(fdir + "/r_pos.npy")
        # print(start_ind_arr)
        # print(stop_ind_arr)
        # print(diff_stop_start)
    except FileNotFoundError:
        print("Файлы start_ind_arr.npy, stop_ind_arr.npy и diff_stop_start.npy не найдены")
    len_f = len(start_ind_arr)
    sum_time = 0
    for i in range(len_f):
        time_start = get_time_from_samples(get_addr_qrs(r_pos[start_ind_arr[i]], start_time[3]))
        time_stop = get_time_from_samples(get_addr_qrs(r_pos[stop_ind_arr[i]], start_time[3]))
        diff = int(r_pos[stop_ind_arr[i]] - r_pos[start_ind_arr[i]])
        sum_time += diff
        time_diff = get_time_from_samples(diff)
        diff_h = time_diff[0] * 24 + time_diff[1]
        text += f"{time_start[1]}:{time_start[2]}:{time_start[3]}  {time_stop[1]}:{time_stop[2]}:{time_stop[3]}  (длит. эпизода {diff_h}:{time_diff[2]}:{time_diff[3]})\n"
        # print(f"{time_start[1]:02d}:{time_start[2]:02d}:{time_start[3]:02d}     {time_stop[1]:02d}:{time_stop[2]:02d}:{time_stop[3]:02d}  (длит. эпизода {diff_h}:{time_diff[2]}:{time_diff[3]})\n")
    sum_fibr_time = get_time_from_samples(sum_time)
    sum_h = sum_fibr_time[0] * 24 + sum_fibr_time[1]
    text += f"\nВсего эпизодов {len_f}, суммарное время фибрилляции {sum_h}:{sum_fibr_time[2]}:{sum_fibr_time[3]}\n\n"
    text += total_time
    with open(fdir + "/F.txt", "w") as f:
        for i, line in enumerate(text):
            f.write(line)



    print(total_time)

def get_time_from_samples(sample_count):
    s = int(sample_count * 0.004)
    m = s // 60
    s = s % 60

    h = m // 60
    m = m % 60

    d = h // 24
    h = h % 24
    # return f"{d:02d} день {h:02d}:{m:02d}:{s:02d}"
    return d, h, m, s

def get_start_time(fname):
    with open(fname, "rb") as f:
        f.seek(151)
        dlmt = f.read(1)
        if dlmt == b":":
            f.seek(150)
            start_h = int(f.read(1))
            f.seek(152)
            start_m = int(f.read(2))
            f.seek(155)
            start_s = int(f.read(2))
        else:
            f.seek(150)
            start_h = int(f.read(2))
            f.seek(153)
            start_m = int(f.read(2))
            f.seek(156)
            start_s = int(f.read(2))
    start_addr = ((start_h * 60 + start_m) * 60 + start_s) * 250
    return start_h, start_m, start_s, start_addr

def get_addr_qrs(addr, start_addr):
    return addr + start_addr

def parse_B_txt():
    r_pos = []
    intervals = []
    chars = []
    forms = []
    with open("C:/EcgVar/B.txt", "r") as f:
        for line in f:
            if ';' in line:
                temp = int(line.split(';')[0])
                r_pos.append(temp)
                temp = int(line.split(';')[1])
                intervals.append(temp)
                temp = line.split(';')[2][0]
                chars.append(temp)
                temp = int(line.split(':')[1])
                forms.append(temp)
        r_pos = np.array(r_pos)
        intervals = np.array(intervals)
        chars = np.array(chars)
        forms = np.array(forms)
    return r_pos, intervals, chars, forms

class Visual:
    def __init__(self):
        self.count = 0

    def print_var(self, var, start, num):
        if (self.count >= start) and (self.count < start + num):
            print(var)
            self.count += 1
        else:
            self.count += 1

    def plot_fragment(self, fragment, num):
        if (self.count >= num) and (self.count < num + 3):
            p = pg.plot()
            p.showGrid(x=True, y=True)
            p.setTitle(f"Plotter{self.count}")
            p.plot(fragment, pen='g')
            self.count += 1
        else:
            self.count += 1

# printer = Visual()
plotter = Visual()
plotter1 = Visual()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    save_F_txt()
    # sys.exit(app.exec())
    sys.exit()









