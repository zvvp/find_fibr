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

k = 31300
bl, al = butter(3, 12.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
# print(f"bl: {bl}")
# print(f"al: {al}")
b, a = butter(4, 15.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
# print(f"b: {b}")
# print(f"a: {a}")
bh, ah = butter(1, 0.2, 'hp', fs=250)  # 3, 1.0, 'hp', fs=250


# print(f"bh: {bh}")
# print(f"ah: {ah}")

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
            if (';N' in lines[i]) or (
                    ';S' in lines[i]):  # and (not ';V' in lines[i - 1]) and (not ';S' in lines[i - 1]):
                periods = get_periods(lines[i - 2:i + 3])
                tf = np.array(periods)
                diff_tf = np.diff(tf)
                diff21 = diff_tf[2] - diff_tf[1]
                ref_t = np.array([tf[1], tf[1] * 0.6, tf[1] * 1.3, tf[1]])
                ref_t0 = np.array([tf[1], tf[1] * 0.6, tf[1], tf[1] * 0.6])
                ref_t1 = np.array([tf[1], tf[1] * 1.35, tf[1], tf[1] * 1.35])
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
        mean_diff = np.mean(abs(diff_intervals[i - 1:i + 2]))  # i - 2:i + 3
        mean_interval = np.mean(intervals[[i - 3, i - 2, i - 1, i + 2, i + 3]])
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
        if ((out[i] / ((out[i - 1] + out[i + 1]) / 2.0) > 1.9) and
                (abs(out[i - 1] - out[i + 1]) < 10)):
            out[i] = (fintervals[i - 1] + fintervals[i + 1]) / 2.0
        elif ((2.2 > (out[i - 1] + out[i + 2]) / (out[i] + out[i + 1]) > 1.8) and
              (abs(out[i - 1] - out[i + 2]) < 15)):
            out[i] = fintervals[i + 2]
            out[i + 1] = fintervals[i - 1]
        elif ((2.2 > (out[i - 1] * 2 + out[i + 3]) / (out[i] + out[i + 1] + out[i + 2]) > 1.8) and
              (abs(out[i - 1] - out[i + 3]) < 15)):
            out[i] = fintervals[i + 3]
            out[i + 1] = fintervals[i - 1]
            out[i + 2] = fintervals[i + 3]

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
        sort_mul_diff = np.sort(mul_diff)  # [:-15]
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
        if flag == False and e_coef_p[i] < intervals[i] and e_coef_p[i + 1] > intervals[i + 1] and mask_pseudo_fibr[
            i] == 1:
            if i - stop_ind > min_size:
                start_ind = i
                start_ind_arr = np.append(start_ind_arr, start_ind)
            else:
                if stop_ind_arr.size > 0:
                    stop_ind_arr = np.delete(stop_ind_arr, -1)
            flag = True
        elif flag == True and e_coef_p[i] > intervals[i] and e_coef_p[i + 1] < intervals[i + 1] and mask_pseudo_fibr[
            i] == 1:
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
        out[i] = dt2_mean  # * (1 + 100.0 / mean_win) # dt2_mean * 175.0 / mean_win
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
    out[:half_win + 1] = out[half_win + 2]
    out[-half_win - 1:] = out[-half_win - 2]
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


def interp_pr(intervals_PR, inds_PR, r_pos_size):
    out = np.array([])
    mean_pr = np.mean(intervals_PR)
    for i in range(0, inds_PR[0]):
        out = np.append(out, mean_pr)
    for i in range(0, len(inds_PR) - 1):
        diff_pr = intervals_PR[i + 1] - intervals_PR[i]
        count_step = int(inds_PR[i + 1] - inds_PR[i])
        step_pr = diff_pr / count_step
        for j in np.arange(count_step):
            out = np.append(out, intervals_PR[i] + int(step_pr * j))
    for i in range(inds_PR[-1], r_pos_size):
        out = np.append(out, mean_pr)
    return out


def interp_pr1(ch):
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

def get_p_amp2(fragment, mean_amp_p):
    pzub = 0.0
    amp_p_arr = np.array([])
    temp_p = 0.0
    amp_p = 0.0
    flag = False
    num_of_samples = 0
    for i in np.arange(1, len(fragment)):
        if (fragment[i] > 0.007) and (fragment[i - 1] <= 0.007):
            flag = True
        elif (fragment[i] <= 0.0) and (fragment[i - 1] > 0.0):
            flag = False
            if num_of_samples > 1:
                amp_p_arr = np.append(amp_p_arr, temp_p)
            temp_p = 0.0
            num_of_samples = 0
        # if (flag == True) and (fragment[i] > temp_p):
        #     temp_p = fragment[i]
        if flag == True:
            num_of_samples += 1
            if fragment[i] > temp_p:
                temp_p = fragment[i]
    if 3 >= len(amp_p_arr) > 0:
        amp_p = np.max(amp_p_arr)
        if 2.0 > amp_p > mean_amp_p * 0.18:
            pzub = 1.0
    return pzub, amp_p

    # ind_max_loc = argrelmax(fragment)[0]
    # if ind_max_loc.size > 0:
    #     ind_max = np.argmax(fragment[ind_max_loc])
    #     amp_p_arr = fragment[ind_max_loc[ind_max]]
    #     if 2.0 > amp_p_arr > 0.005:
    #         amp_pzub = amp_p_arr
    #         pzub = 1.0
    # return pzub, amp_p_arr


def get_p_amp1(fragment, mean_amp_p):
    pzub = 0.0
    amp_pzub = 0.0
    ind_max_loc = argrelmax(fragment)[0]
    if ind_max_loc.size == 1:
        side_l = fragment[ind_max_loc[0]] - np.min(fragment[:ind_max_loc[0]])
        side_r = fragment[ind_max_loc[0]] - np.min(fragment[ind_max_loc[0]:])
        amp_pzub = np.min([side_l, side_r])
    elif ind_max_loc.size == 2:
        side_l = fragment[ind_max_loc[0]] - np.min(fragment[:ind_max_loc[0]])
        side_m1 = fragment[ind_max_loc[0]] - np.min(fragment[ind_max_loc[0]:ind_max_loc[1]])
        side_m2 = fragment[ind_max_loc[1]] - np.min(fragment[ind_max_loc[0]:ind_max_loc[1]])
        side_r = fragment[ind_max_loc[1]] - np.min(fragment[ind_max_loc[1]:])
        amp_pzub1 = np.min([side_l, side_m1])
        amp_pzub2 = np.min([side_m2, side_r])
        amp_pzub = np.max([amp_pzub1, amp_pzub2])
    elif ind_max_loc.size == 3:
        side_l = fragment[ind_max_loc[0]] - np.min(fragment[:ind_max_loc[0]])
        side_m1 = fragment[ind_max_loc[0]] - np.min(fragment[ind_max_loc[0]:ind_max_loc[1]])
        side_m2 = fragment[ind_max_loc[1]] - np.min(fragment[ind_max_loc[0]:ind_max_loc[1]])
        side_m3 = fragment[ind_max_loc[1]] - np.min(fragment[ind_max_loc[1]:ind_max_loc[2]])
        side_m4 = fragment[ind_max_loc[2]] - np.min(fragment[ind_max_loc[1]:ind_max_loc[2]])
        side_r = fragment[ind_max_loc[2]] - np.min(fragment[ind_max_loc[2]:])
        amp_pzub1 = np.min([side_l, side_m1])
        amp_pzub2 = np.min([side_m2, side_m3])
        amp_pzub3 = np.min([side_m4, side_r])
        amp_pzub = np.max([amp_pzub1, amp_pzub2, amp_pzub3])
    if (amp_pzub > mean_amp_p * 0.15):#0.15):# and (amp_pzub > 0.0000005): # (amp_pzub > mean_amp_p * 0.01) and (amp_pzub > 0.001)
        pzub = 1.0
    return pzub, amp_pzub


def get_p_amp(fragment, mean_amp_p):
    pzub = 0.0
    amp_pzub = 0.0
    # amp_pzub1 = 0.0
    # amp_pzub2 = 0.0
    ind_max_loc = argrelmax(fragment)[0]
    if ind_max_loc.size == 0:
        return pzub, amp_pzub
    ind_min_loc = argrelmin(fragment)[0]
    ind_loc = np.concatenate((ind_max_loc, ind_min_loc))
    ind_loc = np.sort(ind_loc)
    loc_arr = fragment[ind_loc]
    loc_arr = np.append(loc_arr, fragment[-1])
    loc_arr = np.append(fragment[0], loc_arr)
    print(f"loc_arr.size = {loc_arr.size}")
    diff_loc = loc_arr[1:] - loc_arr[:-1]
    print(f"diff_loc.size = {diff_loc.size}")
    abs_diff_loc = np.abs(diff_loc)
    if ind_max_loc.size == 1:
        if ind_min_loc.size == 0:
            amp_pzub = np.min(abs_diff_loc)
        elif ind_min_loc.size == 1:
            if ind_max_loc[0] < ind_min_loc[0]:
                amp_pzub = np.min(abs_diff_loc[:2])
            elif ind_max_loc[0] > ind_min_loc[0]:
                amp_pzub = np.min(abs_diff_loc[1:3])
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
        max_amp = np.max([amp_pzub1, amp_pzub2])
        amp_pzub = max_amp
        # min_amp = np.min([amp_pzub1, amp_pzub2])
        # div_amp = max_amp / min_amp
        # if div_amp > 1.3: # div_amp > 10.0
        #     amp_pzub = max_amp
    if (amp_pzub > mean_amp_p * 0.01) and (amp_pzub > 0.001):  # (amp_pzub > mean_amp_p * 0.05) and (amp_pzub > 0.0035):
        pzub = 1.0
    return pzub, amp_pzub


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


def get_inds_min_diff(intervals, chars):
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
    for i in range(1, diff_intervals.size - 1):
        min_diff = intervals[i] * 0.03
        # if min_diff > 7.0:
        #     min_diff = 7.0
        if (diff_intervals[i] <= min_diff) and (chars[i] == "N") and (chars[i - 1] == "N"):
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
    # len_pr = int(intervals[ind] * 0.36 + 5)
    # start = r_pos[ind] - len_pr
    # stop = r_pos[ind] - int(len_pr ** 0.5 * 0.66 + 2)
    # stop = r_pos[ind] - 5
    b, a = butter(2, 18, btype='lowpass', fs=250)
    bl, al = butter(2, 3.5, btype='lowpass', fs=250)

    start1 = r_pos[ind] - mean_PR1[ind] - 15
    start2 = r_pos[ind] - mean_PR2[ind] - 15
    start3 = r_pos[ind] - mean_PR3[ind] - 15
    stop = r_pos[ind] - 7
    # stop1 = r_pos[ind] - int(intervals[ind] * 0.016 + 7)
    # stop2 = r_pos[ind] - int(intervals[ind] * 0.016 + 7)
    # stop3 = r_pos[ind] - int(intervals[ind] * 0.016 + 7)
    stop1 = r_pos[ind] - 10
    stop2 = r_pos[ind] - 10
    stop3 = r_pos[ind] - 10
    fragment1 = lead1[start1:stop1]
    fragment2 = lead2[start2:stop2]
    fragment3 = lead3[start3:stop3]
    # fragment1 = fragment1 - np.mean(fragment1)
    # fragment2 = fragment2 - np.mean(fragment2)
    # fragment3 = fragment3 - np.mean(fragment3)
    # fragment1[fragment1 < 0.0] = 0.0
    # fragment2[fragment2 < 0.0] = 0.0
    # fragment3[fragment3 < 0.0] = 0.0
    p = pg.plot()
    p.showGrid(x=True, y=True)
    p.setTitle('p')
    # p.plot(fragment1, pen='g')
    # p.plot(fragment2, pen='y')
    # p.plot(fragment3, pen='c')
    fragment1 = filtfilt(b, a, fragment1)
    fragment2 = filtfilt(b, a, fragment2)
    fragment3 = filtfilt(b, a, fragment3)
    fragment1 = fragment1 - np.mean(fragment1)
    fragment2 = fragment2 - np.mean(fragment2)
    fragment3 = fragment3 - np.mean(fragment3)
    isoline1 = filtfilt(bl, al, fragment1)
    isoline2 = filtfilt(bl, al, fragment2)
    isoline3 = filtfilt(bl, al, fragment3)
    fragment1 = fragment1 - isoline1
    fragment2 = fragment2 - isoline2
    fragment3 = fragment3 - isoline3
    # isoline1 = filtfilt(bl, al, fragment1)
    # isoline2 = filtfilt(bl, al, fragment2)
    # isoline3 = filtfilt(bl, al, fragment3)
    # fragment1 = fragment1 - isoline1
    # fragment2 = fragment2 - isoline2
    # fragment3 = fragment3 - isoline3
    # fragment1[fragment1 < 0.0] = 0.0
    # fragment2[fragment2 < 0.0] = 0.0
    # fragment3[fragment3 < 0.0] = 0.0
    # fragment1 = filtfilt(bl, al, fragment1)
    # fragment2 = filtfilt(bl, al, fragment2)
    # fragment3 = filtfilt(bl, al, fragment3)
    p.plot(fragment1, pen='g')
    p.plot(fragment2, pen='y')
    p.plot(fragment3, pen='c')
    # p02.plot(lead1[r_pos[ind] - 1600:r_pos[ind] + 1600], pen='g')
    # p02.plot(lead2[r_pos[ind] - 1600:r_pos[ind] + 1600] - 2, pen='y')
    # p02.plot(lead3[r_pos[ind] - 1600:r_pos[ind] + 1600] - 4, pen='c')


@time_fun
def get_p_pos(lead1, lead2, lead3, intervals, r_pos, chars, inds_min):
    global k#, bl, al
    b, a = butter(2, 18, btype='lowpass', fs=250)
    bl, al = butter(2, 2, btype='lowpass', fs=250)
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    # pos_p1 = 0
    # pos_p2 = 0
    # pos_p3 = 0
    # presence_PR1 = np.zeros(r_pos.size, dtype=int)
    # presence_PR2 = np.zeros(r_pos.size, dtype=int)
    # presence_PR3 = np.zeros(r_pos.size, dtype=int)
    intervals_PR1 = np.array([], dtype=int)
    intervals_PR2 = np.array([], dtype=int)
    intervals_PR3 = np.array([], dtype=int)
    inds_PR1 = np.array([], dtype=int)
    inds_PR2 = np.array([], dtype=int)
    inds_PR3 = np.array([], dtype=int)
    # arr_amp_p1 = np.array([])
    # arr_amp_p2 = np.array([])
    # arr_amp_p3 = np.array([])
    arr_amp_p1 = np.zeros(r_pos.size)
    arr_amp_p2 = np.zeros(r_pos.size)
    arr_amp_p3 = np.zeros(r_pos.size)
    print(f"inds_min = {inds_min[:10]}\n{inds_min[-10:]}")
    for i in inds_min:
        if (chars[i] == 'N') and (chars[i - 1] == 'N'):  # and (chars[i + 1] == 'N'):
            len_pr = int(intervals[i] * 0.4)  # len_pr = int(intervals[i] * 0.36 + 5)
            start = r_pos[i] - len_pr
            stop = r_pos[i] - 10  # r_pos[i] - 7
            # stop = r_pos[i] - int(intervals[i] * 0.016 + 7)
            fragment1 = lead1[start:stop]
            fragment2 = lead2[start:stop]
            fragment3 = lead3[start:stop]
            fragment1 = fragment1 - fragment1.mean()
            fragment2 = fragment2 - fragment2.mean()
            fragment3 = fragment3 - fragment3.mean()
            ffragment1 = filtfilt(b, a, fragment1)
            ffragment2 = filtfilt(b, a, fragment2)
            ffragment3 = filtfilt(b, a, fragment3)
            isoline1 = filtfilt(bl, al, ffragment1)
            isoline2 = filtfilt(bl, al, ffragment2)
            isoline3 = filtfilt(bl, al, ffragment3)
            fragment1 = ffragment1 - isoline1
            fragment2 = ffragment2 - isoline2
            fragment3 = ffragment3 - isoline3
            fragment1[fragment1 < 0.0] = 0.0
            fragment2[fragment2 < 0.0] = 0.0
            fragment3[fragment3 < 0.0] = 0.0

            # if (k + 15) > i >= k:
            #     p04.plot(fragment1 - 0.1, pen='g', title='Fragment1')
            #     p04.plot(fragment2 - 0.3, pen='y', title='Fragment2')
            #     p04.plot(fragment3 - 0.5, pen='c', title='Fragment3')
            # fragment1 = filtfilt(bl, al, fragment1)
            # fragment2 = filtfilt(bl, al, fragment2)
            # fragment3 = filtfilt(bl, al, fragment3)
            # fragment1[fragment1 < 0.0] = 0.0
            # fragment2[fragment2 < 0.0] = 0.0
            # fragment3[fragment3 < 0.0] = 0.0
            # fragment1 = filtfilt(bl, al, fragment1)
            # fragment2 = filtfilt(bl, al, fragment2)
            # fragment3 = filtfilt(bl, al, fragment3)
            # fragment1[fragment1 < 0.0] = 0.0
            # fragment2[fragment2 < 0.0] = 0.0
            # fragment3[fragment3 < 0.0] = 0.0
            # fragment1 = filtfilt(bl, al, fragment1)
            # fragment2 = filtfilt(bl, al, fragment2)
            # fragment3 = filtfilt(bl, al, fragment3)
            # fragment1[fragment1 < 0.0] = 0.0
            # fragment2[fragment2 < 0.0] = 0.0
            # fragment3[fragment3 < 0.0] = 0.0
            # if (k + 15) > i >= k:
            #     p04.plot(fragment1 - 0.2, pen='g', title='Fragment1')
            #     p04.plot(fragment2 - 0.4, pen='y', title='Fragment2')
            #     p04.plot(fragment3 - 0.6, pen='c', title='Fragment3')
            pos_loc_max_frag1 = argrelmax(fragment1)[0]
            pos_loc_max_frag2 = argrelmax(fragment2)[0]
            pos_loc_max_frag3 = argrelmax(fragment3)[0]

            if pos_loc_max_frag1.size >= 1:
                ind_max1 = np.argmax(fragment1[pos_loc_max_frag1])
                amp_p1 = fragment1[pos_loc_max_frag1[ind_max1]]
                # isoline1 = np.min(
                #     (np.min(fragment1[:pos_loc_max_frag1[ind_max1]]), np.min(fragment1[pos_loc_max_frag1[ind_max1]:])))
                # amp_p1 = fragment1[pos_loc_max_frag1[ind_max1]] - isoline1
                if 2.0 > amp_p1 > 0.005:  # if 2.0 > amp_p1 > 0.01
                    intervals_PR1 = np.append(intervals_PR1, len_pr - pos_loc_max_frag1[ind_max1])
                    inds_PR1 = np.append(inds_PR1, i)
                    arr_amp_p1[i] = amp_p1
            if pos_loc_max_frag2.size >= 1:
                ind_max2 = np.argmax(fragment2[pos_loc_max_frag2])
                amp_p2 = fragment2[pos_loc_max_frag2[ind_max2]]
                # isoline2 = np.min(
                #     (np.min(fragment2[:pos_loc_max_frag2[ind_max2]]), np.min(fragment2[pos_loc_max_frag2[ind_max2]:])))
                # amp_p2 = fragment2[pos_loc_max_frag2[ind_max2]] - isoline2
                if 2.0 > amp_p2 > 0.005:  # 0.05
                    intervals_PR2 = np.append(intervals_PR2, len_pr - pos_loc_max_frag2[ind_max2])
                    inds_PR2 = np.append(inds_PR2, i)
                    arr_amp_p2[i] = amp_p2
            if pos_loc_max_frag3.size >= 1:
                ind_max3 = np.argmax(fragment3[pos_loc_max_frag3])
                amp_p3 = fragment3[pos_loc_max_frag3[ind_max3]]
                # isoline3 = np.min(
                #     (np.min(fragment3[:pos_loc_max_frag3[ind_max3]]), np.min(fragment3[pos_loc_max_frag3[ind_max3]:])))
                # amp_p3 = fragment3[pos_loc_max_frag3[ind_max3]] - isoline3
                if 2.0 > amp_p3 > 0.005:
                    intervals_PR3 = np.append(intervals_PR3, len_pr - pos_loc_max_frag3[ind_max3])
                    inds_PR3 = np.append(inds_PR3, i)
                    arr_amp_p3[i] = amp_p3
    # mean_amp_p1 = np.mean(arr_amp_p1)
    # mean_amp_p2 = np.mean(arr_amp_p2)
    # mean_amp_p3 = np.mean(arr_amp_p3)
    mean_amp_p1 = np.median(arr_amp_p1[arr_amp_p1 > 0.0])
    mean_amp_p2 = np.median(arr_amp_p2[arr_amp_p2 > 0.0])
    mean_amp_p3 = np.median(arr_amp_p3[arr_amp_p3 > 0.0])
    print(f"mean_amp_p1: {mean_amp_p1}")
    print(f"mean_amp_p2: {mean_amp_p2}")
    print(f"mean_amp_p3: {mean_amp_p3}")
    mean_PR1 = int(np.median(intervals_PR1))
    mean_PR2 = int(np.median(intervals_PR2))
    mean_PR3 = int(np.median(intervals_PR3))
    # PR = int((mean_PR1 + mean_PR2 + mean_PR3) / 3)
    # mean_PR1 = PR
    # mean_PR2 = PR
    # mean_PR3 = PR
    # mean_PR1 = int(intervals_PR1.mean())
    # mean_PR2 = int(intervals_PR2.mean())
    # mean_PR3 = int(intervals_PR3.mean())
    print(f"mean_PR1: {mean_PR1}     {intervals_PR1.size}")
    print(f"mean_PR2: {mean_PR2}     {intervals_PR2.size}")
    print(f"mean_PR3: {mean_PR3}     {intervals_PR3.size}")
    # marr_amp_p1 = interp_pr(arr_amp_p1)
    # marr_amp_p2 = interp_pr(arr_amp_p2)
    # marr_amp_p3 = interp_pr(arr_amp_p3)
    # marr_amp_p1 = moving_average(marr_amp_p1, 15)
    # marr_amp_p2 = moving_average(marr_amp_p2, 15)
    # marr_amp_p3 = moving_average(marr_amp_p3, 15)

    # if intervals_PR1.size > 351:  # 21
    #     intervals_PR1 = moving_average(intervals_PR1, 351)
    # if intervals_PR2.size > 351:
    #     intervals_PR2 = moving_average(intervals_PR2, 351)
    # if intervals_PR3.size > 351:
    #     intervals_PR3 = moving_average(intervals_PR3, 351)
    intervals_PR1 = medfilt(intervals_PR1, 5)
    intervals_PR2 = medfilt(intervals_PR2, 5)
    intervals_PR3 = medfilt(intervals_PR3, 5)
    # presence_PR1[inds_PR1] = intervals_PR1
    # presence_PR2[inds_PR2] = intervals_PR2
    # presence_PR3[inds_PR3] = intervals_PR3
    # med_PR, ind_min = get_median_intervals_pr(intervals_PR1, intervals_PR2, intervals_PR3)
    # med_PR = medfilt(med_PR, 5)
    # r_pos_size = r_pos.size
    # if ind_min == 0:
    #     presence_PR1 = interp_pr(med_PR, inds_PR1, r_pos_size)
    #     presence_PR2 = interp_pr(med_PR, inds_PR1, r_pos_size)
    #     presence_PR3 = interp_pr(med_PR, inds_PR1, r_pos_size)
    # elif ind_min == 1:
    #     presence_PR1 = interp_pr(med_PR, inds_PR2, r_pos_size)
    #     presence_PR2 = interp_pr(med_PR, inds_PR2, r_pos_size)
    #     presence_PR3 = interp_pr(med_PR, inds_PR2, r_pos_size)
    # elif ind_min == 2:
    #     presence_PR1 = interp_pr(med_PR, inds_PR3, r_pos_size)
    #     presence_PR2 = interp_pr(med_PR, inds_PR3, r_pos_size)
    #     presence_PR3 = interp_pr(med_PR, inds_PR3, r_pos_size)
    r_pos_size = r_pos.size
    presence_PR1 = interp_pr(intervals_PR1, inds_PR1, r_pos_size)
    presence_PR2 = interp_pr(intervals_PR2, inds_PR2, r_pos_size)
    presence_PR3 = interp_pr(intervals_PR3, inds_PR3, r_pos_size)
    # print(f"r_pos_size: {r_pos_size}")
    # print(f"presence_PR1: {presence_PR1.size}")
    # print(f"presence_PR2: {presence_PR2.size}")
    # print(f"presence_PR3: {presence_PR3.size}")
    # presence_PR1 = truncate_win2(presence_PR1, 0.5, 200)
    # presence_PR2 = truncate_win2(presence_PR2, 0.5, 200)
    # presence_PR3 = truncate_win2(presence_PR3, 0.5, 200)
    # presence_PR1 = moving_average(presence_PR1, 50)
    # presence_PR2 = moving_average(presence_PR2, 50)
    # presence_PR3 = moving_average(presence_PR3, 50)
    presence_PR1 = presence_PR1.astype(int)
    presence_PR2 = presence_PR2.astype(int)
    presence_PR3 = presence_PR3.astype(int)
    np.save("presence_PR1.npy", presence_PR1)
    np.save("presence_PR2.npy", presence_PR2)
    np.save("presence_PR3.npy", presence_PR3)
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
    return mean_amp_p1, mean_amp_p2, mean_amp_p3, presence_PR1, presence_PR2, presence_PR3


@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars, mean_amp_p1, mean_amp_p2, mean_amp_p3, presence_PR1,
          presence_PR2, presence_PR3):  # mean_PR1, mean_PR2, mean_PR3):
    global k#bl, al, k, b, a
    b, a = butter(2, 18, btype='lowpass', fs=250)
    bl, al = butter(2, 3.5, btype='lowpass', fs=250) # (2, 2, btype='lowpass', fs=250) 3.5
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
    # buff = np.array([3, 3, 3])
    # mean_interval = (np.mean(intervals) + 400) * 0.06
    # mean_interval = (np.mean(intervals) + 400) * 1.5

    # mean_PR1_copy = mean_PR1
    # mean_PR2_copy = mean_PR2
    # mean_PR3_copy = mean_PR3
    for i in range(4, len(r_pos)):  # range(4, len(r_pos))
        start_range_interval = int(intervals[i] * 0.4)  # int(intervals[i]**0.5 * 3.5)
        # stop_range_interval = int(intervals[i] * 0.016 + 7)
        # start_range_interval = int(intervals[i] ** 0.5 * 3.7)  # int(intervals[i]**0.5 * 3.5)
        stop_range_interval = 10#int(intervals[i] ** 0.5 * 0.66)
        start_slice = r_pos[i] - start_range_interval
        end_slice = r_pos[i] - stop_range_interval

        start_slice1 = r_pos[i] - presence_PR1[i] - 15
        start_slice2 = r_pos[i] - presence_PR2[i] - 15
        start_slice3 = r_pos[i] - presence_PR3[i] - 15
        # fragment_l1 = lead1[start_slice:end_slice]
        # fragment_l2 = lead2[start_slice:end_slice]
        # fragment_l3 = lead3[start_slice:end_slice]
        # if len_pr * 0.2 > 10:
        #     end_slice = r_pos[i] - 10
        # end_slice = start_slice + 30
        shift_end = int(intervals[i] * 0.016 + 7)
        if shift_end > 15:
            shift_end = 15
        # end_slice1 = r_pos[i] - int(intervals[i] * 0.016 + 7)
        # end_slice2 = r_pos[i] - int(intervals[i] * 0.016 + 7)
        # end_slice3 = r_pos[i] - int(intervals[i] * 0.016 + 7)
        # end_slice1 = r_pos[i] - shift_end
        # end_slice2 = r_pos[i] - shift_end
        # end_slice3 = r_pos[i] - shift_end
        end_slice1 = r_pos[i] - 10#int(presence_PR1[i] ** 0.5 * 1.9)  # * 3.0
        end_slice2 = r_pos[i] - 10#int(presence_PR2[i] ** 0.5 * 1.9)  # * 3.0
        end_slice3 = r_pos[i] - 10#int(presence_PR3[i] ** 0.5 * 1.9)  # * 3.0
        if presence_PR1[i] > intervals[i] * 0.4:  # // 3
            # if len_pr < mean_PR1:
            # fragment_l1 = lead1[start_slice1:end_slice1]
            fragment_l1 = lead1[start_slice1:end_slice1]
        else:
            fragment_l1 = lead1[start_slice1:end_slice1]
        if presence_PR2[i] > intervals[i] * 0.4:
            # if len_pr < mean_PR2:
            # fragment_l2 = lead2[start_slice2:end_slice2]
            fragment_l2 = lead2[start_slice2:end_slice2]
        else:
            fragment_l2 = lead2[start_slice2:end_slice2]
        if presence_PR3[i] > intervals[i] * 0.4:
            # if len_pr < mean_PR3:
            # fragment_l3 = lead3[start_slice3:end_slice3]
            fragment_l3 = lead3[start_slice3:end_slice3]
        else:
            fragment_l3 = lead3[start_slice3:end_slice3]
        fragment_l1 = fragment_l1 - fragment_l1.mean()
        fragment_l2 = fragment_l2 - fragment_l2.mean()
        fragment_l3 = fragment_l3 - fragment_l3.mean()
        ffragment_l1 = filtfilt(b, a, fragment_l1)
        ffragment_l2 = filtfilt(b, a, fragment_l2)
        ffragment_l3 = filtfilt(b, a, fragment_l3)
        isoline1 = filtfilt(bl, al, ffragment_l1)
        isoline2 = filtfilt(bl, al, ffragment_l2)
        isoline3 = filtfilt(bl, al, ffragment_l3)
        fragment_l1 = ffragment_l1 - isoline1
        fragment_l2 = ffragment_l2 - isoline2
        fragment_l3 = ffragment_l3 - isoline3
        # fragment_l1 = fragment_l1 - fragment_l1.mean()
        # fragment_l2 = fragment_l2 - fragment_l2.mean()
        # fragment_l3 = fragment_l3 - fragment_l3.mean()
        # fragment_l1[fragment_l1 < 0.0] = 0.0
        # fragment_l2[fragment_l2 < 0.0] = 0.0
        # fragment_l3[fragment_l3 < 0.0] = 0.0
        # fragment_l1 = lead1[r_pos[i] - mean_PR1_copy - 15:r_pos[i] - mean_PR1_copy + 15]  # -+15
        # fragment_l2 = lead2[r_pos[i] - mean_PR2_copy - 15:r_pos[i] - mean_PR2_copy + 15]
        # fragment_l3 = lead3[r_pos[i] - mean_PR3_copy - 15:r_pos[i] - mean_PR3_copy + 15]
        bp = 0.0
        amp_p1 = 0.0
        amp_p2 = 0.0
        amp_p3 = 0.0
        if not (chars[i] == 'N'):
            p1[i] = 1.0
            p2[i] = 1.0
            p3[i] = 1.0
            # p1[i] = 0.6
            # p2[i] = 0.6
            # p3[i] = 0.6
        else:
            # fragment_l1_f = filtfilt(b, a, fragment_l1)
            # fragment_l1_f[fragment_l1_f < 0.0] = 0.0
            # fragment_l1_f = filtfilt(b, a, fragment_l1_f)
            # fragment_l1_f = filtfilt(bh, ah, fragment_l1_f)
            pzub, amp_pzub1 = get_p_amp2(fragment_l1, mean_amp_p1)
            p1[i] = pzub
            # ind_max1 = argrelmax(fragment_l1_f)[0]
            # if ind_max1.size == 1:  # if ind_max1.size == 1:
            #     isoline1 = np.max((np.min(fragment_l1_f[:ind_max1[-1]]), np.min(fragment_l1_f[ind_max1[-1]:])))
            #     amp_p1 = fragment_l1_f[ind_max1[-1]] - isoline1
            #     if (amp_p1 > mean_amp_p1 * 0.1) and (amp_p1 > 0.001):  # > 0.0001
            #         # if amp_p1 > 0.0015:   # > 0.0001
            #         p1[i] = 1.0
            # fragment_l2_f = filtfilt(b, a, fragment_l2)
            # fragment_l2_f[fragment_l2_f < 0.0] = 0.0
            # fragment_l2_f = filtfilt(b, a, fragment_l2_f)
            # fragment_l2_f = filtfilt(bh, ah, fragment_l2_f)
            pzub, amp_pzub2 = get_p_amp2(fragment_l2, mean_amp_p2)
            p2[i] = pzub
            # ind_max2 = argrelmax(fragment_l2_f)[0]
            # if ind_max2.size == 1:
            #     isoline2 = np.max((np.min(fragment_l2_f[:ind_max2[-1]]), np.min(fragment_l2_f[ind_max2[-1]:])))
            #     amp_p2 = fragment_l2_f[ind_max2[-1]] - isoline2
            #     if (amp_p2 > mean_amp_p2 * 0.1) and (amp_p2 > 0.001):
            #         # if amp_p2 > 0.0015:
            #         p2[i] = 1.0
            # fragment_l3_f = filtfilt(b, a, fragment_l3)
            # fragment_l3_f[fragment_l3_f < 0.0] = 0.0
            # fragment_l3_f = filtfilt(b, a, fragment_l3_f)
            # fragment_l3_f = filtfilt(bh, ah, fragment_l3_f)
            pzub, amp_pzub3 = get_p_amp2(fragment_l3, mean_amp_p3)
            p3[i] = pzub
            # ind_max3 = argrelmax(fragment_l3_f)[0]
            # if 3 > ind_max3.size >= 1:
            #     isoline3 = np.max((np.min(fragment_l3_f[:ind_max3[-1]]), np.min(fragment_l3_f[ind_max3[-1]:])))
            #     amp_p3 = fragment_l3_f[ind_max3[-1]] - isoline3
            #     if (amp_p3 > mean_amp_p3 * 0.1) and (amp_p3 > 0.001):
            #         # if amp_p3 > 0.0015:
            #         p3[i] = 1.0

            # logging.info(f"amp_p1 = {amp_p1:.4f}, amp_p2 = {amp_p2:.4f}, amp_p3 = {amp_p3:.4f}")
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
            print(f"{mean_amp_p1:.5f}, {mean_amp_p2:.5f}, {mean_amp_p3:.5f}")
            print(f"{amp_pzub1:.5f}, {amp_pzub2:.5f}, {amp_pzub3:.5f}")
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
        sum_buff = (sum_p1 * sum_p2 + sum_p1 * sum_p3 + sum_p2 * sum_p3) * 0.5
        # sum_buff = sum_p1 * sum_p2 * sum_p3 * 0.25  # 0.38
        out[i - 2] = sum_buff
    out = -(out - np.max(out))
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


def get_median_intervals_pr(intervals_PR1, intervals_PR2, intervals_PR3):
    out = np.array([])
    len1 = len(intervals_PR1)
    len2 = len(intervals_PR2)
    len3 = len(intervals_PR3)
    arr_len = np.array([len1, len2, len3])
    len_g = np.min(arr_len)
    ind_min = np.argmin(arr_len)
    for i in range(len_g):
        out = np.append(out, np.median([intervals_PR1[i], intervals_PR2[i], intervals_PR3[i]]))
    return out, ind_min


if __name__ == '__main__':
    app = QApplication(sys.argv)
    save_F_txt()
    # sys.exit(app.exec())
    sys.exit()
