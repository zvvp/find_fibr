from PyQt6.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelextrema
from scipy.stats import pearsonr
import glob
import os
from numba import njit
from time import time


def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n
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

def get_start_shift(pos_h, pos_m, pos_s):
    fname = QFileDialog.getOpenFileName()[0]
    print(fname)
    with open(fname, "rb") as f:
        head = f.read(1024)
        if head[151] == ":":
            start_h = int(head[150])
            start_m = int(head[152:154])
            start_s = int(head[155:157])
        else:
            start_h = int(head[150:152])
            start_m = int(head[153:155])
            start_s = int(head[156:158])
    n = ((pos_h - start_h) * 3600 + (pos_m - start_m) * 60 + pos_s - start_s) * 250
    return n

def get_r_pos():
    out = []
    with open("C:/EcgVar/B.txt", "r") as f:
        for line in f:
            if ';' in line:
                r = int(line.split(';')[0])
                out.append(r)
    return np.array(out)

def get_intervals():
    out = []
    with open("C:/EcgVar/B.txt", "r") as f:
        for line in f:
            if ';' in line:
                interval = int(line.split(';')[1])
                out.append(interval)
    return np.array(out)

# @time_fun
def parse_B_txt():
    r_pos = []
    intervals = []
    chars = []
    forms = []
    with open("C:/EcgVar/B1.txt", "r") as f:
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
        intervals[intervals < 50] = np.mean(intervals)
        intervals[intervals > 500] = np.mean(intervals)

    return r_pos, intervals, chars, forms

def del_isoline(ch):
    isoline = medfilt(ch, 151)
    isoline = savgol_filter(isoline, 51, 0)
    out = ch - isoline
    out = savgol_filter(out, 3, 0)
    return out

def clean_ch(ch):
    isoline = medfilt(ch, 91)
    out = ch - isoline
    b, a = butter(2, 35, 'lp', fs=250)  # 2, 20, 'lp', fs=250
    out = filtfilt(b, a, out)
    return out

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

@njit
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
            if (';N' in lines[i]) and (not ';V' in lines[i - 1]) and (not ';S' in lines[i - 1]):
                periods = get_periods(lines[i - 2:i + 3])
                tf = np.array(periods)
                ref_t = np.array([tf[1], tf[1] * 0.8, tf[1] * 1.2, tf[1]])
                ref_t0 = np.array([tf[1], tf[1] * 0.65, tf[1] * 1.15, tf[1]])
                ref_t1 = np.array([tf[1], tf[1] * 0.65, tf[1] * 1.1, tf[1] * 0.65])
                ref_t3 = np.array([tf[1], tf[1] * 1.6, tf[1], tf[1] * 1.6])
                ref_tA = np.array([tf[1], tf[1] * 0.5, tf[1] * 0.5, tf[1]])
                ref_tA1 = np.array([tf[1], tf[1] * 0.65, tf[1] * 0.35, tf[1]])
                ref_tA2 = np.array([tf[1], tf[1] * 0.35, tf[1] * 0.65, tf[1]])
                # max_t = np.max(tf)
                # min_t = np.min(tf)
                mean_t = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
                if (tf[2] > 50) and (tf[3] > 50) and (tf[2] < mean_t * 2) and (tf[3] < mean_t * 2):
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

def get_begin_end(start, stop):
    period = stop - start
    if period < 80:
        len_fragment = int(np.round(0.2 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 90 > period >= 80:
        len_fragment = int(np.round(0.22 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 100 > period >= 90:
        len_fragment = int(np.round(0.25 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 120 > period >= 100:
        len_fragment = int(np.round(0.28 * period))
        begin = stop - len_fragment - 7  # -12
        end = begin + len_fragment
    elif 150 > period >= 120:
        len_fragment = int(np.round(0.28 * period))
        begin = stop - len_fragment - 7  # -12
        end = begin + len_fragment
    elif 200 > period >= 150:
        len_fragment = int(np.round(0.29 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 250 > period >= 200:
        len_fragment = int(np.round(0.3 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif period >= 250:
        len_fragment = int(np.round(0.3 * period))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    return begin, end

# def get_fragment(start, stop, lead):
#     # t = stop - start
#     # shift = 46   # 33 49
#     # k1 = 0.7    # 0.8
#     # k2 = 0.6  # 0.63  0.615
#     # start = start + int((t * k1 + shift) * k2)
#     # stop = stop - int(0.075 * t)  # int(0.065 * t) int(0.07 * t)
#     start = stop - 38  # stop - 40
#     stop = stop - 11  # stop - 13
#     fragment = lead[start:stop]  # start + int(0.65 * t)+4:stop - int(0.04 * t)
#     b, a = butter(2, 31, 'low', fs=250)
#     # b, a = butter(2, 11, 'low', fs=250)
#     # b = [0.01591456, 0.03182911, 0.01591456]
#     # a = [1.0, -1.61277905,  0.67643727]
#     # bh, ah = butter(1, 0.08, 'high', fs=250)
#     bh = [0.9989957, -0.9989957]
#     ah = [1.0, -0.9979914]
#     if fragment.size > 9:
#         fragment = filtfilt(b, a, fragment)
#         fragment = filtfilt(bh, ah, fragment)
#     return fragment

def get_offset(stop, ch1, ch2, ch3):
    sum_slice = ch1[stop - 7: stop] + ch2[stop - 7: stop] + ch3[stop - 7: stop]
    offset = 7 - np.argmax(sum_slice)
    return offset

def get_number_of_peaks1(fragment):
    mean_frg = np.mean(fragment)
    local_max_idx = []
    for i in range(3, fragment.size - 3):
        if ((fragment[i] - fragment[i - 3]) > 0.002) and ((fragment[i] - fragment[i + 3]) > 0.002):
        # if fragment[i] > fragment[i - 1] and fragment[i] > fragment[i + 1]:
            local_max_idx.append(i)
    local_max_values = fragment[local_max_idx]
    if len(local_max_values) == 1:
        return 1
    elif len(local_max_values) == 2:
        return 1
    else:
        return 0

    # if len(local_max_values) == 1:  # if len(local_max_values) > 0:
    #     max_peak = np.max(local_max_values)
    # elif len(local_max_values) == 2:
    #     max_peak = np.max(local_max_values)
    # else:
    #     max_peak = mean_frg
    # # if max_peak >
    # if (max_peak - mean_frg) * 0.2 > 0.0:  # 0.022
    #     return 1
    # else:
    #     return 0

def get_number_of_peaks(fragment):
    for i in range(3, fragment.size - 3):
        if ((fragment[i] - fragment[i - 3]) > 0.003) and ((fragment[i] - fragment[i + 3]) > 0.003): # 0.002
            return 1
    return 0

def del_V_S(intervals, chars):
    len_in = len(intervals)
    out = intervals.copy()
    for i in np.arange(3, len_in - 3):
        if ('V' in chars[i]) or ('S' in chars[i]) or ('A' in chars[i]):
            out[i:i + 2] = np.mean(intervals[i:i + 2])
    return out

def get_coef_fibr(intervals):
    len_in = len(intervals)
    out = np.zeros(len_in)
    for i in np.arange(3, len_in - 4):
        win_t = intervals[i - 3:i + 4]
        win_t = np.sort(win_t)[1:-1]
        mean_win_t = np.mean(win_t[1:-1])
        diff_t = np.abs(win_t - np.roll(win_t, 1))
        diff_t = diff_t[1:]
        diff_t = np.sort(diff_t)
        sum_diff_tf = np.sum((diff_t[:-1] * 10)**2)
        # win_t = np.sort(win_t)
        # out[i] = sum_diff_tf + 5000/mean_win_t
        out[i] = sum_diff_tf / mean_win_t
    # mean_out = np.mean(out)
    return out
    # return (out - mean_out) * 0.8 + mean_out * 1.6

def detect(arr, win):
    out = np.zeros(len(arr))
    w = win // 2
    for i in range(w, len(arr) - w):
        out[i] = np.max(arr[i - w:i + w])
    return out

def sum3(arr):
    out = np.zeros(len(arr))
    for i in range(1, len(arr) - 1):
        out[i] = np.sum(arr[i - 1:i + 2])
    return out

def mean3(arr):
    out = np.zeros(len(arr))
    for i in range(2, len(arr) - 2):
        win = arr[i - 2:i + 3]
        win = np.sort(win)
        out[i] = np.mean(win[1:-1])
    return out

def get_ranges_fibr(fintervals, fcoef_fibr, r_pos):
    start = []
    stop = []
    temp = 0
    w = 7500   # 2150
    for i in range(1, len(fintervals)):
        if (fcoef_fibr[i-1] <= fintervals[i-1]) and (fcoef_fibr[i] > fintervals[i]):
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
        elif (fcoef_fibr[i-1] >= fintervals[i-1]) and (fcoef_fibr[i] < fintervals[i]):
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

def get_fname():
    fname = ""
    fname1 = ""
    dir = "d:/Kp_01"
    for file in os.listdir(dir):
        if file.endswith(".ecg"):
            fname = os.path.join(dir, file)
            fname1 = fname[9:]
            # print(fname1)
    with open("C:/EcgVar/B1.txt", "r") as f:
        lines = f.readlines()
    fname3 = lines[2].strip()
    fname2 = fname3[10:-1]
    # print(fname2)
    if fname1 == fname2:
        return fname

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
    return start_h, start_m, start_s

def get_time_qrs(addr, start_time):
    s = addr * 4 // 1000
    m = s // 60
    s = s % 60
    s = s + start_time[2]
    if s >= 60:
        s = s - 60
        m = m + 1
    h = m // 60
    m = m % 60
    m = m + start_time[1]
    if m >= 60:
        m = m - 60
        h = h + 1
    d = h // 24
    h = h % 24
    h = h + start_time[0]
    if h >= 24:
        h = h - 24
        d = d + 1
    # d = d + 1
    h = 24 * d + h
    # return f"{d:02d} день {h:02d}:{m:02d}:{s:02d}"
    return h, m, s

def get_diff_time(start, stop):
    h1, m1, s1 = start
    h2, m2, s2 = stop
    if s1 <= s2:
        diff_s = s2 - s1
    else:
        diff_s = s2 + 60 - s1
        m2 -= 1
    if m1 <= m2:
        diff_m = m2 - m1
    else:
        diff_m = m2 + 60 - m1
        h2 -= 1
    if h1 <= h2:
        diff_h = h2 - h1
    else:
        diff_h = h2 + 24 - h1

    return diff_h, diff_m, diff_s

def moving_average(data, window_size):
    out = np.zeros(data.size)
    for i in range(window_size // 2, len(data) - window_size // 2):
        out[i] = np.mean(data[i - window_size // 2:i + window_size // 2])
    return out

def correct_fibr(p_coef_fibr, fintervals, window_size):
    len_in = len(p_coef_fibr)
    out = p_coef_fibr.copy()
    win = window_size // 2
    for i in range(win, len_in - win):
        mean_fibr = np.mean(p_coef_fibr[i - win:i + win])
        mean_win = np.mean(fintervals[i - win:i + win])
        if mean_fibr > mean_win:
            out[i] = (p_coef_fibr[i] - mean_fibr) * 0.8 + mean_fibr * 1.1
        else:
            out[i] = (p_coef_fibr[i] - mean_fibr) * 0.8 + mean_fibr * 0.9
    return out
