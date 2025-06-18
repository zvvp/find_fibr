from PyQt6.QtWidgets import QFileDialog
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelmax, argrelmin
from scipy.interpolate import interp1d
import glob
import os
from numba import njit
# from time import time
from functools import wraps
from timeit import default_timer


def time_fun(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure execution time using default_timer
        start_time = default_timer()
        result = func(*args, **kwargs)
        execution_time = default_timer() - start_time

        # Print elapsed time
        print(f"Execution time for {func.__name__}: {execution_time:.9f} seconds")

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
        # intervals[intervals < 50] = np.mean(intervals)
        # intervals[intervals > 500] = np.mean(intervals)

    return r_pos, intervals, chars, forms

# @time_fun
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
                # temp = int(line.split(';')[0])
                # r_pos.append(temp)
                # temp = int(line.split(';')[1])
                # intervals.append(temp)
                # temp = line.split(';')[2][0]
                # chars.append(temp)
                # temp = int(line.split(':')[1])
                # forms.append(temp)
        r_pos = np.array(r_pos)
        intervals = np.array(intervals)
        mean_interval = np.mean(intervals)
        chars = np.array(chars)
        forms = np.array(forms)
        intervals[intervals < 50] = mean_interval
        intervals[intervals > 550] = mean_interval
        end_pos = int(len(r_pos) * 0.99)
    return r_pos[: end_pos], intervals[: end_pos], chars[: end_pos], forms[: end_pos]
    # return r_pos, intervals, chars, forms

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
                ref_t = np.array([tf[1], tf[1] * 0.55, tf[1] * 1.5, tf[1]])
                ref_t0 = np.array([tf[1], tf[1] * 0.55, tf[1], tf[1] * 0.55])
                ref_t1 = np.array([tf[1], tf[1] * 1.5, tf[1] , tf[1] * 1.5])
                # ref_t3 = np.array([tf[1], tf[1] * 1.6, tf[1] * 0.65, tf[1]])
                # ref_t4 = np.array([tf[1], tf[1] * 1.6, tf[1] * 0.5, tf[1]])
                coef_cor = get_coef_cor(ref_t, tf[1:])
                coef_cor0 = get_coef_cor(ref_t0, tf[1:])
                coef_cor1 = get_coef_cor(ref_t1, tf[1:])
                # coef_cor3 = get_coef_cor(ref_t3, tf[1:])
                # coef_cor4 = get_coef_cor(ref_t4, tf[1:])
                arr_cor = np.array([coef_cor, coef_cor0, coef_cor1])
                trs = 0.985  # 0.95
                if np.max(arr_cor) > trs:
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


def del_V_S(intervals, chars):
    len_in = len(intervals)
    out = intervals.copy()
    diff_intervals = np.abs(intervals - np.roll(intervals, 1))[1:]
    # mean_diff = np.mean(diff_intervals)
    # print(f"mean_diff = {mean_diff:.2f}")
    # diff_intervals = np.abs(diff_intervals - np.roll(diff_intervals, 1))[1:]
    # mean_diff = np.mean(diff_intervals)
    # print(f"mean_diff = {mean_diff:.2f}")
    # diff_intervals = np.abs(diff_intervals - np.roll(diff_intervals, 1))[1:]
    # mean_diff = np.mean(diff_intervals)
    # print(f"mean_diff = {mean_diff:.2f}")
    for i in np.arange(5, len_in - 5):
        max_diff = np.max(diff_intervals[i - 2:i + 2])
        if 'V' in chars[i] and (max_diff > 70.0):
            mean_interval = np.mean(intervals[i - 5:i + 5])
            out[i - 1:i + 2] = mean_interval + (intervals[i - 1:i + 2] - mean_interval) * 0.2
            # bp = 0
        elif 'Q' in chars[i] and (max_diff > 40.0):
            mean_interval = np.mean(intervals[i - 5:i + 5])
            out[i - 1:i + 2] = mean_interval + (intervals[i - 1:i + 2] - mean_interval) * 0.1
    return out


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
    for i in np.arange(25, len_in - 26):  # np.arange(15, len_in - 16)  (35, len_in - 36)
        win_t = intervals[i - 25:i + 26].copy()
        # sort_win_t = np.sort(win_t)[:-10]   # [:-12]
        # mean_win = np.median(sort_win_t)
        diff_t = np.abs(win_t - np.roll(win_t, 1))[1:]
        diff_t2 = np.abs(diff_t - np.roll(diff_t, 1))[1:]
        # diff_t2 = np.abs(diff_t2 - np.roll(diff_t2, 1))[1:]
        diff_t2 = np.sort(diff_t2)[:-12]   # [:-12]
        dt2_mean = np.mean(diff_t2)
        # if diff_t2[diff_t2 > dt2_mean].size > 0:
        #     threshold = np.mean(diff_t2[diff_t2 > dt2_mean])
        #     diff_t2 = diff_t2[diff_t2 < threshold]
        # mean_diff_t2 = diff_t2.std() * (1.5 + 200 / mean_win)  # (1.9 + 210 / mean_win)
        out[i] = dt2_mean #* (1 + 100.0 / mean_win) # dt2_mean * 175.0 / mean_win
        tempvar = 0
        out[i-2] = np.mean(out[i-5:i+1])
    # out[:35] = np.mean(out[30:50])
    # out[-36:] = np.mean(out[-50:-30])
    out[:26] = out[26]
    out[-26:] = out[-26]
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
    diff_loc = loc_arr[1:] - loc_arr[:-1]
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
        min_amp = np.min([amp_pzub1, amp_pzub2])
        div_amp = max_amp / min_amp
        if div_amp > 10.0:
            amp_pzub = max_amp
    if (amp_pzub > mean_amp_p * 0.06) and (amp_pzub > 0.002):   # (amp_pzub > mean_amp_p * 0.05) and (amp_pzub > 0.001):
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







