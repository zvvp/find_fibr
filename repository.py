from PyQt6.QtWidgets import QFileDialog
import numpy as np
import os
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelmax, argrelmin


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

def del_isoline(ch):
    """
    Функция удаляет изолинию из сигнала.
    Параметры:
    ch (numpy.ndarray): Входной сигнал.
    Возвращает:
    numpy.ndarray: Сигнал без изолинии.
    """
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

def get_offset(stop, ch1, ch2, ch3):
    sum_slice = ch1[stop - 7: stop] + ch2[stop - 7: stop] + ch3[stop - 7: stop]
    offset = 7 - np.argmax(sum_slice)
    return offset

def get_number_of_peaks(fragment, n):
    gnp = get_number_of_peaks
    if not hasattr(gnp, "counter1"):
        gnp.counter1 = 0
    if not hasattr(gnp, "sum1"):
        gnp.sum1 = 0
    if not hasattr(gnp, "mean1"):
        gnp.mean1 = 0
    if not hasattr(gnp, "counter2"):
        gnp.counter2 = 0
    if not hasattr(gnp, "sum2"):
        gnp.sum2 = 0
    if not hasattr(gnp, "mean2"):
        gnp.mean2 = 0
    if not hasattr(gnp, "counter3"):
        gnp.counter3 = 0
    if not hasattr(gnp, "sum3"):
        gnp.sum3 = 0
    if not hasattr(gnp, "mean3"):
        gnp.mean3 = 0
    k1 = 10.0
    k2 = 0.01
    # k3 = 0.005
    diff_loc = get_diff_loc(fragment)
    if (diff_loc.size == 0) or (diff_loc.size > 2):
        return 0
    max_diff = np.max(diff_loc)
    if max_diff > 0.001:
        if n == 1:
            gnp.counter1 += 1
            gnp.sum1 += max_diff
            gnp.mean1 = gnp.sum1 / gnp.counter1
            if (gnp.mean1 * k1 > max_diff > gnp.mean1 * k2):
                return 1
        if n == 2:
            gnp.counter2 += 1
            gnp.sum2 += max_diff
            gnp.mean2 = gnp.sum2 / gnp.counter2
            if (gnp.mean2 * k1 > max_diff > gnp.mean2 * k2):
                return 1
        if n == 3:
            gnp.counter3 += 1
            gnp.sum3 += max_diff
            gnp.mean3 = gnp.sum3 / gnp.counter3
            if (gnp.mean3 * k1 > max_diff > gnp.mean3 * k2):
                return 1
    return 0

    #     if n == 1:
    #         gnp.counter1 += 2
    #         gnp.sum1 += (front_peak + back_peak)
    #         gnp.mean1 = gnp.sum1 / gnp.counter1
    #         if (front_peak > gnp.mean1 * k2) and (back_peak > gnp.mean1 * k2):
    #         # if (gnp.mean1 * k1 > front_peak > gnp.mean1 * k2) and (gnp.mean1 * k1 > back_peak > gnp.mean1 * k2):
    #             return 1
    #     elif n == 2:
    #         gnp.counter2 += 2
    #         gnp.sum2 += (front_peak + back_peak)
    #         gnp.mean2 = gnp.sum2 / gnp.counter2
    #         if (front_peak > gnp.mean2 * k2) and (back_peak > gnp.mean2 * k2):
    #         # if (gnp.mean2 * k1 > front_peak > gnp.mean2 * k2) and (gnp.mean2 * k1 > back_peak > gnp.mean2 * k2):
    #             return 1
    #     elif n == 3:
    #         gnp.counter3 += 2
    #         gnp.sum3 += (front_peak + back_peak)
    #         gnp.mean3 = gnp.sum3 / gnp.counter3
    #         if (front_peak > gnp.mean3 * k2) and (back_peak > gnp.mean3 * k2):
    #         # if (gnp.mean3 * k1 > front_peak > gnp.mean3 * k2) and (gnp.mean3 * k1 > back_peak > gnp.mean3 * k2):
    #             return 1
    # return 0

def sigmoid(arr, k):
    x_arr = arr.copy()
    threshold = get_threshold(x_arr)
    x_arr = x_arr - threshold
    for i in range(len(x_arr)):
        x_arr[i] = 1.0 / (1.0 + np.exp(-x_arr[i] * k)) #* max_arr  # * 2.0
    return x_arr

def over_range(arr):
    mean_arr = np.mean(arr)
    over_mean = np.mean(arr[arr > mean_arr])
    over_mean = np.mean(arr[arr > over_mean])
    # over_mean = np.mean(arr[arr > over_mean])
    return over_mean

def get_threshold(arr):
    mean_arr = np.mean(arr)
    over_mean = np.mean(arr[arr > mean_arr])
    over_mean = np.mean(arr[arr > over_mean])
    under_mean = np.mean(arr[arr < mean_arr])
    under_mean = np.mean(arr[arr < under_mean])
    return (over_mean + under_mean) / 2.0

def detect(arr, win):
    out = np.zeros(len(arr))
    w = win // 2
    for i in range(w, len(arr) - w):
        out[i] = np.max(arr[i - w:i + w])
    return out

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
    return start_h, start_m, start_s  # start_addr


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

def get_mean_line(data):
    mean_max = np.mean(data[data > np.mean(data)])
    mean_max = np.mean(data[data > mean_max])
    mean_min = np.mean(data[data < np.mean(data)])
    mean_min = np.mean(data[data < mean_min])
    return (mean_max + mean_min) / 2

def get_p2p0(data, win_size):
    out = np.zeros(len(data))
    half_size = win_size // 2
    for i in range(half_size, len(data) - half_size):
        win = data[i - half_size:i + half_size]
        out[i] = np.max(win) - np.min(win)
    return out

def truncate_win1(ch, k, win_size):
    in_ch = ch.copy()
    out = ch.copy()
    half_win = win_size // 2
    for i in range(half_win, len(in_ch) - half_win, 5):
        buff = out[i - half_win:i + half_win]
        mean_buff = np.mean(buff)
        buff = mean_buff + (buff - mean_buff) * k
        out[i - half_win:i + half_win] = buff
    return out

def get_p2p(ch, win):
    len_ch = ch.size
    p2p = np.zeros(len_ch)
    for i in range(win, len_ch - win):
        win_p2p = np.ptp(ch[i - win:i + win])
        p2p[i] = win_p2p
    return p2p