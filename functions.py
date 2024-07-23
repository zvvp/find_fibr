from PyQt6.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt


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
    # pos_h = start_h
    # pos_m = start_m
    # pos_s = start_s
    # pos_h = 19
    # pos_m = 43
    # pos_s = 00
    n = ((pos_h - start_h) * 3600 + (pos_m - start_m) * 60 + pos_s - start_s) * 250
    return n

def get_r_pos():
    out = []
    with open("C:/EcgVar/B.txt", "r") as f:
        for line in f:
            if ';' in line:
                r = int(line.split(';')[0])
                out.append(r)
    return out

def del_isoline(ch):
    isoline = medfilt(ch, 151)
    isoline = savgol_filter(isoline, 51, 0)
    out = ch - isoline
    out = savgol_filter(out, 3, 0)
    return out

def clean_ch(ch):
    isoline = medfilt(ch, 91)
    out = ch - isoline
    b, a = butter(2, 35, 'lp', fs=250) # 2, 20, 'lp', fs=250
    out = filtfilt(b, a, out)
    return out

def get_fibr(lead1, lead2, lead3):
    start = 121
    with open("C:/EcgVar/B.txt", "r+") as f:
        lines = f.readlines()
        # mean_period = lines[6].split(' ')[1]
        # mean_period = mean_period.split('\n')[0]
        # mean_period = int(float(mean_period))
        f.seek(0)
        for i, line in enumerate(lines):
            if (i > 13) and (i < len(lines) - 1): # i > 13   i < len(lines) - 1
                # if ';N' in line:
                    # if 'N' in lines[i + 1]:
                period1 = int(lines[i-1].split(';')[1])
                period2 = int(lines[i+1].split(';')[1])
                period = int(line.split(';')[1])
                t = np.array([period1, period, period2])
                std_t = np.std(t)
                mean_t = np.mean(t)
                # mean_d = np.mean([np.abs(period1 - period), np.abs(period1 - period2), np.abs(period - period2)])
                stop = int(line.split(';')[0])
                if (period > 50) and (period < mean_t * 2.0):
                    fragment1 = get_fragment(start, stop, lead1)
                    fragment2 = get_fragment(start, stop, lead2)
                    fragment3 = get_fragment(start, stop, lead3)
                    number_of_peaks1 = get_number_of_peaks(fragment1)
                    number_of_peaks2 = get_number_of_peaks(fragment2)
                    number_of_peaks3 = get_number_of_peaks(fragment3)
                    kv = (period2 - period) / (period2 + period) * 100.0
                    mean_kv_std = np.mean([kv, std_t])
                    sum_kv_std = kv + std_t
                    if (kv <= 5) and (number_of_peaks1 + number_of_peaks2 + number_of_peaks3 == 0):
                        line = line.replace(';N', ';F')
                    elif (5 < kv < 15) and ((number_of_peaks1 + number_of_peaks2 == 0) \
                                    or (number_of_peaks1 + number_of_peaks3 == 0) or (number_of_peaks2 + number_of_peaks3 == 0)):
                        line = line.replace(';N', ';F')
                start = stop
            f.write(line)
        f.truncate()

def get_fragment(start, stop, lead):
    t = stop - start
    shift = 32
    k1 = 0.8
    k2 = 0.615  # 0.63
    start = start + int((t * k1 + shift) * k2)
    # start = start + int(0.35 * t) + 36
    stop = stop - int(0.065 * t)
    fragment = lead[start:stop]  # start + int(0.65 * t)+4:stop - int(0.04 * t)
    # b, a = butter(2, 11, 'low', fs=250)
    # bh, ah = butter(1, 0.08, 'high', fs=250)
    b = [0.01591456, 0.03182911, 0.01591456]
    a = [1.0, -1.61277905,  0.67643727]
    bh = [0.9989957, -0.9989957]
    ah = [1.0, -0.9979914]
    if fragment.size > 9:
        fragment = filtfilt(b, a, fragment)
        fragment = filtfilt(bh, ah, fragment)
    # fragment = fragment[int(0.65 * t)+4:]
    return fragment

def get_number_of_peaks(fragment):
    peaks = (np.diff(np.sign(np.diff(fragment))) < 0).nonzero()[0] + 1
    return len(peaks)
