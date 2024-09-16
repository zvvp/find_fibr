from PyQt6.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelextrema
from scipy.stats import pearsonr
import glob


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
    stop = 122
    b, a = butter(2, 14, 'low', fs=250)   # 2, 14, 'low', fs=250
    bh, ah = butter(1, 0.7, 'high', fs=250)   # 1, 0.7, 'high', fs=250
    with open("C:/EcgVar/B.txt", "r") as f:    
        lines = f.readlines()
        ref_t = np.array([200, 100, 300, 200])   # 200, 160, 240, 200
        for i, line in enumerate(lines):
            if (i > 14) and (i < len(lines) - 2): # i > 13   i < len(lines) - 1
                stop = int(line.split(';')[0])
                form_1 = int(lines[i-1].split(':')[1])
                form = int(line.split(':')[1])
                if (';N' in line):
                    period_2 = int(lines[i-2].split(';')[1])
                    period_1 = int(lines[i-1].split(';')[1])
                    period = int(line.split(';')[1])
                    period1 = int(lines[i+1].split(';')[1])
                    period2 = int(lines[i+2].split(';')[1])
                    tf = np.array([period_2, period_1, period, period1, period2])
                    diff_t = np.array([np.abs(period_1 - period_2), np.abs(period - period_1), \
                                       np.abs(period1 - period), np.abs(period2 - period1)])
                    t = np.array([period_1, period, period1, period2])
                    max_t = np.max(t)
                    min_t = np.min(t)
                    mean_t = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
                    # mean_t = (tf[0] + tf[1] + tf[4]) / 3
                    # k_fibr = (np.abs(tf[0] - tf[1]) + np.abs(tf[0] - tf[4]) + np.abs(tf[1] - tf[4])) / mean_t
                    k_fibr = (np.sum(diff_t) - np.max(diff_t) - np.min(diff_t)) / mean_t     
                    if (min_t > 50) and (max_t < mean_t * 2):
                        coef_cor = pearsonr(ref_t, t)[0]
                        if (coef_cor > 0.9) and (t[1] < t[0] * 0.97) and (t[2] > t[0] * 1.03) and (k_fibr < 2.0):  #  (coef_cor > 0.9) and (t[1] < t[0] * 0.97) and (t[2] > t[0] * 1.03) and (k_fibr < 2.0)
                            if (form == 0):
                                lines[i] = lines[i].replace(';N', ';A')
                                lines[i+1] = lines[i+1].replace(';N', ';A')
                            elif (form_1 == 0):
                                lines[i] = lines[i].replace(';N', ';A')
                                lines[i-1] = lines[i-1].replace(';N', ';A')
                            else:
                                lines[i] = lines[i].replace(';N', ';S')
                        elif (k_fibr > 0.25) and (k_fibr < 0.97):     #   (k_fibr > 0.5) and (coef_cor < 0.8)
                            begin, end = get_begin_end(start, stop)
                            offset = get_offset(stop, lead1, lead2, lead3)
                            begin = begin - offset
                            end = end - offset
                            fragment1 = lead1[begin:end]
                            fragment2 = lead2[begin:end]
                            fragment3 = lead3[begin:end]
                            fragment1 = filtfilt(b, a, fragment1)
                            fragment2 = filtfilt(b, a, fragment2)
                            fragment3 = filtfilt(b, a, fragment3)
                            fragment1 = filtfilt(bh, ah, fragment1)
                            fragment2 = filtfilt(bh, ah, fragment2)
                            fragment3 = filtfilt(bh, ah, fragment3)                          
                            number_of_peaks1 = get_number_of_peaks(fragment1)
                            number_of_peaks2 = get_number_of_peaks(fragment2)
                            number_of_peaks3 = get_number_of_peaks(fragment3)
                            sum_peaks12 = number_of_peaks1 + number_of_peaks2
                            sum_peaks13 = number_of_peaks1 + number_of_peaks3
                            sum_peaks23 = number_of_peaks2 + number_of_peaks3
                            if (sum_peaks12 == 0) or (sum_peaks13 == 0) or (sum_peaks23 == 0):
                                if (';N' in line):
                                    if (form == 0):
                                        lines[i] = lines[i].replace(';N', ';A')
                                        lines[i+1] = lines[i+1].replace(';N', ';A')
                                    elif (form_1 == 0):
                                        lines[i] = lines[i].replace(';N', ';A')
                                        lines[i-1] = lines[i-1].replace(';N', ';A')
                                    elif ((';N' in lines[i-2]) or (';F' in lines[i-2])) and ((';N' in lines[i-1]) or (';F' in lines[i-1])) and (';N' in lines[i+1]) and (';N' in lines[i+2]):
                                        time_qrs = get_time_from_addr(line)
                                        lines[i] = lines[i][:-1] + time_qrs
                                        lines[i] = lines[i].replace(';N', ';F')
            start = stop
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
        begin = stop - len_fragment - 7  #  -12
        end = begin + len_fragment
    elif 150 > period >= 120:
        len_fragment = int(np.round(0.28 * period))
        begin = stop - len_fragment - 7  #  -12
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

def get_number_of_peaks(fragment):
    mean_frg = np.mean(fragment)
    local_max_idx = argrelextrema(fragment, np.greater)[0]
    local_max_values = fragment[local_max_idx]

    if len(local_max_values) > 0:
        max_peak = np.max(local_max_values)
    else:
        max_peak = 0
    if (max_peak - mean_frg) > 0.022:
        return 1
    else:
        return 0
    




