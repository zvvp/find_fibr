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
    # b, a = butter(1, 4, 'low', fs=250)
    start = 121
    stop = 122
    b, a = butter(2, 14, 'low', fs=250)   # 2, 14, 'low', fs=250
    bh, ah = butter(1, 0.7, 'high', fs=250)   # 1, 0.7, 'high', fs=250
    with open("C:/EcgVar/B1.txt", "r+") as f:    
        lines = f.readlines()
        f.seek(0)
        for i, line in enumerate(lines):
            if (i > 14) and (i < len(lines) - 2): # i > 13   i < len(lines) - 1
                stop = int(line.split(';')[0])
                # form_2 = int(lines[i-2].split(':')[1])
                form_1 = int(lines[i-1].split(':')[1])
                form = int(line.split(':')[1])
                # form1 = int(lines[i+1].split(':')[1])
                if (';N' in line):
                    period_2 = int(lines[i-2].split(';')[1])
                    period_1 = int(lines[i-1].split(';')[1])
                    period = int(line.split(';')[1])
                    period1 = int(lines[i+1].split(';')[1])
                    period2 = int(lines[i+2].split(';')[1])
                    t = np.array([period_2, period_1, period, period1, period2])
                    mean_t = np.mean(t[:-3])  # mean(t[:-1])
                    std_t = np.std(t)
                    dp_1_2 = period_1 - period_2
                    dp0_1 = period - period_1
                    dp10 = period1 - period
                    # kv1 = period2 - period_1
                    per_std_t = std_t        
                    begin, end = get_begin_end(start, stop)
                    offset = get_offset(stop, lead1, lead2, lead3)
                    begin = begin - offset
                    end = end - offset
                    if (period > 30) and (period < 900):
                        if (t[2] / t[1] < 0.95) \
                                and (t[3] > t[1] * 1.05) and (np.abs(dp_1_2) < 9):
                            if (form == 0):
                                line = line.replace(';N', ';A')
                                lines[i+1] = lines[i+1].replace(';N', ';A')
                            elif (form_1 == 0):
                                line = line.replace(';N', ';A')
                                lines[i-1] = lines[i-1].replace(';N', ';A')
                            else:
                                line = line.replace(';N', ';S')
                        elif (t[2] / t[1] < 0.9) and (('S' in lines[i-2]) or ('V' in lines[i-2])) \
                                and (dp10 > -dp0_1 * 0.9) and (dp0_1 < 0):
                            if (form == 0):
                                line = line.replace(';N', ';A')
                                lines[i+1] = lines[i+1].replace(';N', ';A')
                            elif (form_1 == 0):
                                line = line.replace(';N', ';A')
                                lines[i-1] = lines[i-1].replace(';N', ';A')
                            else:
                                line = line.replace(';N', ';S')
                        
                        else:
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
                            if (per_std_t > 300.0) and ((sum_peaks12 == 0) or (sum_peaks13 == 0) or (sum_peaks23 == 0)):
                                line = line.replace(';N', ';F')
            start = stop
            f.write(line)
        f.truncate()

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
    if np.std(fragment) > 0:
        peaks = (np.diff(np.sign(np.diff(fragment))) < 0).nonzero()[0] + 1
        len_peaks = len(peaks)
    else:
        len_peaks = 100  # np.nan
    return len_peaks



