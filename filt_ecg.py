from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter, iirpeak, lfilter, filtfilt, butter, medfilt
from functions import get_number_of_peaks, get_offset


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
p2 = pg.plot()
p2.showGrid(x=True, y=True)
p1 = pg.plot()
p1.showGrid(x=True, y=True)

ch1 = np.load("d:/Kp_01/clean_lead1.npy")
ch2 = np.load("d:/Kp_01/clean_lead2.npy")
ch3 = np.load("d:/Kp_01/clean_lead3.npy")
# ch = ch1.copy()

def get_p2p(signal, window_size):
    peak_to_peak = []
    for i in range(len(signal)):
        window = signal[i:i+window_size]
        peak_to_peak.append(np.max(window) - np.min(window))
    return np.array(peak_to_peak)

def del_isoline(ch):
    isoline = medfilt(ch, 41)
    out = ch - isoline
    return out
def get_begin_end(start, stop):
    t = stop - start
    if t < 80:
        len_fragment = int(np.round(0.2 * t))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 90 > t >= 80:
        len_fragment = int(np.round(0.22 * t))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 100 > t >= 90:
        len_fragment = int(np.round(0.25 * t))  #  np.round(0.20 * t)
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 120 > t >= 100:
        len_fragment = int(np.round(0.28 * t))
        begin = stop - len_fragment - 7   #  -12
        end = begin + len_fragment
    elif 150 > t >= 120:
        len_fragment = int(np.round(0.28 * t))
        begin = stop - len_fragment - 7   #  -12
        end = begin + len_fragment
    elif 200 > t >= 150:
        len_fragment = int(np.round(0.29 * t))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif 250 > t >= 200:
        len_fragment = int(np.round(0.3 * t))
        begin = stop - len_fragment - 7
        end = begin + len_fragment
    elif t >= 250:
        len_fragment = int(np.round(0.3 * t))
        begin = stop - len_fragment - 7
        end = begin + len_fragment      
    return begin, end

start = 30417
stop = 30682
print(stop - start)

p1.plot(ch1[start-600:stop+600])
p1.plot(ch2[start-600:stop+600]-2)
p1.plot(ch3[start-600:stop+600]-4)

begin, end = get_begin_end(start, stop)
offset = get_offset(stop, ch1, ch2, ch3)
begin = begin - offset
end = end - offset
fragment1 = ch1[begin:end]
fragment2 = ch2[begin:end]
fragment3 = ch3[begin:end]

b, a = butter(2, 14, 'low', fs=250) #  butter(2, 14, 'low', fs=250)
bh, ah = butter(1, 0.7, 'high', fs=250) # 1, 0.7, 'high', fs=250
fragment1 = filtfilt(b, a, fragment1)
fragment1 = filtfilt(bh, ah, fragment1)
fragment2 = filtfilt(b, a, fragment2)
fragment2 = filtfilt(bh, ah, fragment2)
fragment3 = filtfilt(b, a, fragment3)
fragment3 = filtfilt(bh, ah, fragment3)
print(get_number_of_peaks(fragment1), get_number_of_peaks(fragment2), get_number_of_peaks(fragment3))

p2.plot(fragment1, pen='r')
p2.plot(fragment2 - 1, pen='r')
p2.plot(fragment3 - 2, pen='r')

p.plot(ch1[start:stop], pen='r')
p.plot(ch2[start:stop] - 1, pen='r')
p.plot(ch3[start:stop] - 2, pen='r')
t = stop - start
l1 = t - stop + begin
l2 = l1 + end - begin
p.addLine(l1, pen='c')
p.addLine(l2, pen='c')

sys.exit(app.exec())

def filt12(ch):
    fch12 = np.zeros(ch.size)
    spec12 = np.zeros(ch.size)
    for i in np.arange(ch.size - 10):
        fch12[i] = 0.4 * ch[i - 5] + 0.4 * ch[i + 5]
        spec12[i] = np.abs(fch12[i] - ch[i])
    spec12 = savgol_filter(spec12, 21, 0)
    return fch12, spec12
    
def filt50(ch):
    fch50 = iirpeak(50, 25, 250)
    return fch50


def clean_ch(ch):
    b50, a50 = iirpeak(50, 10, 250)
    b12, a12 = iirpeak(12, 2.4, 250)
    bl, al = butter(2, 20, 'low', fs=250)
    out = ch.copy()
    fch = filtfilt(bl, al, ch)
    # spec50 = np.abs(lfilter(b50, a50, ch))
    w = 6
    spec50 = get_p2p(lfilter(b50, a50, ch), w, 1)
    spec50 = savgol_filter(spec50, 5, 0)*1.4
    # spec12 = np.abs(lfilter(b12, a12, ch))
    spec12 = get_p2p(lfilter(b12, a12, ch), w, 1)
    spec12 = savgol_filter(spec12, 7, 0)
    out[spec50 > spec12] = fch[spec50 > spec12]
    return out
