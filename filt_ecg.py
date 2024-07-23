from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter, iirpeak, lfilter, filtfilt, butter, medfilt
from functions import get_number_of_peaks


app = QApplication(sys.argv)

# p = pg.plot()
# p.showGrid(x=True, y=True)
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
def get_fragment(start, stop, lead):
    t = stop - start
    print(t)
    shift = 32
    k1 = 0.8
    k2 = 0.615   # 0.63
    start = start + int((t * k1 + shift) * k2)
    # start = start + int(0.35 * t) + 36
    stop = stop - int(0.065 * t)
    fragment = lead[start:stop]
    # fragment = lead[start + int(0.15 * t):stop - int(0.04 * t)]
    return fragment
start = 6216943
stop = 6217095
p1.plot(ch1[start-400:stop+400])
p1.plot(ch2[start-400:stop+400]-2)
p1.plot(ch3[start-400:stop+400]-4)
fragment1 = get_fragment(start, stop, ch1)
fragment2 = get_fragment(start, stop, ch2)
fragment3 = get_fragment(start, stop, ch3)

# p.plot(fragment1, pen='c')
# p.plot(fragment2 - 1, pen='c')
# p.plot(fragment3 - 2, pen='c')
b, a = butter(2, 11, 'low', fs=250) #  2, 8, 'low', fs=250 3, 10
bh, ah = butter(1, 0.08, 'high', fs=250) # 1, 0.07, 'high', fs=250 1, 0.08
fragment1 = filtfilt(b, a, fragment1)
fragment1 = filtfilt(bh, ah, fragment1)
fragment2 = filtfilt(b, a, fragment2)
fragment2 = filtfilt(bh, ah, fragment2)
fragment3 = filtfilt(b, a, fragment3)
fragment3 = filtfilt(bh, ah, fragment3)
print(get_number_of_peaks(fragment1))
print(get_number_of_peaks(fragment2))
print(get_number_of_peaks(fragment3))
# p.plot(fragment1, pen='r')
# p.plot(fragment2 - 1, pen='r')
# p.plot(fragment3 - 2, pen='r')

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
