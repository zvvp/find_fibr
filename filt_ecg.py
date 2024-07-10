from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter, iirpeak, lfilter, filtfilt, butter


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

f_name = QFileDialog.getOpenFileName()[0]
print(f_name)
ecg = np.fromfile(f_name, dtype=np.int16, offset=1024, count=900_000)

ch1 = (-ecg[::3] + 1024) / 100
ch2 = (-ecg[1::3] + 1024) / 100
ch3 = (-ecg[2::3] + 1024) / 100
ch1 = ch1 - np.mean(ch1)
ch2 = ch2 - np.mean(ch2)
ch3 = ch3 - np.mean(ch3)
ch = ch3.copy()

n = 4000
x = np.arange(n)
nous50 = np.ones(n)*np.sin(2*np.pi*50*x/250)
ch[:n] += nous50 * 0.15

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

# def clean_ch(ch):
#     fch12 = np.zeros(ch.size)
#     spec12 = np.zeros(ch.size)
#     fch50 = np.zeros(ch.size)
#     spec50 = np.zeros(ch.size)
#     for i in np.arange(ch.size - 10):
#         fch50[i] = 0.1382 * ch[i - 2] + 0.3618 * ch[i - 1] + 0.3618 * ch[i + 1] + 0.1382 * ch[i + 2]
#         spec50[i] = np.abs(fch50[i] - ch[i])
#         fch12[i] = 0.1382 * ch[i - 3] + 0.3618 * ch[i - 2] + 0.3618 * ch[i + 2] + 0.1382 * ch[i + 3]
#         # fch12[i] = 0.5 * ch[i - 2] + 0.5 * ch[i + 2]
#         spec12[i] = np.abs(fch12[i] - ch[i])
#     spec50 = savgol_filter(spec50, 31, 0)
#     spec12 = savgol_filter(spec12, 31, 0)
#     spec50 = spec50#**2 * 10 #+ 0.15
#     spec12 = spec12#**2 * 10
#     ch[spec50 > spec12] = fch50[spec50 > spec12]
#     return spec12, spec50

def clean_ch(ch):
    b50, a50 = iirpeak(50, 10, 250)
    b12, a12 = iirpeak(12, 2.4, 250)
    bl, al = butter(2, 20, 'low', fs=250)
    out = ch.copy()
    fch = filtfilt(bl, al, ch)
    spec50 = np.abs(lfilter(b50, a50, ch))
    spec50 = savgol_filter(spec50, 5, 0)*1.4
    spec12 = np.abs(lfilter(b12, a12, ch))
    spec12 = savgol_filter(spec12, 7, 0)
    out[spec50 > spec12] = fch[spec50 > spec12]
    return out

fch = clean_ch(ch)

p.plot(ch, pen='m')
p.plot(fch, pen='y')
# p.plot(spec50, pen='b')
# p.plot(spec501, pen='w')
# p.plot(spec12, pen='y')
# p.plot(spec121, pen='r')

sys.exit(app.exec())