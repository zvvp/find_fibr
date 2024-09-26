from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from functions import get_intervals, get_r_pos
from get_time import get_time_qrs


def get_coef_fibr():
    with open("C:/EcgVar/B1.txt", "r") as f:    
        lines = f.readlines()
        lines = lines[12:]
        len_lines = len(lines)
        out = np.zeros(len_lines)
        for i, line in enumerate(lines):
            if (i > 1) and (i < len_lines - 2) and (not ';V' in lines[i-2]) and (not ';S' in lines[i-2])\
                and (not ';V' in lines[i-1]) and (not ';S' in lines[i-1])\
                and (not ';V' in line) and (not ';S' in line)\
                and (not ';V' in lines[i+1]) and (not ';S' in lines[i+1])\
                and (not ';V' in lines[i+2]) and (not ';S' in lines[i+2]):
                period_2 = int(lines[i-2].split(';')[1])
                period_1 = int(lines[i-1].split(';')[1])
                period = int(line.split(';')[1])
                period1 = int(lines[i+1].split(';')[1])
                period2 = int(lines[i+2].split(';')[1])
                tf = np.array([period_2, period_1, period, period1, period2])

                diff_t = np.array([np.abs(period_1 - period_2), np.abs(period - period_1),
                                   np.abs(period1 - period), np.abs(period2 - period1)])#, np.abs(period2 - period_2)])
                # sort_diff_t = np.sort(diff_t)
                mean_t = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
                # mean_t = np.sum(tf) - np.max(tf) - np.min(tf)
                # out[i] = np.sum(diff_t) / mean_t**2 * 75_000
                # out[i] = (np.sum(diff_t) - sort_diff_t[-1] - sort_diff_t[-2]) / mean_t**2 * 4_000_000
                # out[i] = (np.sum(diff_t) - sort_diff_t[-1]) / mean_t**2 * 130_000
                out[i] = (np.sum(diff_t) - np.max(diff_t)) / mean_t**2 * 350_000
                pass
                # out[i] = (np.sum(diff_t) - np.max(diff_t)) / mean_t**2 * 75_000
            elif (i > 1) and (i < len_lines - 2):
                out[i] = out[i-2]
                out[i-1] = out[i-2]
        return out 

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
# ch1 = np.load("d:/Kp_01/clean_lead1.npy")
# ch2 = np.load("d:/Kp_01/clean_lead2.npy")
# ch3 = np.load("d:/Kp_01/clean_lead3.npy")

intervals = get_intervals()
intervals[intervals > 500] = 500
fintervals = medfilt(intervals, 111)
# p.plot(intervals, pen="c")
p.plot(fintervals, pen="c")
# get_coef_fibr()
coef_fibr = get_coef_fibr()
fcoef_fibr = medfilt(coef_fibr, 111)
# b, a = butter(2, 0.01, 'hp', fs=1.5)
# fcoef_fibr = filtfilt(b, a, fcoef_fibr)
# fcoef_fibr = savgol_filter(fcoef_fibr, 100, 1)
# p.plot(coef_fibr, pen="g")
p.plot(fcoef_fibr, pen="r")
# p.plot(medfilt(intervals*coef_fibr, 111), pen="r")
# p.setYRange(max=500, min=0)

r_pos = get_r_pos()
addr = r_pos[53057]
fname = QFileDialog.getOpenFileName()[0]
get_time_qrs(addr, fname)

sys.exit(app.exec())