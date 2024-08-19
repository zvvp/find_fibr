from PyQt6.QtWidgets import QApplication
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from functions import get_intervals, get_r_pos


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
ch1 = np.load("d:/Kp_01/clean_lead1.npy")
ch2 = np.load("d:/Kp_01/clean_lead2.npy")
ch3 = np.load("d:/Kp_01/clean_lead3.npy")

r_pos = get_r_pos()
intervals = get_intervals()
diff_intervals = np.diff(intervals)
# diff_intervals = diff_intervals + np.roll(diff_intervals, 1)
diff_intervals = np.append(diff_intervals, 0)
# print(r_pos.size, intervals.size, diff_intervals.size)
start = 6626047 - 10_000
stop = 6626182 + 10_000
start_r = np.argwhere(r_pos >= start)[0][0]
stop_r = np.argwhere(r_pos >= stop)[0][0]
# print(start_r, stop_r)
x = np.arange(start, stop)
arr_diff = np.zeros(ch2.size)
arr_diff[r_pos[start_r:stop_r]] = diff_intervals[start_r:stop_r]
# arr_diff = np.abs(arr_diff)
# arr_diff[arr_diff < 0] = 0
# b, a = butter(2, 0.1, 'low', fs=250)
# arr_diff = filtfilt(b, a, arr_diff**2)
p.plot(x, ch2[start:stop]*10, pen='c')
p.plot(x, arr_diff[start:stop]-100, pen=pg.mkPen(width=1, color='m'))

sys.exit(app.exec())