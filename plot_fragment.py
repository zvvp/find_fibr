from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter, iirpeak, lfilter, filtfilt, butter, medfilt
from functions import parse_B_txt


app = QApplication(sys.argv)

b, a = butter(2, 8.0, 'lp', fs=250)
bh, ah = butter(1, 4, 'hp', fs=250)

p = pg.plot()
p.showGrid(x=True, y=True)
p1 = pg.plot()
p1.showGrid(x=True, y=True)

ch1 = np.load("d:/Kp_01/clean_lead1.npy")
ch2 = np.load("d:/Kp_01/clean_lead2.npy")
ch3 = np.load("d:/Kp_01/clean_lead3.npy")

r_pos, intervals, chars, forms = parse_B_txt()
k = 3
start = int(r_pos[k] - (intervals[k] * 0.25 + 10))
stop = int(r_pos[k] - intervals[k] * 0.06)

p1.plot(ch1[r_pos[k]-300:r_pos[k]+300])
p1.plot(ch2[r_pos[k]-300:r_pos[k]+300]-2)
p1.plot(ch3[r_pos[k]-300:r_pos[k]+300]-4)

fragment1 = ch1[start:stop]
fragment2 = ch2[start:stop]
fragment3 = ch3[start:stop]

p.plot(fragment1, pen='g')
p.plot(fragment2-0.5, pen='g')
p.plot(fragment3-1, pen='g')

fragment1 = filtfilt(b, a, fragment1)
fragment2 = filtfilt(b, a, fragment2)
fragment3 = filtfilt(b, a, fragment3)
# fragment1 = filtfilt(bh, ah, fragment1)
# fragment2 = filtfilt(bh, ah, fragment2)
# fragment3 = filtfilt(bh, ah, fragment3)

p.plot(fragment1, pen='r')
p.plot(fragment2-0.5, pen='r')
p.plot(fragment3-1, pen='r')


sys.exit(app.exec())
