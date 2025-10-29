from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from functions import k
from scipy.signal import butter, filtfilt

# k = 30000
b, a = butter(2, 18.0, 'lp', fs=250) # 2, 10.0, 'lp', fs=250
bi, ai = butter(1, 3.0, 'lp', fs=250)
bh, ah = butter(10, 1.0, 'hp', fs=250)
app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
p1 = pg.plot()
p1.showGrid(x=True, y=True)

fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar/fibr")
print(fdir)
lead1 = np.load(fdir + "/clean_lead1.npy")
lead2 = np.load(fdir + "/clean_lead2.npy")
lead3 = np.load(fdir + "/clean_lead3.npy")
presence_PR = np.load("D:/PycharmProjects/experiment/presence_PR.npy")
presence_PR = presence_PR.astype(int)
try:
    r_pos = np.load(fdir + "/r_pos.npy")
    intervals = np.load(fdir + "/intervals.npy")
    # start = r_pos[k] - presence_PR[k] - 17
    # stop = r_pos[k] - 10#int(intervals[k] ** 0.5 * 0.66 + 3.0)
    len_pr = int(intervals[k] * 0.4) # intervals[k] * 0.36 + 5
    start = r_pos[k] - len_pr
    stop = r_pos[k] - 10
    fragment1 = lead1[start:stop]
    fragment2 = lead2[start:stop]
    fragment3 = lead3[start:stop]
    p.plot(fragment1, pen='g')
    p.plot(fragment2-0.5, pen='g')
    p.plot(fragment3-1.0, pen='g')
    ffragment1 = filtfilt(b, a, fragment1)
    ffragment2 = filtfilt(b, a, fragment2)
    ffragment3 = filtfilt(b, a, fragment3)
    # fragment1 = filtfilt(bh, ah, fragment1)
    # fragment2 = filtfilt(bh, ah, fragment2)
    # fragment3 = filtfilt(bh, ah, fragment3)
    # fragment1 = fragment1 - np.mean(fragment1)
    # fragment1[fragment1 < 0] = 0
    # fragment2 = fragment2 - np.mean(fragment2)
    # fragment2[fragment2 < 0] = 0
    # fragment3 = fragment3 - np.mean(fragment3)
    # fragment3[fragment3 < 0] = 0
    # fragment1 = filtfilt(b, a, fragment1)
    # fragment2 = filtfilt(b, a, fragment2)
    # fragment3 = filtfilt(b, a, fragment3)
    p.plot(ffragment1, pen='r')
    p.plot(ffragment2-0.5, pen='r')
    p.plot(ffragment3-1.0, pen='r')
    isoline1 = filtfilt(bi, ai, fragment1)
    isoline2 = filtfilt(bi, ai, fragment2)
    isoline3 = filtfilt(bi, ai, fragment3)
    p.plot(isoline1, pen='y')
    p.plot(isoline2-0.5, pen='y')
    p.plot(isoline3-1.0, pen='y')
    p1.plot(lead1[r_pos[k]-1000:r_pos[k]+1000], pen='g')
    p1.plot(lead2[r_pos[k]-1000:r_pos[k]+1000]-2.0, pen='g')
    p1.plot(lead3[r_pos[k]-1000:r_pos[k]+1000]-4.0, pen='g')
except:
    pass

sys.exit(app.exec())
