from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from functions import k


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
print(k)

try:
    r_pos = np.load(fdir + "/r_pos.npy")
    intervals = np.load(fdir + "/intervals.npy")

    start = r_pos[k] - int(intervals[k] * 0.4)
    stop = r_pos[k] - 10
    print(intervals[k])
    print(int(intervals[k] * 0.4))
    fragment1 = lead1[start:stop]
    fragment2 = lead2[start:stop]
    fragment3 = lead3[start:stop]
    fragment1 = fragment1 - fragment1.mean()
    fragment2 = fragment2 - fragment2.mean()
    fragment3 = fragment3 - fragment3.mean()

    vline = pg.InfiniteLine(angle=90, movable=False)
    vline.setPen(color="r", width=2)
    vline.setZValue(10)
    vline.setPos(int(intervals[k] - intervals[k] * 0.4))
    vline1 = pg.InfiniteLine(angle=90, movable=False)
    vline1.setPen(color="r", width=2)
    vline1.setZValue(10)
    vline1.setPos(int(intervals[k] - 10))

    b, a = butter(2, 18, btype='lowpass', fs=250)
    bl, al = butter(2, 3.5, btype='lowpass', fs=250) # butter(2, 2, btype='lowpass', fs=250)
    bl1, al1 = butter(2, 5.0, btype='lowpass', fs=250)
    # isoline1 = filtfilt(bl, al, fragment1)
    # isoline2 = filtfilt(bl, al, fragment2)
    # isoline3 = filtfilt(bl, al, fragment3)
    # p.plot(isoline1, pen='w')
    # p.plot(isoline2, pen='w')
    # p.plot(isoline3, pen='w')
    p.plot(fragment1, pen='g')
    p.plot(fragment2-0.5, pen='y')
    p.plot(fragment3-1.0, pen='c')

    ffragment1 = filtfilt(b, a, fragment1)
    ffragment2 = filtfilt(b, a, fragment2)
    ffragment3 = filtfilt(b, a, fragment3)
    isoline1 = filtfilt(bl, al, ffragment1)
    isoline2 = filtfilt(bl, al, ffragment2)
    isoline3 = filtfilt(bl, al, ffragment3)
    ffragment1 = ffragment1 - isoline1
    ffragment2 = ffragment2 - isoline2
    ffragment3 = ffragment3 - isoline3
    p.plot(ffragment1, pen='r')
    p.plot(ffragment2-0.5, pen='r')
    p.plot(ffragment3-1.0, pen='r')
    # p.plot(isoline1, pen='w')
    # p.plot(isoline2-0.5, pen='w')
    # p.plot(isoline3-1.0, pen='w')
    rr1 = lead1[r_pos[k - 1]+20:r_pos[k]-10]
    rr2 = lead2[r_pos[k - 1]+20:r_pos[k]-10]
    rr3 = lead3[r_pos[k - 1]+20:r_pos[k]-10]
    # rr1 = lead1[r_pos[k - 1]:r_pos[k]]
    # rr2 = lead2[r_pos[k - 1]:r_pos[k]]
    # rr3 = lead3[r_pos[k - 1]:r_pos[k]]
    p1.plot(rr1, pen='g')
    p1.plot(rr2 - 1, pen='y')
    p1.plot(rr3 - 2, pen='c')
    rr1 = filtfilt(bl1, al1, rr1)
    rr2 = filtfilt(bl1, al1, rr2)
    rr3 = filtfilt(bl1, al1, rr3)
    p1.plot(rr1, pen='g')
    p1.plot(rr2 - 1, pen='y')
    p1.plot(rr3 - 2, pen='c')
    # p1.addItem(vline)
    # p1.addItem(vline1)
except:
    pass

sys.exit(app.exec())
