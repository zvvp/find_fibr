from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from functions import parse_B_txt, get_coef_fibr, get_r_pos, detect
from get_time import get_time_qrs
# from numba import njit, jit 


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
vline = pg.InfiniteLine(label='{value:0.0F}', movable=True, labelOpts={'position': 0.1, 'color': (250, 250, 200), 'fill': (0, 0, 0), 'movable': True})
vline.setPen(color='c', width=1)
vline.setValue(2000)
vline.setZValue(10)
p.addItem(vline)

r_pos, intervals, chars, forms = parse_B_txt()

n = 111
# win = 5
fintervals = medfilt(intervals, n)
# dfintervals = detect(fintervals, win)
# fintervals = savgol_filter(fintervals, n, 0)
# fintervals = medfilt(fintervals, n1)
# fintervals = savgol_filter(fintervals, n, 0)


# intervals = intervals - np.roll(intervals, -1)
# p.plot(intervals, pen="c")

coef_fibr, intervals1 = get_coef_fibr(intervals, chars)
fcoef_fibr = medfilt(coef_fibr, n)
# fcoef_fibr = medfilt(fcoef_fibr, n)
# fcoef_fibr = medfilt(fcoef_fibr, n)

# b, a = butter(1, 0.001, 'hp', fs=1)
# fcoef_fibr = np.abs(filtfilt(b, a, fcoef_fibr))

# p.plot(coef_fibr, pen="y")
# p.plot(intervals, pen="y")
# p.plot(intervals1, pen="b")
p.plot(fintervals, pen="m")
p.plot(fcoef_fibr, pen="r")
# p.plot(medfilt(intervals*coef_fibr, 111), pen="r")
# p.setYRange(max=500, min=0)

# r_pos = get_r_pos()
# addr = r_pos[114039]
# fname = QFileDialog.getOpenFileName()[0]
# get_time_qrs(addr, fname)

sys.exit(app.exec())