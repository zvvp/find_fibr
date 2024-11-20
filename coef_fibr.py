from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from functions import parse_B_txt, get_coef_fibr, del_V_S
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
# p.plot(intervals, pen="g")
# intervals = del_V_S(intervals, chars)
# p.plot(intervals, pen="r")
n = 111
intervals = del_V_S(intervals, chars)
fintervals = medfilt(intervals, n)
# intervals = del_V_S(intervals, chars)
coef_fibr = get_coef_fibr(intervals, chars)
# intervals = del_V_S(intervals, chars)
# p.plot(intervals, pen="r")
# coef_fibr[coef_fibr > 3000] = 3000
fcoef_fibr = medfilt(coef_fibr, n)

# p.plot(intervals, pen="m")
# intervals = del_V_S(intervals, chars)
# p.plot(intervals, pen="r")
# intervals = medfilt(intervals, n)
p.plot(fintervals, pen="c")
p.plot(fcoef_fibr, pen="y")

addr = r_pos[114038]
fname = QFileDialog.getOpenFileName()[0]
get_time_qrs(addr, fname)

sys.exit(app.exec())