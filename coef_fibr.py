from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from functions import parse_B_txt, get_coef_fibr, get_r_pos, detect, sum3, mean3
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
fintervals = medfilt(intervals, 51)

coef_fibr = get_coef_fibr(intervals, chars)
coef_fibr[coef_fibr > 3000] = 3000
fcoef_fibr = medfilt(coef_fibr, n)

p.plot(fintervals, pen="m")
p.plot(fcoef_fibr, pen="r")

# addr = r_pos[66710]
# fname = QFileDialog.getOpenFileName()[0]
# get_time_qrs(addr, fname)

sys.exit(app.exec())