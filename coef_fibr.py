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
        for i, line in enumerate(lines, start=2):
            if i < len_lines - 2:
                period_2 = int(lines[i-2].split(';')[1])
                period_1 = int(lines[i-1].split(';')[1])
                period = int(lines[i].split(';')[1])
                period1 = int(lines[i+1].split(';')[1])
                period2 = int(lines[i+2].split(';')[1])
                chars = np.array([lines[i-2].split(';')[2][0], lines[i-1].split(';')[2][0], lines[i].split(';')[2][0], lines[i+1].split(';')[2][0], lines[i+2].split(';')[2][0]])
                tf = np.array([period_2, period_1, period, period1, period2])
                # tf0 = np.array([period_2, period_1, period, period1, period2])
                # max_tf = np.max(tf)
                # min_tf = np.min(tf)
                # median_tf = np.median(tf)
                mean_tf = np.median(tf)
                # mean_tf3 = (np.sum(tf) - np.max(tf) - np.min(tf)) / 3
                # ind_max = np.argmax(tf)
                # ind_min = np.argmin(tf)
                # tf[ind_max] = np.min(tf)
                # tf[ind_min] = np.min(tf)
                diff_tf_mean = np.abs(tf - mean_tf)
                diff_tf_mean = np.sort(diff_tf_mean)
                diff_tf_mean = diff_tf_mean[:3]
                sum_diff_tf_mean = np.sum(diff_tf_mean)
                out[i] = sum_diff_tf_mean / (mean_tf * 0.0014)**2
                if ('V' in chars) or ('S' in chars):
                    out[i] = out[i] * 0.5
                # diff_tf_median = np.abs(tf - median_tf)
                diff_t = np.abs(np.array([tf[1] - tf[0], tf[2] - tf[1],
                                    tf[3] - tf[2], tf[4] - tf[3]]))
                pass
        #         if 'F' in chars:
        #             out[i] = np.sum(diff_t) / np.min(tf)
        #         elif ('V' in chars) or ('S' in chars):
        #             out[i] = np.mean(diff_t) / np.max(tf) * 300
        #             # out[i] = np.mean(diff_t)**2 / sort_tf[2]**2 * 1_000
        #             # out[i] = np.mean(diff_t)**2 / np.max(tf)**2 * 20_000
        #         elif not 'A' in chars:
        #             out[i] = (np.sum(diff_t) - np.mean(diff_t)) / (np.sum(tf) - np.max(tf)) * 2_000
        #             # out[i] = np.mean(diff_t)**2 / np.max(tf)**2 * 20_000
        #         else:
        #             out[i] = median_tf * 0.7
        out[out > 2000] = 2000
        return out 

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
vline = pg.InfiniteLine(label='{value:0.0F}', movable=True, labelOpts={'position': 0.1, 'color': (250, 250, 200), 'fill': (0, 0, 0), 'movable': True})
vline.setPen(color='c', width=1)
vline.setValue(2000)
vline.setZValue(10)
p.addItem(vline)

intervals = get_intervals()
intervals[intervals > 500] = 500
fintervals = medfilt(intervals, 111)
# p.plot(intervals, pen="c")
p.plot(fintervals, pen="c")
coef_fibr = get_coef_fibr()
fcoef_fibr = medfilt(coef_fibr, 111)

# b, a = butter(1, 0.001, 'hp', fs=1)
# fcoef_fibr = np.abs(filtfilt(b, a, fcoef_fibr))

p.plot(fcoef_fibr, pen="r")
# p.plot(medfilt(intervals*coef_fibr, 111), pen="r")
# p.setYRange(max=500, min=0)

# r_pos = get_r_pos()
# addr = r_pos[79963]
# fname = QFileDialog.getOpenFileName()[0]
# get_time_qrs(addr, fname)

sys.exit(app.exec())