from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import filtfilt, medfilt, butter, savgol_filter
from functions import get_S, parse_B_txt, get_number_of_peaks, get_coef_fibr, del_V_S, moving_average, correct_fibr
from time import time


app = QApplication(sys.argv)

def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n

    return wrapper


@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars):
    bl = [0.01591456, 0.03182911, 0.01591456]
    al = [1.0, -1.61277905,  0.67643727]
    bh = [0.9989957, - 0.9989957]
    ah = [1.0, - 0.9979914]
    mean_interval = np.mean(intervals) * 2.5
    out = np.zeros(len(r_pos))
    for i in range(1, len(r_pos) - 1):
        if chars[i] == 'N':
            len_pr = int(round(intervals[i] * 0.36))
            start = r_pos[i] - len_pr
            stop = r_pos[i] - 11
            fragment_l1 = lead1[start:stop]
            fragment_l1 = filtfilt(bl, al, fragment_l1)
            fragment_l1 = filtfilt(bh, ah, fragment_l1)
            fragment_l2 = lead2[start:stop]
            fragment_l2 = filtfilt(bl, al, fragment_l2)
            fragment_l2 = filtfilt(bh, ah, fragment_l2)
            fragment_l3 = lead3[start:stop]
            fragment_l3 = filtfilt(bl, al, fragment_l3)
            fragment_l3 = filtfilt(bh, ah, fragment_l3)
            n1 = get_number_of_peaks(fragment_l1)
            n2 = get_number_of_peaks(fragment_l2)
            n3 = get_number_of_peaks(fragment_l3)
            sum_n = n1 + n2 + n3

            if sum_n == 3:
                out[i] = 0.0
            elif sum_n == 2:
                out[i] = 50.0
            elif sum_n == 1:
                out[i] = mean_interval
            elif sum_n == 0:
                out[i] = mean_interval
        else:
            out[i] = 50.0
    out = moving_average(out, 111)
    return out


p = pg.plot()
p.showGrid(x=True, y=True)
# vline = pg.InfiniteLine(label='{value:0.0F}', movable=True,
#                         labelOpts={'position': 0.1, 'color': (250, 250, 200), 'fill': (0, 0, 0), 'movable': True})
# vline.setPen(color='c', width=1)
# vline.setValue(2000)
# vline.setZValue(10)
# p.addItem(vline)

# p1 = pg.plot()
# p1.showGrid(x=True, y=True)

@time_fun
def main():
    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")

    get_S()
    r_pos, intervals, chars, forms = parse_B_txt()
    pzub = get_P(lead1, lead2, lead3, intervals, r_pos, chars)
    mean_pzub = np.mean(pzub)
    pzub = (pzub - mean_pzub) * 0.8 + mean_pzub
    fintervals = del_V_S(intervals, chars)
    coef_fibr = get_coef_fibr(fintervals)
    n = 51
    fintervals = medfilt(fintervals, n)
    fintervals = moving_average(fintervals, n)
    # mean_intervals = np.mean(fintervals)
    # fintervals = (fintervals - mean_intervals) * 0.8 + mean_intervals
    p.plot(fintervals, pen="g")
    coef_fibr = medfilt(coef_fibr, n)
    coef_fibr = moving_average(coef_fibr, n)
    mean_fibr = np.mean(coef_fibr)
    # p.plot(pzub, pen='y')
    coef_fibr = (coef_fibr - mean_fibr) * 0.8 + mean_fibr
    p_coef_fibr = coef_fibr * pzub / 200
    mean_p_fibr = np.mean(p_coef_fibr)
    p_coef_fibr = (p_coef_fibr - mean_p_fibr) * 0.9 + mean_p_fibr
    p.plot(p_coef_fibr, pen='c')

    # bl, al = butter(2, 31, 'low', fs=250)
    # print(f"bl = {bl}")
    # print(f"al = {al}")
    # bh, ah = butter(1, 0.08, 'high', fs=250)
    # print(f"bh = {bh}")
    # print(f"ah = {ah}")

if __name__ == "__main__":
    main()

sys.exit(app.exec())
