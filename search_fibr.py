from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import filtfilt, medfilt, butter, savgol_filter
from functions import get_S, parse_B_txt, get_number_of_peaks, get_coef_fibr, del_V_S, moving_average
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
    bl = [0.09635722, 0.19271443, 0.09635722]
    al = [1.0, -0.95071164, 0.33614051]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905,  0.67643727]
    bh = [0.9989957, - 0.9989957]
    ah = [1.0, - 0.9979914]
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
                out[i] = 0.0  #  out[i] = 0.3
            elif sum_n == 2:
                out[i] = 0.0  # 0.7
            elif sum_n == 1:
                out[i] = 100.0  # 1.7
            elif sum_n == 0:
                out[i] = 100.0
                # out[i - 1:i + 2] = 1000.0  # 2.0
        else:
            out[i] = 0.0   # 0.3
    win = int(np.mean(intervals) * 1.0)
    out = moving_average(out, win)
    return out


p = pg.plot()
p.showGrid(x=True, y=True)
vline = pg.InfiniteLine(label='{value:0.0F}', movable=True,
                        labelOpts={'position': 0.1, 'color': (250, 250, 200), 'fill': (0, 0, 0), 'movable': True})
vline.setPen(color='c', width=1)
vline.setValue(2000)
vline.setZValue(10)
p.addItem(vline)

# p1 = pg.plot()
# p1.showGrid(x=True, y=True)

@time_fun
def main():
    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")

    get_S()
    r_pos, intervals, chars, forms = parse_B_txt()
    pzub = get_P(lead1, lead2, lead3, intervals, r_pos, chars) * 0.3
    fintervals = del_V_S(intervals, chars)
    coef_fibr = get_coef_fibr(fintervals)
    # fintervals = moving_average(fintervals, 15)
    # p_coef_fibr = coef_fibr + pzub
    # p_coef_fibr = moving_average(p_coef_fibr, 179) * 2.0
    # p_coef_fibr = moving_average(p_coef_fibr, 79) * 1.9
    n = 111
    fintervals = medfilt(fintervals, n)
    coef_fibr = medfilt(coef_fibr, n)

    # p.plot(fintervals / 100, pen="g")
    fintervals = moving_average(fintervals, n)
    p.plot((fintervals - np.mean(fintervals)) * 0.0065 + 0.98, pen="g")
    p.plot(pzub, pen='y')
    coef_fibr = moving_average(coef_fibr, n)
    p.plot(coef_fibr + pzub, pen='r')
    # p.plot(coef_fibr, pen='r')
    # bl, al = butter(2, 31, 'low', fs=250)
    # print(f"bl = {bl}")
    # print(f"al = {al}")
    # bh, ah = butter(1, 0.08, 'high', fs=250)
    # print(f"bh = {bh}")
    # print(f"ah = {ah}")

if __name__ == "__main__":
    main()

sys.exit(app.exec())
