from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import filtfilt, medfilt, butter, savgol_filter, argrelmax
from functions import (get_S, parse_B1_txt, get_number_of_peaks1, get_coef_fibr, del_V_S,
                       moving_average, get_p2p, get_max_p, truncate_win2, truncate_ch, div_intervals)
from time import time


app = QApplication(sys.argv)

p01 = pg.plot()
p01.showGrid(x=True, y=True)

bl, al = butter(2, 8.0, 'lp', fs=250)

def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n
    return wrapper


# @time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars):
    global bl, al
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905,  0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    p1 = np.zeros(len(r_pos))
    p2 = np.zeros(len(r_pos))
    p3 = np.zeros(len(r_pos))
    mean_interval = np.mean(intervals) * 0.3 #1.71
    # print(mean_interval)
    out = np.zeros(len(r_pos))
    for i in range(1, len(r_pos) - 1):
        if chars[i] == 'N':
            len_pr = int(round(intervals[i] * 0.25 + 10))   # 0.36 0.45 0.4
            start = r_pos[i] - len_pr
            stop = r_pos[i] - int(round(len_pr * 0.06))   # 12  int(round(len_pr * 0.11 + 5))
            fragment_l1 = lead1[start:stop]
            fragment_l1 = filtfilt(bl, al, fragment_l1)
            # fragment_l1 = filtfilt(bh, ah, fragment_l1)
            num_max1 = argrelmax(fragment_l1)[0]
            p1[i] = get_max_p(fragment_l1)
            fragment_l2 = lead2[start:stop]
            fragment_l2 = filtfilt(bl, al, fragment_l2)
            # fragment_l2 = filtfilt(bh, ah, fragment_l2)
            num_max2 = argrelmax(fragment_l2)[0]
            p2[i] = get_max_p(fragment_l2)
            fragment_l3 = lead3[start:stop]
            fragment_l3 = filtfilt(bl, al, fragment_l3)
            # fragment_l3 = filtfilt(bh, ah, fragment_l3)
            num_max3 = argrelmax(fragment_l3)[0]
            p3[i] = get_max_p(fragment_l3)
            if i == 1200:
                p01.plot(fragment_l1)
                p01.plot(fragment_l2 - 0.5)
                p01.plot(fragment_l3 - 1.0)
            # n1 = get_number_of_peaks(fragment_l1, 1)
            # n2 = get_number_of_peaks(fragment_l2, 2)
            # n3 = get_number_of_peaks(fragment_l3, 3)
            n1 = get_number_of_peaks1(fragment_l1)
            n2 = get_number_of_peaks1(fragment_l2)
            n3 = get_number_of_peaks1(fragment_l3)
            sum_n = n1 + n2 + n3

            if sum_n == 3:
                out[i] = 0.0
            elif sum_n == 2:
                # out[i] = 1.0
                out[i] = mean_interval * 0.05   # 0.2
            elif sum_n == 1:
                out[i] = mean_interval * 0.95   # 0.5
            elif sum_n == 0:
                out[i] = mean_interval
        else:
            out[i] = out[i - 1]
    out = medfilt(out, 3)
    out = moving_average(out, 31)
    # out = truncate_win(out, 0.5, 40)
    # out = medfilt(out, 45)   # 27
    # out = moving_average(out, 11)
    return out#, p1, p2, p3


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

# @time_fun
def main():
    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")

    get_S()
    r_pos, intervals, chars, forms = parse_B1_txt()

    fintervals = del_V_S(intervals, chars)
    m_fintervals = medfilt(fintervals, 21) # 5
    cleen_fintervals = m_fintervals + (fintervals - m_fintervals) * 0.2

    coef_fibr = get_coef_fibr(fintervals)
    # p2p_fibr = get_p2p(coef_fibr, 11)
    # p2p_fibr = truncate_win(p2p_fibr, 0.5, 20)
    # p2p_fibr = get_p2p(p2p_fibr, 7)
    # isoline_fibr = moving_average(coef_fibr, 61)
    # line_fibr = np.abs(coef_fibr - isoline_fibr)
    # f_line_fibr = moving_average(line_fibr, 51)
    p.plot(coef_fibr, pen='c')
    # p.plot(p2p_fibr, pen='r')
    # p.plot(line_fibr, pen='y')
    # p.plot(f_line_fibr, pen='w')

    pzub = get_P(lead1, lead2, lead3, intervals, r_pos, chars)
    # p1 = medfilt(p1, 17)
    # p2 = medfilt(p3, 17)
    # p3 = medfilt(p3, 17)
    # p1 = moving_average(p1, 30)
    # p2 = moving_average(p3, 40)
    # p3 = moving_average(p3, 50)
    p.plot(pzub, pen='y')
    # p.plot(p1, pen='w')
    # p.plot(p2 - 0.5, pen='c')
    # p.plot(p3 - 1, pen='r')

    p.plot(cleen_fintervals, pen="g")

    # p_coef_fibr = pzub * coef_fibr

    # p.plot(p_coef_fibr, pen='r')


if __name__ == "__main__":
    main()

sys.exit(app.exec())
