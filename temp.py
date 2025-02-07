from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt, argrelmax
from functions import parse_B1_txt, get_S, moving_average, del_V_S, truncate_win
from time import time

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
p01 = pg.plot()
p01.showGrid(x=True, y=True)
p02 = pg.plot()
p02.showGrid(x=True, y=True)
p03 = pg.plot()
p03.showGrid(x=True, y=True)

bl, al = butter(2, 13.0, 'lp', fs=250)  # 2, 8.0, 'lp', fs=250

def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n

    return wrapper


@time_fun
def get_inds_min_diff(intervals):
    diff_intervals = np.abs(intervals - np.roll(intervals, -1))
    return np.where(diff_intervals <= 2)[0]


@time_fun
def get_p_pos(lead1, lead2, lead3, intervals, r_pos, chars, inds_min):
    global bl, al
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    # pos_p1 = 0
    # pos_p2 = 0
    # pos_p3 = 0
    # intervals_PR1 = np.array([])
    # intervals_PR2 = np.array([])
    # intervals_PR3 = np.array([])
    intervals_PR1 = np.zeros(r_pos.size)
    intervals_PR2 = np.zeros(r_pos.size)
    intervals_PR3 = np.zeros(r_pos.size)
    pr1 = 0
    pr2 = 0
    pr3 = 0
    for i in inds_min:
        if (chars[i] == 'N') and (chars[i - 1] == 'N'):# and (chars[i + 1] == 'N'):
            len_pr = int(intervals[i] * 0.2 + 20)
            start = r_pos[i] - len_pr
            stop = r_pos[i] - int(len_pr * 0.05 + 10)
            fragment1 = lead1[start:stop]
            fragment2 = lead2[start:stop]
            fragment3 = lead3[start:stop]

            fragment1 = filtfilt(bl, al, fragment1)
            fragment2 = filtfilt(bl, al, fragment2)
            fragment3 = filtfilt(bl, al, fragment3)
            # if i == 1:
            #     p01.plot(fragment1, pen='g')
            #     p01.plot(fragment2, pen='y')
            #     p01.plot(fragment3, pen='c')

            pos_loc_max_frag1 = argrelmax(fragment1)[0]
            pos_loc_max_frag2 = argrelmax(fragment2)[0]
            pos_loc_max_frag3 = argrelmax(fragment3)[0]
            if (pos_loc_max_frag1.size == 1) and ((fragment1[pos_loc_max_frag1] - np.min(fragment1)) > 0.1):
                intervals_PR1[i] = len_pr - pos_loc_max_frag1[0]
                # intervals_PR1 = np.append(intervals_PR1, len_pr - pos_loc_max_frag1[0])
            if (pos_loc_max_frag2.size == 1) and ((fragment2[pos_loc_max_frag2] - np.min(fragment2)) > 0.1):
                intervals_PR2[i] = len_pr - pos_loc_max_frag2[0]
                # intervals_PR2 = np.append(intervals_PR2, len_pr - pos_loc_max_frag2[0])
            if (pos_loc_max_frag3.size == 1) and ((fragment3[pos_loc_max_frag3] - np.min(fragment3)) > 0.1):
                intervals_PR3[i] = len_pr - pos_loc_max_frag3[0]
                # intervals_PR3 = np.append(intervals_PR3, len_pr - pos_loc_max_frag3[0])

    if intervals_PR1.size > 0:
        pr1 = int(np.mean(intervals_PR1[intervals_PR1 > 0]))
    if intervals_PR2.size > 0:
        pr2 = int(np.mean(intervals_PR2[intervals_PR2 > 0]))
    if intervals_PR3.size > 0:
        pr3 = int(np.mean(intervals_PR3[intervals_PR3 > 0]))
    p03.plot(intervals_PR1, pen='g')
    p03.plot(intervals_PR2, pen='y')
    p03.plot(intervals_PR3, pen='c')

    return pr1, pr2, pr3


@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars, pr1, pr2, pr3):
    global bl, al
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    p1 = np.zeros(len(r_pos))
    p2 = np.zeros(len(r_pos))
    p3 = np.zeros(len(r_pos))
    out = np.zeros(len(r_pos))
    mean_interval = np.mean(intervals)
    for i in range(1, len(r_pos)):
        if chars[i] == 'N':
            fragment_l1 = lead1[r_pos[i] - pr1 - 10:r_pos[i] - pr1 + 11]    # -14 +15
            fragment_l1 = filtfilt(bl, al, fragment_l1)
            ind_max1 = argrelmax(fragment_l1)[0]
            if (ind_max1.size == 1) and ((fragment_l1[ind_max1] - np.min(fragment_l1)) > 0.02):
                p1[i] = 1
            fragment_l2 = lead2[r_pos[i] - pr2 - 10:r_pos[i] - pr2 + 11]
            fragment_l2 = filtfilt(bl, al, fragment_l2)
            ind_max2 = argrelmax(fragment_l2)[0]
            if (ind_max2.size == 1) and ((fragment_l2[ind_max2] - np.min(fragment_l2)) > 0.02):
                p2[i] = 1
            fragment_l3 = lead3[r_pos[i] - pr3 - 10:r_pos[i] - pr3 + 11]
            fragment_l3 = filtfilt(bl, al, fragment_l3)
            ind_max3 = argrelmax(fragment_l3)[0]
            if (ind_max3.size == 1) and ((fragment_l3[ind_max3] - np.min(fragment_l3)) > 0.02):
                p3[i] = 1

            if i == 36400:
                p01.plot(fragment_l1, pen='g')
                p01.plot(fragment_l2, pen='y')
                p01.plot(fragment_l3, pen='c')
                p02.plot(lead1[r_pos[i]-300:r_pos[i]+300], pen='g')
                p02.plot(lead2[r_pos[i]-300:r_pos[i]+300]-2, pen='y')
                p02.plot(lead3[r_pos[i]-300:r_pos[i]+300]-4, pen='c')

            sum_p = p1[i] + p2[i] + p3[i]
            if sum_p == 3:
                out[i] = 0.0
            elif sum_p == 2:
                # out[i] = 1.0
                out[i] = mean_interval * 0.05  # 0.2
            elif sum_p == 1:
                out[i] = mean_interval * 0.95  # 0.5
            elif sum_p == 0:
                out[i] = mean_interval
        else:
            out[i] = out[i - 1]
    out = medfilt(out, 3)
    out = moving_average(out, 31)
    out = truncate_win(out, 0.3, 50)
    return out


def main():
    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")
    get_S()
    r_pos, intervals, chars, forms = parse_B1_txt()
    # p.plot(intervals, pen="g")
    fintervals = del_V_S(intervals, chars)
    m_fintervals = medfilt(fintervals, 21)  # 5
    cleen_fintervals = m_fintervals + (fintervals - m_fintervals) * 0.2
    p.plot(cleen_fintervals, pen="g")

    inds_min = get_inds_min_diff(intervals)
    pos_p1, pos_p2, pos_p3 = get_p_pos(lead1, lead2, lead3, intervals, r_pos, chars, inds_min)
    print(pos_p1, pos_p2, pos_p3)

    coef_p = get_P(lead1, lead2, lead3, intervals, r_pos, chars, pos_p1, pos_p2, pos_p3)

    p.plot(coef_p, pen='r')



if __name__ == "__main__":
    main()

sys.exit(app.exec())
