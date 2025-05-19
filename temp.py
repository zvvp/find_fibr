from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, argrelmax
from functions import parse_B1_txt, get_S1, moving_average, del_V_S, truncate_win2, interp_pr, truncate_ch, \
    get_scatter_coef, magnific_ch, get_attr
from time import time
import logging

logging.basicConfig(filename='experiment.log', level=logging.INFO, filemode="w",
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
p.setTitle('p')
# p01 = pg.plot()
# p01.showGrid(x=True, y=True)
# p01.setTitle('p01')
p02 = pg.plot()
p02.showGrid(x=True, y=True)
p02.setTitle('p02')
# p03 = pg.plot()
# p03.showGrid(x=True, y=True)
# p03.setTitle('p03')
# p04 = pg.plot()
# p04.showGrid(x=True, y=True)
# p04.setTitle('p04')

bl, al = butter(2, 13.0, 'lp', fs=250)  # 2, 8.0, 'lp', fs=250


def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n

    return wrapper


# @time_fun
def get_inds_min_diff(intervals):
    """
    Возвращает индексы интервалов, где абсолютная разница между
    последовательными интервалами меньше или равна 3.

    Параметры:
    intervals (numpy.ndarray): 1D numpy массив интервалов.

    Возвращает:
    numpy.ndarray: 1D numpy массив индексов, где абсолютная разница
    между последовательными интервалами меньше или равна 3.
    """
    diff_intervals = np.abs(intervals - np.roll(intervals, -1))
    return np.where(diff_intervals <= 3)[0]


def plot_select_p(lead1, lead2, lead3, intervals, r_pos, ind):
    """
    Строит графики выбранных фрагментов трех каналов и их отфильтрованные версии.

    Параметры:
    lead1 (numpy.ndarray): 1D numpy массив данных lead1.
    lead2 (numpy.ndarray): 1D numpy массив данных lead2.
    lead3 (numpy.ndarray): 1D numpy массив данных lead3.
    intervals (numpy.ndarray): 1D numpy массив интервалов.
    r_pos (numpy.ndarray): 1D numpy массив позиций r.
    ind (int): Индекс интервала для построения.

    Возвращает:
    None
    """
    len_pr = int(intervals[ind] * 0.36 + 5)
    start = r_pos[ind] - len_pr
    stop = r_pos[ind] - int(len_pr * 0.05 + 10)
    fragment1 = lead1[start:stop]
    fragment2 = lead2[start:stop]
    fragment3 = lead3[start:stop]
    # p01.plot(fragment1, pen='g')
    # p01.plot(fragment2, pen='y')
    # p01.plot(fragment3, pen='c')
    fragment1 = filtfilt(bl, al, fragment1)
    fragment2 = filtfilt(bl, al, fragment2)
    fragment3 = filtfilt(bl, al, fragment3)
    # p01.plot(fragment1, pen='g')
    # p01.plot(fragment2, pen='y')
    # p01.plot(fragment3, pen='c')
    # p02.plot(lead1[r_pos[ind] - 1600:r_pos[ind] + 1600], pen='g')
    # p02.plot(lead2[r_pos[ind] - 1600:r_pos[ind] + 1600] - 2, pen='y')
    # p02.plot(lead3[r_pos[ind] - 1600:r_pos[ind] + 1600] - 4, pen='c')


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
    presence_PR1 = np.zeros(r_pos.size, dtype=int)
    presence_PR2 = np.zeros(r_pos.size, dtype=int)
    presence_PR3 = np.zeros(r_pos.size, dtype=int)
    intervals_PR1 = np.array([], dtype=int)
    intervals_PR2 = np.array([], dtype=int)
    intervals_PR3 = np.array([], dtype=int)
    inds_PR1 = np.array([], dtype=int)
    inds_PR2 = np.array([], dtype=int)
    inds_PR3 = np.array([], dtype=int)
    # arr_amp_p1 = np.array([])
    # arr_amp_p2 = np.array([])
    # arr_amp_p3 = np.array([])
    arr_amp_p1 = np.zeros(r_pos.size)
    arr_amp_p2 = np.zeros(r_pos.size)
    arr_amp_p3 = np.zeros(r_pos.size)

    for i in inds_min:
        if (chars[i] == 'N') and (chars[i - 1] == 'N'):  # and (chars[i + 1] == 'N'):
            len_pr = int(intervals[i] * 0.36 + 5)  # len_pr = int(intervals[i] * 0.36 + 5)
            start = r_pos[i] - len_pr
            stop = r_pos[i] - 6
            # stop = r_pos[i] - int(len_pr * 0.05 + 7)
            fragment1 = lead1[start:stop]
            fragment2 = lead2[start:stop]
            fragment3 = lead3[start:stop]
            # if i == 67152:
            #     p04.plot(fragment1, pen='g', title='Fragment1')
            #     p04.plot(fragment2, pen='y', title='Fragment2')
            #     p04.plot(fragment3, pen='c', title='Fragment3')
            fragment1 = filtfilt(bl, al, fragment1)
            fragment2 = filtfilt(bl, al, fragment2)
            fragment3 = filtfilt(bl, al, fragment3)
            # if i == 67154:
            #     p04.plot(fragment1, pen='g', title='Fragment1')
            #     p04.plot(fragment2, pen='y', title='Fragment2')
            #     p04.plot(fragment3, pen='c', title='Fragment3')

            pos_loc_max_frag1 = argrelmax(fragment1)[0]
            pos_loc_max_frag2 = argrelmax(fragment2)[0]
            pos_loc_max_frag3 = argrelmax(fragment3)[0]

            if pos_loc_max_frag1.size >= 1:
                ind_max1 = np.argmax(fragment1[pos_loc_max_frag1])
                isoline1 = np.min(
                    (np.min(fragment1[:pos_loc_max_frag1[ind_max1]]), np.min(fragment1[pos_loc_max_frag1[ind_max1]:])))
                amp_p1 = fragment1[pos_loc_max_frag1[ind_max1]] - isoline1
                if 1.0 > amp_p1 > 0.01:  # if 1.0 > amp_p1 > 0.001
                    intervals_PR1 = np.append(intervals_PR1, len_pr - pos_loc_max_frag1[ind_max1])
                    inds_PR1 = np.append(inds_PR1, i)
                    arr_amp_p1[i] = amp_p1
            if pos_loc_max_frag2.size >= 1:
                ind_max2 = np.argmax(fragment2[pos_loc_max_frag2])
                isoline2 = np.min(
                    (np.min(fragment2[:pos_loc_max_frag2[ind_max2]]), np.min(fragment2[pos_loc_max_frag2[ind_max2]:])))
                amp_p2 = fragment2[pos_loc_max_frag2[ind_max2]] - isoline2
                if 1.0 > amp_p2 > 0.01:  # 0.05
                    intervals_PR2 = np.append(intervals_PR2, len_pr - pos_loc_max_frag2[ind_max2])
                    inds_PR2 = np.append(inds_PR2, i)
                    arr_amp_p2[i] = amp_p2
            if pos_loc_max_frag3.size >= 1:
                ind_max3 = np.argmax(fragment3[pos_loc_max_frag3])
                isoline3 = np.min(
                    (np.min(fragment3[:pos_loc_max_frag3[ind_max3]]), np.min(fragment3[pos_loc_max_frag3[ind_max3]:])))
                amp_p3 = fragment3[pos_loc_max_frag3[ind_max3]] - isoline3
                if 1.0 > amp_p3 > 0.01:
                    intervals_PR3 = np.append(intervals_PR3, len_pr - pos_loc_max_frag3[ind_max3])
                    inds_PR3 = np.append(inds_PR3, i)
                    arr_amp_p3[i] = amp_p3

    marr_amp_p1 = interp_pr(arr_amp_p1)
    marr_amp_p2 = interp_pr(arr_amp_p2)
    marr_amp_p3 = interp_pr(arr_amp_p3)
    marr_amp_p1 = moving_average(marr_amp_p1, 15)
    marr_amp_p2 = moving_average(marr_amp_p2, 15)
    marr_amp_p3 = moving_average(marr_amp_p3, 15)

    # if intervals_PR1.size > 351:  # 21
    #     intervals_PR1 = moving_average(intervals_PR1, 351)
    # if intervals_PR2.size > 351:
    #     intervals_PR2 = moving_average(intervals_PR2, 351)
    # if intervals_PR3.size > 351:
    #     intervals_PR3 = moving_average(intervals_PR3, 351)

    presence_PR1[inds_PR1] = intervals_PR1
    presence_PR2[inds_PR2] = intervals_PR2
    presence_PR3[inds_PR3] = intervals_PR3
    presence_PR1 = interp_pr(presence_PR1)
    presence_PR2 = interp_pr(presence_PR2)
    presence_PR3 = interp_pr(presence_PR3)
    # presence_PR = np.array([presence_PR1], [presence_PR2], [presence_PR3])
    # presence_PR = np.mean(presence_PR)
    # presence_PR = presence_PR.astype(int)
    # p03.plot(presence_PR1, pen='g')
    # p03.plot(presence_PR2, pen='y')
    # p03.plot(presence_PR3, pen='c')
    presence_PR1 = moving_average(presence_PR1, 151)
    presence_PR2 = moving_average(presence_PR2, 151)
    presence_PR3 = moving_average(presence_PR3, 151)
    presence_PR1 = presence_PR1.astype(int)
    presence_PR2 = presence_PR2.astype(int)
    presence_PR3 = presence_PR3.astype(int)
    mean_PR = np.mean([presence_PR1, presence_PR2, presence_PR3], axis=0)
    mean_PR = moving_average(mean_PR, 15)
    mean_PR = mean_PR.astype(int)
    # p03.plot(presence_PR1, pen='g')
    # p03.plot(presence_PR2, pen='y')
    # p03.plot(presence_PR3, pen='c')
    # p03.plot(mean_PR, pen='y')
    # p03.plot(arr_amp_p1, pen='g')
    # p03.plot(arr_amp_p2-1, pen='g')
    # p03.plot(arr_amp_p3-2, pen='g')
    # p03.plot(marr_amp_p1, pen='g')
    # p03.plot(marr_amp_p2 - 1, pen='g')
    # p03.plot(marr_amp_p3 - 2, pen='g')

    return presence_PR1, presence_PR2, presence_PR3, marr_amp_p1, marr_amp_p2, marr_amp_p3, mean_PR


@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars, pr1, pr2, pr3, marr_amp_p1, marr_amp_p2, marr_amp_p3, mean_PR):
    global bl, al
    # bl = [0.00475052, 0.01425157, 0.01425157, 0.00475052]
    # al = [1.0, -2.25008508, 1.75640138, -0.46831211]
    # bl = [0.01591456, 0.03182911, 0.01591456]
    # al = [1.0, -1.61277905, 0.67643727]
    # bh = [0.9989957, - 0.9989957]
    # ah = [1.0, - 0.9979914]
    p1 = np.zeros(len(r_pos), dtype=np.float32)
    p1[:5] = 1.0
    p2 = np.zeros(len(r_pos), dtype=np.float32)
    p2[:5] = 1.0
    p3 = np.zeros(len(r_pos), dtype=np.float32)
    p3[:5] = 1.0
    out = np.zeros(len(r_pos), dtype=np.float32)
    # buff = np.array([3, 3, 3])
    # mean_interval = (np.mean(intervals) + 400) * 0.06
    # mean_interval = (np.mean(intervals) + 400) * 1.5
    for i in range(4, len(r_pos)):
        fragment_l1 = lead1[r_pos[i] - mean_PR[i] - 17:r_pos[i] - mean_PR[i] + 16]  # -15 +16
        fragment_l2 = lead2[r_pos[i] - mean_PR[i] - 17:r_pos[i] - mean_PR[i] + 16]
        fragment_l3 = lead3[r_pos[i] - mean_PR[i] - 17:r_pos[i] - mean_PR[i] + 16]
        # fragment_l1 = lead1[r_pos[i] - int(mean_PR[i] * 1.46):r_pos[i] - int(mean_PR[i] * 0.57)]
        # fragment_l2 = lead2[r_pos[i] - int(mean_PR[i] * 1.46):r_pos[i] - int(mean_PR[i] * 0.57)]
        # fragment_l3 = lead3[r_pos[i] - int(mean_PR[i] * 1.46):r_pos[i] - int(mean_PR[i] * 0.57)]
        if not (chars[i] == 'N'):
            p1[i] = 1.0
            p2[i] = 1.0
            p3[i] = 1.0
            # p1[i] = p1[i-1]
            # p2[i] = p2[i-1]
            # p3[i] = p3[i-1]
        else:
            if mean_PR[i] < int(intervals[i] * 0.5):

                fragment_l1_f = filtfilt(bl, al, fragment_l1)[5:-3]  # [7:-7]
                ind_max1 = argrelmax(fragment_l1_f)[0]
                if ind_max1.size >= 1:  # if ind_max1.size == 1:
                    isoline1 = np.min((np.min(fragment_l1_f[:ind_max1[-1]]), np.min(fragment_l1_f[ind_max1[-1]:])))
                    amp_p1 = fragment_l1_f[ind_max1[-1]] - isoline1
                    if amp_p1 > 0.003:
                        p1[i] = 1.0

                fragment_l2_f = filtfilt(bl, al, fragment_l2)[5:-3]
                ind_max2 = argrelmax(fragment_l2_f)[0]
                if ind_max2.size >= 1:
                    isoline2 = np.min((np.min(fragment_l2_f[:ind_max2[-1]]), np.min(fragment_l2_f[ind_max2[-1]:])))
                    amp_p2 = fragment_l2_f[ind_max2[-1]] - isoline2
                    if amp_p2 > 0.003:
                        p2[i] = 1.0

                fragment_l3_f = filtfilt(bl, al, fragment_l3)[5:-3]
                ind_max3 = argrelmax(fragment_l3_f)[0]
                if ind_max3.size >= 1:
                    isoline3 = np.min((np.min(fragment_l3_f[:ind_max3[-1]]), np.min(fragment_l3_f[ind_max3[-1]:])))
                    amp_p3 = fragment_l3_f[ind_max3[-1]] - isoline3
                    if amp_p3 > 0.003:
                        p3[i] = 1.0

            # logging.info(f"amp_p1 = {amp_p1:.4f}, amp_p2 = {amp_p2:.4f}, amp_p3 = {amp_p3:.4f}")
        if i == 54000:
            # p01.plot(fragment_l1, pen='g')
            # p01.plot(fragment_l2, pen='y')
            # p01.plot(fragment_l3, pen='c')
            # p01.plot(fragment_l1_f, pen='g')
            # p01.plot(fragment_l2_f, pen='y')
            # p01.plot(fragment_l3_f, pen='c')
            p02.plot(lead1[r_pos[i] - 2000:r_pos[i] + 2000], pen='g')
            p02.plot(lead2[r_pos[i] - 2000:r_pos[i] + 2000] - 2, pen='y')
            p02.plot(lead3[r_pos[i] - 2000:r_pos[i] + 2000] - 4, pen='c')
            print(p1[i - 4:i + 1])
            print(p2[i - 4:i + 1])
            print(p3[i - 4:i + 1])
            print(f"{amp_p1:.5f}, {amp_p2:.5f}, {amp_p3:.5f}")
        # sum_p1 = np.sum(p1[i - 4:i - 1]) + p1[i-1] * 2.0 + p1[i] * 3.0
        # sum_p2 = np.sum(p2[i - 4:i - 1]) + p2[i-1] * 2.0 + p2[i] * 3.0
        # sum_p3 = np.sum(p3[i - 4:i - 1]) + p3[i-1] * 2.0 + p3[i] * 3.0
        sum_p1 = p1[i - 4] + p1[i - 3] * 1.5 + p1[i - 2] * 3.0 + p1[i - 1] * 1.5 + p1[i]
        sum_p2 = p2[i - 4] + p2[i - 3] * 1.5 + p2[i - 2] * 3.0 + p2[i - 1] * 1.5 + p2[i]
        sum_p3 = p3[i - 4] + p3[i - 3] * 1.5 + p3[i - 2] * 3.0 + p3[i - 1] * 1.5 + p3[i]
        # sum_p1 = p1[i - 4] + p1[i - 3] * 1.5 + p1[i - 2] * 2.0 + p1[i - 1] * 1.5 + p1[i]
        # sum_p2 = p2[i - 4] + p2[i - 3] * 1.5 + p2[i - 2] * 2.0 + p2[i - 1] * 1.5 + p2[i]
        # sum_p3 = p3[i - 4] + p3[i - 3] * 1.5 + p3[i - 2] * 2.0 + p3[i - 1] * 1.5 + p3[i]

        if np.sum(fragment_l1) == 0.0:
            sum_p1 = (sum_p2 + sum_p3) / 2.0
        elif np.sum(fragment_l2) == 0.0:
            sum_p2 = (sum_p1 + sum_p3) / 2.0
        elif np.sum(fragment_l3) == 0.0:
            sum_p3 = (sum_p1 + sum_p2) / 2.0
        # if ((sum_p1 < 4.5) and (sum_p2 > 4.5) and (sum_p3 > 4.5)) or (
        #         (sum_p1 > 4.5) and (sum_p2 < 4.5) and (sum_p3 < 4.5)):
        #     sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif ((sum_p2 < 4.5) and (sum_p1 > 4.5) and (sum_p3 > 4.5)) or (
        #         (sum_p2 > 4.5) and (sum_p1 < 4.5) and (sum_p3 < 4.5)):
        #     sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif ((sum_p3 < 4.5) and (sum_p1 > 4.5) and (sum_p2 > 4.5)) or (
        #         (sum_p3 > 4.5) and (sum_p1 < 4.5) and (sum_p2 < 4.5)):
        #     sum_p3 = (sum_p1 + sum_p2) / 2.0
        #     print(p1[i - 4:i + 1])
        #     print(p2[i - 4:i + 1])
        #     print(p3[i - 4:i + 1])
        # sum_p1 = np.sum(p1[i - 4:i + 1])
        # sum_p2 = np.sum(p2[i - 4:i + 1])
        # sum_p3 = np.sum(p3[i - 4:i + 1])
        # if np.sum(fragment_l1) == 0.0:
        #     sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif np.sum(fragment_l2) == 0.0:
        #     sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif np.sum(fragment_l3) == 0.0:
        #     sum_p3 = (sum_p1 + sum_p2) / 2.0
        # if ((sum_p1 < 2.5) and (sum_p2 > 2.5) and (sum_p3 > 2.5)) or (
        #         (sum_p1 > 2.5) and (sum_p2 < 2.5) and (sum_p3 < 2.5)):
        #     if sum_p1 < 5.0:
        #         sum_p1 = (sum_p2 + sum_p3) / 2.0
        # elif ((sum_p2 < 2.5) and (sum_p1 > 2.5) and (sum_p3 > 2.5)) or (
        #         (sum_p2 > 2.5) and (sum_p1 < 2.5) and (sum_p3 < 2.5)):
        #     if sum_p2 < 5.0:
        #         sum_p2 = (sum_p1 + sum_p3) / 2.0
        # elif ((sum_p3 < 2.5) and (sum_p1 > 2.5) and (sum_p2 > 2.5)) or (
        #         (sum_p3 > 2.5) and (sum_p1 < 2.5) and (sum_p2 < 2.5)):
        #     if sum_p3 < 5.0:
        #         sum_p3 = (sum_p1 + sum_p2) / 2.0
        sum_buff = sum_p1 + sum_p2 + sum_p3
        # sum_p_i = p1[i] + p2[i] + p3[i]
        # if sum_p_i == 3.0:
        #     sum_buff += 8.0
        # elif (sum_p_i == 0.0):
        #     sum_buff -= 8.0
        # if sum_buff < 0.0:
        #     sum_buff = 0.0
        # sum_buff = 27.0 - sum_buff   # 24.0 - sum_buff
        # sum_buff = 24.0 - sum_buff
        # if sum_buff > 20.0:
        #     sum_buff = 20.0
        # if sum_buff < 0.0:
        #     sum_buff = 0.0
        # out[i - 2] = sum_buff
        out[i - 2] = sum_buff * 3.75#4.55
        # out[i - 2] = (20.0 - sum_buff) * 3.0  # (18.0 - sum_buff) * 5.0
        # a = 1
    # out = medfilt(out, 15)
    # out = moving_average(out, 40)
    # out = medfilt(out, 15)
    # out = moving_average(out, 50)
    # out = moving_average(out, 25)
    # out = moving_average(out, 31)
    # out = moving_average(out, 31)
    # out = truncate_win(out, 0.2, 10)
    # out = truncate_win(out, 0.2, 20)
    # out = truncate_win(out, 0.2, 30)
    # out = truncate_win(out, 0.2, 10)
    # out = truncate_win2(out, 0.8, 50)
    out = -(out - np.max(out))
    return out


def main():
    fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
    print(fdir)
    lead1 = np.load(fdir + "/clean_lead1.npy")
    lead2 = np.load(fdir + "/clean_lead2.npy")
    lead3 = np.load(fdir + "/clean_lead3.npy")
    # len_lead = lead1.size
    # end_lead = int(len_lead * 0.9)
    # lead1 = lead1[:end_lead]
    # lead2 = lead2[:end_lead]
    # lead3 = lead3[:end_lead]
    get_S1(fdir)
    r_pos, intervals, chars, forms = parse_B1_txt(fdir)
    # plot_select_p(lead1, lead2, lead3, intervals, r_pos, 44215)
    # p.plot(intervals, pen="g")
    fintervals = del_V_S(intervals, chars)
    # p.plot(fintervals, pen="g")
    m_fintervals = medfilt(fintervals, 21)  # 5
    # p.plot(m_fintervals, pen="y")
    cleen_fintervals = m_fintervals + (fintervals - m_fintervals) * 0.2
    p.plot(cleen_fintervals, pen="g")
    mean_fintervals = np.mean(cleen_fintervals)
    line_fintervals = np.ones(cleen_fintervals.size) * mean_fintervals
    p.plot(line_fintervals, pen="m")

    inds_min = get_inds_min_diff(intervals)
    pos_p1, pos_p2, pos_p3, mean_p1, mean_p2, mean_p3, mean_PR = get_p_pos(lead1, lead2, lead3, intervals, r_pos, chars,
                                                                           inds_min)
    coef_p = get_P(lead1, lead2, lead3, intervals, r_pos, chars, pos_p1, pos_p2, pos_p3, mean_p1, mean_p2, mean_p3,
                   mean_PR)
    # p.plot(coef_p, pen='m')
    # coef_p = medfilt(coef_p, 5)
    # coef_p = moving_average(coef_p, 31)
    coef_p = truncate_win2(coef_p, 0.8, 500)  # 0.9, 500
    coef_p = moving_average(coef_p, 31)
    coef_p = truncate_win2(coef_p, 0.7, 300)
    p.plot(coef_p, pen='y')

    coef_fibr = get_scatter_coef(fintervals)
    # p.plot(coef_fibr, pen='c')
    # p.plot(cleen_fintervals - coef_fibr, pen='r')
    # s_coef_fibr = sigmoid(coef_fibr, 1.0)
    coef_fibr = truncate_win2(coef_fibr, 0.8, 500)
    coef_fibr = truncate_win2(coef_fibr, 0.7, 300)
    # coef_fibr = magnific_ch(coef_fibr, 1.5)
    p.plot(coef_fibr, pen='c')
    # p.plot(s_coef_fibr - 50, pen='m')

    # s_coef_fibr = sigmoid(coef_fibr, 0.1)
    # p.plot(s_coef_fibr - 50, pen='w')
    e_coef_p = coef_p * coef_fibr * 1.0  # 0.3
    # mean_ch, over_mean, under_mean = get_attr(e_coef_p, 10)
    # attr = np.abs(e_coef_p - mean_ch)
    # p.plot(attr, pen='w')
    # p.plot(mean_ch, pen='c')
    # p.plot(over_mean, pen='w')
    # p.plot(under_mean, pen='b')
    # p.plot((over_mean * mean_ch * under_mean) * 0.000015, pen='y')
    # mean_ch, over_mean, under_mean = get_attr(e_coef_p, 100)
    # p.plot((over_mean * mean_ch * under_mean) * 0.000015, pen='y')
    # p.plot((over_mean * under_mean)**0.5, pen='brown')
    # p.plot(e_coef_p, pen='m')
    p_count_coef = round(e_coef_p[e_coef_p > cleen_fintervals].size / e_coef_p.size, 4)
    print(f"p_count_coef = {p_count_coef:.4f}")

    # max_coef_fibr = over_range(coef_fibr)
    # k = 0.1
    # print(f"k = {k:.4f}")
    # m = 1.0
    # if p_count_coef <= 0.0001:
    #     m = 0.5
    # print(f"m = {m:.4f}")
    # s_coef_fibr = sigmoid(coef_fibr, k) * max_coef_fibr * m
    # p.plot(s_coef_fibr, pen='w')
    # e_coef_p = coef_p * s_coef_fibr * 1.0

    mean_coef_p = np.mean(e_coef_p)
    mean_coef_p2 = np.mean(e_coef_p[e_coef_p > mean_coef_p])
    mean_coef_p2 = np.mean(e_coef_p[e_coef_p > mean_coef_p2])
    mean_coef_p2 = np.mean(e_coef_p[e_coef_p > mean_coef_p2])
    if len(e_coef_p[e_coef_p > mean_coef_p2]) > 0:
        mean_coef_p2 = np.mean(e_coef_p[e_coef_p > mean_coef_p2])
    mean_coef_p3 = np.mean(e_coef_p[e_coef_p < mean_coef_p])
    mean_coef_p3 = np.mean(e_coef_p[e_coef_p < mean_coef_p3])
    mean_coef_p3 = np.mean(e_coef_p[e_coef_p < mean_coef_p3])
    if len(e_coef_p[e_coef_p < mean_coef_p3]) > 0:
        mean_coef_p3 = np.mean(e_coef_p[e_coef_p < mean_coef_p3])
    mean_23 = (mean_coef_p2 + mean_coef_p3) / 2

    mean_line = np.ones(len(e_coef_p)) * mean_coef_p
    mean_line2 = np.ones(len(e_coef_p)) * mean_coef_p2
    mean_line3 = np.ones(len(e_coef_p)) * mean_coef_p3
    # mean_line23 = np.ones(len(e_coef_p)) * mean_23
    coef2 = mean_coef_p2 / mean_fintervals
    coef = mean_coef_p / mean_fintervals
    coef3 = mean_coef_p3 / mean_fintervals
    coef23 = mean_23 / mean_fintervals
    max_coef_p = np.max(e_coef_p) / mean_fintervals
    sum_coef = coef2 + coef + coef3
    print(f"max_coef_p = {max_coef_p:.2f}")
    print(f"coef2 = {coef2:.2f}, coef = {coef:.2f}, coef3 = {coef3:.2f}")
    print(f"sum coef = {sum_coef:.2f}")
    norm_coef = 1.0
    if coef3 > 1.0:
        e_coef_p = truncate_ch(e_coef_p, 0.8)
        print(f"1 over = 2, 23, 3")
    elif (coef2 < 1.0) and (p_count_coef == 0.0000):
        norm_coef = 0.75
        print(f"2 (p_count_coef == 0.0) under = 2, 23, 3")
    elif (coef2 < 1.0) and (p_count_coef > 0.0000) and (sum_coef < 0.98):
        coef_fibr = magnific_ch(coef_fibr, 1.5)
        e_coef_p = coef_p * coef_fibr
        print(f"3 (p_count_coef > 0.0) and (sum_coef < 0.98) under = 2, 23, 3")
    elif (coef2 < 1.0) and (p_count_coef > 0.0000) and (sum_coef >= 0.98):
        # coef_fibr = magnific_ch(coef_fibr, 1.5)
        # e_coef_p = coef_p * coef_fibr
        # norm_coef = 0.75
        print(f"4 (p_count_coef > 0.0) and (sum_coef >= 0.98) under = 2, 23, 3")
    else:
        # coef_fibr = truncate_win2(coef_fibr, 0.7, 500)
        # e_coef_p = coef_p * coef_fibr
        norm_coef = 1.0
        print(f"5 else")

    print(f"norm_coef = {norm_coef:.2f}")
    e_coef_p *= norm_coef
    p.plot(e_coef_p[:int(len(e_coef_p) * 0.999)], pen='r')
    # p.plot(e_coef_p / cleen_fintervals * 100, pen='w')
    # p.plot(np.ones(len(e_coef_p)) * 100, pen='w')
    p.plot(mean_line, pen='w')
    p.plot(mean_line2, pen='w')
    p.plot(mean_line3, pen='w')
    # p.plot(mean_line23, pen='g')


if __name__ == "__main__":
    main()

sys.exit(app.exec())
