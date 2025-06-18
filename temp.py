from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, argrelmax
from functions import parse_B1_txt, get_Q, moving_average, del_V_S, truncate_win2, interp_pr, truncate_ch, \
    get_scatter_coef, win_sum_coef, step_moving_average, get_p_amp, get_over_threshold
from time import time
import logging

logging.basicConfig(filename='experiment.log', level=logging.INFO, filemode="w",
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
p.setTitle('p')
p01 = pg.plot()
p01.showGrid(x=True, y=True)
p01.setTitle('p01')
p02 = pg.plot()
p02.showGrid(x=True, y=True)
p02.setTitle('p02')
# p03 = pg.plot()
# p03.showGrid(x=True, y=True)
# p03.setTitle('p03')
p04 = pg.plot()
p04.showGrid(x=True, y=True)
p04.setTitle('p04')

bl, al = butter(3, 12.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
b, a = butter(2, 13.0, 'lp', fs=250)  # 3, 2.0, 'lp', fs=250
bh, ah = butter(1, 0.2, 'hp', fs=250)  # 3, 1.0, 'hp', fs=250
# print(f"bl: {bl}, al: {al}")
# print(f"b: {b}, a: {a}")
k = 47400


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
    global bl, al, k
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
            stop = r_pos[i] - 7  # r_pos[i] - 6
            # stop = r_pos[i] - int(len_pr * 0.05 + 7)
            fragment1 = lead1[start:stop]
            fragment2 = lead2[start:stop]
            fragment3 = lead3[start:stop]
            if (k + 15) > i >= k:
                p04.plot(fragment1 - 0.1, pen='g', title='Fragment1')
                p04.plot(fragment2 - 0.3, pen='y', title='Fragment2')
                p04.plot(fragment3 - 0.5, pen='c', title='Fragment3')
            fragment1 = filtfilt(bl, al, fragment1)
            fragment2 = filtfilt(bl, al, fragment2)
            fragment3 = filtfilt(bl, al, fragment3)
            if (k + 15) > i >= k:
                p04.plot(fragment1 - 0.2, pen='g', title='Fragment1')
                p04.plot(fragment2 - 0.4, pen='y', title='Fragment2')
                p04.plot(fragment3 - 0.6, pen='c', title='Fragment3')

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
    mean_amp_p1 = np.mean(arr_amp_p1)
    mean_amp_p2 = np.mean(arr_amp_p2)
    mean_amp_p3 = np.mean(arr_amp_p3)
    print(f"mean_amp_p1: {mean_amp_p1}")
    print(f"mean_amp_p2: {mean_amp_p2}")
    print(f"mean_amp_p3: {mean_amp_p3}")
    mean_PR1 = int(intervals_PR1.mean())
    mean_PR2 = int(intervals_PR2.mean())
    mean_PR3 = int(intervals_PR3.mean())
    print(f"mean_PR1: {mean_PR1}     {intervals_PR1.size}")
    print(f"mean_PR2: {mean_PR2}     {intervals_PR2.size}")
    print(f"mean_PR3: {mean_PR3}     {intervals_PR3.size}")
    # marr_amp_p1 = interp_pr(arr_amp_p1)
    # marr_amp_p2 = interp_pr(arr_amp_p2)
    # marr_amp_p3 = interp_pr(arr_amp_p3)
    # marr_amp_p1 = moving_average(marr_amp_p1, 15)
    # marr_amp_p2 = moving_average(marr_amp_p2, 15)
    # marr_amp_p3 = moving_average(marr_amp_p3, 15)

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
    # presence_PR1 = moving_average(presence_PR1, 151)
    # presence_PR2 = moving_average(presence_PR2, 151)
    # presence_PR3 = moving_average(presence_PR3, 151)
    # presence_PR1 = presence_PR1.astype(int)
    # presence_PR2 = presence_PR2.astype(int)
    # presence_PR3 = presence_PR3.astype(int)
    # mean_PR = np.mean([presence_PR1, presence_PR2, presence_PR3], axis=0)
    # mean_PR = moving_average(mean_PR, 15)
    # mean_PR = mean_PR.astype(int)
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

    return mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1, mean_PR2, mean_PR3


@time_fun
def get_P(lead1, lead2, lead3, intervals, r_pos, chars, mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1, mean_PR2,
          mean_PR3):
    global bl, al, k, b, a
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
    mean_PR1_copy = mean_PR1
    mean_PR2_copy = mean_PR2
    mean_PR3_copy = mean_PR3
    for i in range(4, len(r_pos)):  # range(4, len(r_pos))
        start_range_interval = int(intervals[i] ** 0.5 * 3.5)  # int(intervals[i]**0.5 * 3.0)
        stop_range_interval = int(intervals[i] ** 0.5 * 0.65)
        start_slice = r_pos[i] - start_range_interval
        end_slice = r_pos[i] - stop_range_interval
        start_slice1 = r_pos[i] - mean_PR1_copy - 15
        start_slice2 = r_pos[i] - mean_PR2_copy - 15
        start_slice3 = r_pos[i] - mean_PR3_copy - 15
        # fragment_l1 = lead1[start_slice:end_slice]
        # fragment_l2 = lead2[start_slice:end_slice]
        # fragment_l3 = lead3[start_slice:end_slice]
        # if start_range_interval * 0.2 > 10:
        #     end_slice = r_pos[i] - 10
        # end_slice = start_slice + 30
        end_slice1 = r_pos[i] - mean_PR1_copy + int(mean_PR1_copy ** 0.5 * 3.0)  # +15
        end_slice2 = r_pos[i] - mean_PR2_copy + int(mean_PR2_copy ** 0.5 * 3.0)  # +15
        end_slice3 = r_pos[i] - mean_PR3_copy + int(mean_PR3_copy ** 0.5 * 3.0)  # +15
        if mean_PR1 > intervals[i] // 3:
            # if start_range_interval < mean_PR1:
            # fragment_l1 = lead1[start_slice1:end_slice1]
            fragment_l1 = lead1[start_slice:end_slice]
        else:
            fragment_l1 = lead1[start_slice1:end_slice1]
        if mean_PR2 > intervals[i] // 3:
            # if start_range_interval < mean_PR2:
            # fragment_l2 = lead2[start_slice2:end_slice2]
            fragment_l2 = lead2[start_slice:end_slice]
        else:
            fragment_l2 = lead2[start_slice2:end_slice2]
        if mean_PR3 > intervals[i] // 3:
            # if start_range_interval < mean_PR3:
            # fragment_l3 = lead3[start_slice3:end_slice3]
            fragment_l3 = lead3[start_slice:end_slice]
        else:
            fragment_l3 = lead3[start_slice3:end_slice3]
        # fragment_l1 = lead1[r_pos[i] - mean_PR1_copy - 15:r_pos[i] - mean_PR1_copy + 15]  # -+15
        # fragment_l2 = lead2[r_pos[i] - mean_PR2_copy - 15:r_pos[i] - mean_PR2_copy + 15]
        # fragment_l3 = lead3[r_pos[i] - mean_PR3_copy - 15:r_pos[i] - mean_PR3_copy + 15]
        bp = 0.0
        amp_p1 = 0.0
        amp_p2 = 0.0
        amp_p3 = 0.0
        if not (chars[i] == 'N'):
            p1[i] = 1.0
            p2[i] = 1.0
            p3[i] = 1.0
            # p1[i] = 0.6
            # p2[i] = 0.6
            # p3[i] = 0.6
        else:
            # if mean_PR[i] < int(intervals[i] * 0.5):
            # if True:
            fragment_l1_f = filtfilt(b, a, fragment_l1)
            fragment_l1_f = filtfilt(bh, ah, fragment_l1_f)
            pzub, amp_pzub1 = get_p_amp(fragment_l1_f, mean_amp_p1)
            p1[i] = pzub
            # ind_max1 = argrelmax(fragment_l1_f)[0]
            # if ind_max1.size == 1:  # if ind_max1.size == 1:
            #     isoline1 = np.max((np.min(fragment_l1_f[:ind_max1[-1]]), np.min(fragment_l1_f[ind_max1[-1]:])))
            #     amp_p1 = fragment_l1_f[ind_max1[-1]] - isoline1
            #     if (amp_p1 > mean_amp_p1 * 0.1) and (amp_p1 > 0.001):  # > 0.0001
            #         # if amp_p1 > 0.0015:   # > 0.0001
            #         p1[i] = 1.0

            fragment_l2_f = filtfilt(b, a, fragment_l2)
            fragment_l2_f = filtfilt(bh, ah, fragment_l2_f)
            pzub, amp_pzub2 = get_p_amp(fragment_l2_f, mean_amp_p2)
            p2[i] = pzub
            # ind_max2 = argrelmax(fragment_l2_f)[0]
            # if ind_max2.size == 1:
            #     isoline2 = np.max((np.min(fragment_l2_f[:ind_max2[-1]]), np.min(fragment_l2_f[ind_max2[-1]:])))
            #     amp_p2 = fragment_l2_f[ind_max2[-1]] - isoline2
            #     if (amp_p2 > mean_amp_p2 * 0.1) and (amp_p2 > 0.001):
            #         # if amp_p2 > 0.0015:
            #         p2[i] = 1.0

            fragment_l3_f = filtfilt(b, a, fragment_l3)
            fragment_l3_f = filtfilt(bh, ah, fragment_l3_f)
            pzub, amp_pzub3 = get_p_amp(fragment_l3_f, mean_amp_p3)
            p3[i] = pzub
            # ind_max3 = argrelmax(fragment_l3_f)[0]
            # if 3 > ind_max3.size >= 1:
            #     isoline3 = np.max((np.min(fragment_l3_f[:ind_max3[-1]]), np.min(fragment_l3_f[ind_max3[-1]:])))
            #     amp_p3 = fragment_l3_f[ind_max3[-1]] - isoline3
            #     if (amp_p3 > mean_amp_p3 * 0.1) and (amp_p3 > 0.001):
            #         # if amp_p3 > 0.0015:
            #         p3[i] = 1.0

            # logging.info(f"amp_p1 = {amp_p1:.4f}, amp_p2 = {amp_p2:.4f}, amp_p3 = {amp_p3:.4f}")
        if i == k:
            p01.plot(fragment_l1, pen='g')
            p01.plot(fragment_l2, pen='y')
            p01.plot(fragment_l3, pen='c')
            p01.plot(fragment_l1_f - 0.2, pen='g')
            p01.plot(fragment_l2_f - 0.2, pen='y')
            p01.plot(fragment_l3_f - 0.2, pen='c')
            p02.plot(lead1[r_pos[i] - 2000:r_pos[i] + 2000], pen='g')
            p02.plot(lead2[r_pos[i] - 2000:r_pos[i] + 2000] - 2, pen='y')
            p02.plot(lead3[r_pos[i] - 2000:r_pos[i] + 2000] - 4, pen='c')
            print(f"i = {i}")
            print(p1[i - 4:i + 1])
            print(p2[i - 4:i + 1])
            print(p3[i - 4:i + 1])
            print(f"{amp_pzub1:.5f}, {amp_pzub2:.5f}, {amp_pzub3:.5f}")
        # sum_p1 = np.sum(p1[i - 4:i - 1]) + p1[i-1] * 2.0 + p1[i] * 3.0
        # sum_p2 = np.sum(p2[i - 4:i - 1]) + p2[i-1] * 2.0 + p2[i] * 3.0
        # sum_p3 = np.sum(p3[i - 4:i - 1]) + p3[i-1] * 2.0 + p3[i] * 3.0
        # sum_p1 = p1[i - 4] + p1[i - 3] * 2.0 + p1[i - 2] * 3.0 + p1[i - 1] * 2.0 + p1[i]
        # sum_p2 = p2[i - 4] + p2[i - 3] * 2.0 + p2[i - 2] * 3.0 + p2[i - 1] * 2.0 + p2[i]
        # sum_p3 = p3[i - 4] + p3[i - 3] * 2.0 + p3[i - 2] * 3.0 + p3[i - 1] * 2.0 + p3[i]
        sum_p1 = p1[i - 4] + p1[i - 3] + p1[i - 2] + p1[i - 1] + p1[i]
        sum_p2 = p2[i - 4] + p2[i - 3] + p2[i - 2] + p2[i - 1] + p2[i]
        sum_p3 = p3[i - 4] + p3[i - 3] + p3[i - 2] + p3[i - 1] + p3[i]
        # arr_p = np.array([[p1[i - 4] + p1[i - 3] + p1[i - 2] + p1[i - 1] + p1[i]],
        #                   [p2[i - 4] + p2[i - 3] + p2[i - 2] + p2[i - 1] + p2[i]],
        #                   [p3[i - 4] + p3[i - 3] + p3[i - 2] + p3[i - 1] + p3[i]]])
        # if np.sum(fragment_l1) == 0.0:
        #     arr_p[0] = (arr_p[1] + arr_p[2]) / 2.0
        # elif np.sum(fragment_l2) == 0.0:
        #     arr_p[1] = (arr_p[0] + arr_p[2]) / 2.0
        # elif np.sum(fragment_l3) == 0.0:
        #     arr_p[2] = (arr_p[0] + arr_p[1]) / 2.0
        # sum_buff = arr_p.sum()
        # if arr_p[:][2].sum() == 3.0:
        #     sum_buff = 15.0
        # elif arr_p[:][2].sum() == 0.0:
        #     sum_buff = 0.0
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
        # win_t = intervals[i - 25:i + 26].copy()
        # sort_win_t = np.sort(win_t)[:-12]
        # mean_win = np.median(sort_win_t)
        out[i - 2] = sum_buff #* 1.4
        # out[i] = sum_buff * 4.5
        # out[i - 2] = (20.0 - sum_buff) * 3.0  # (18.0 - sum_buff) * 5.0
        bp1 = 1
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
    # out[-20:] = out[-20]
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
    get_Q(fdir)
    r_pos, intervals, chars, forms = parse_B1_txt(fdir)
    # plot_select_p(lead1, lead2, lead3, intervals, r_pos, 44215)
    # p.plot(intervals, pen="c")
    fintervals = del_V_S(intervals, chars)
    # fintervals = del_V_S(fintervals, chars)
    # fintervals = intervals
    # p.plot(fintervals, pen="m")
    # m_fintervals = medfilt(fintervals, 21)  # 5
    # p.plot(m_fintervals, pen="y")
    # clean_fintervals = m_fintervals + (fintervals - m_fintervals) * 0.2
    # clean_fintervals = m_fintervals #+ (np.abs(fintervals - m_fintervals))**0.5 * 0.2 * np.sign(fintervals - m_fintervals)
    # clean_fintervals = step_moving_average(fintervals, 6)
    clean_fintervals = step_moving_average(fintervals, 4)
    clean_fintervals = step_moving_average(clean_fintervals, 6)
    clean_fintervals = step_moving_average(clean_fintervals, 4)
    p.plot(clean_fintervals, pen="g")
    # mean_fintervals = np.mean(clean_fintervals)
    # line_fintervals = np.ones(clean_fintervals.size) * mean_fintervals
    # p.plot(line_fintervals, pen="m")

    inds_min = get_inds_min_diff(intervals)
    mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1, mean_PR2, mean_PR3 = get_p_pos(lead1, lead2, lead3, intervals,
                                                                                    r_pos, chars, inds_min)
    coef_p = get_P(lead1, lead2, lead3, intervals, r_pos, chars, mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1,
                   mean_PR2, mean_PR3)
    coef_p = truncate_win2(coef_p, 0.85, 500)  # 0.9, 500
    coef_p = moving_average(coef_p, 31)
    coef_p = truncate_win2(coef_p, 0.75, 300)
    # p.plot(coef_p, pen='y')
    coef_fibr = get_scatter_coef(fintervals)
    coef_fibr = truncate_win2(coef_fibr, 0.8, 500)
    coef_fibr = truncate_win2(coef_fibr, 0.7, 300)
    # p.plot(coef_fibr, pen='c')
    # e_coef_p = (3.1 * coef_p + 0.4 * coef_fibr) ** 2.0 * (0.5 + 100 / clean_fintervals)
    # p.plot(coef_p, pen='y')
    # m = 1.0
    n_coef_p = (coef_p + 0.0) * 1.0
    n_coef_p[n_coef_p < 0.0] = 0.0
    p.plot(n_coef_p, pen='y')
    # p.plot(coef_fibr, pen='c')
    n_coef_fibr = (coef_fibr + 1.5) * 1.0
    n_coef_fibr[n_coef_fibr < 0.0] = 0.0
    p.plot(n_coef_fibr, pen='c')
    n_mul_p_f = n_coef_p * n_coef_fibr * 3.0
    p.plot(n_mul_p_f, pen='m')
    n_sum_p_f = 0.1 * (1.8 * (n_coef_p + 0.0) + 0.15 * (n_coef_fibr - 0.0)) ** 2.0
    p.plot(n_sum_p_f, pen='w')
    n_intervals = clean_fintervals * 0.05
    e_coef_p = (n_sum_p_f + n_mul_p_f - n_intervals + 11.0) * 3.0
    e_coef_p[e_coef_p < 0.0] = 0.0
    # e_coef_p = (0.2 * (n_coef_p + n_coef_fibr) ** 2.0 + 5.0 * coef_p * coef_fibr) #* (0.5 + 100 / clean_fintervals)
    p_count_coef = round(e_coef_p[e_coef_p > clean_fintervals].size / e_coef_p.size, 6)
    print(f"p_count_coef = {p_count_coef:.6f}")

    norm_coef = 1.0
    print(f"norm_coef = {norm_coef:.2f}")
    e_coef_p *= norm_coef
    # p.plot(e_coef_p, pen='r', fillLevel=0.0, brush=(60, 180, 60, 140))
    p.plot(e_coef_p, pen='r')
    # p.plot(e_coef_p * (0.5 + 100 / clean_fintervals), pen='c')
    # over_thresh = get_over_threshold(e_coef_p)
    # over_line = np.ones(e_coef_p.size) * over_thresh
    # mean_line = np.ones(e_coef_p.size) * np.mean(e_coef_p)
    # p.plot(mean_line, pen='r')

    # w_sum_coef = win_sum_coef(e_coef_p, clean_fintervals, 10)
    # p.plot(w_sum_coef, pen='m')
    # w_mean_line = np.ones(w_sum_coef.size) * np.mean(w_sum_coef)
    # w_over_thresh = get_over_threshold(w_sum_coef)
    # w_over_line = np.ones(w_sum_coef.size) * w_over_thresh
    # p.plot(w_mean_line, pen='m')
    # w_o_sum_coef = w_sum_coef * (over_thresh / w_over_thresh * 2.0)
    # p.plot(w_o_sum_coef, pen='c')
    # w_norm_coef = np.mean(e_coef_p) / np.mean(w_sum_coef)
    # w_sum_coef = w_sum_coef * w_norm_coef
    # p.plot(w_sum_coef, pen='y')


if __name__ == "__main__":
    main()

sys.exit(app.exec())
