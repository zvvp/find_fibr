from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, argrelmax
from functions import parse_B1_txt, get_Q, moving_average, del_V_S, truncate_win2, get_scatter_coef1, \
    step_moving_average, get_fibr_num_samples, del_artifacts, get_inds_min_diff, get_p_pos, get_P, k
from time import time
import logging

logging.basicConfig(filename='experiment.log', level=logging.INFO, filemode="w",
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def plot_fragment_ecg(lead1, lead2, lead3, r_pos, ind, half_width, ind_plot = 1):
    start = r_pos[ind] - half_width
    stop = r_pos[ind] + half_width
    fragment1 = lead1[start:stop]
    fragment2 = lead2[start:stop]
    fragment3 = lead3[start:stop]
    if ind_plot == 1:
        p01.plot(fragment1, pen='g')
        p01.plot(fragment2-3, pen='y')
        p01.plot(fragment3-6, pen='c')
    elif ind_plot == 2:
        p02.plot(fragment1, pen='g')
        p02.plot(fragment2-3, pen='y')
        p02.plot(fragment3-6, pen='c')

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
# # p03 = pg.plot()
# # p03.showGrid(x=True, y=True)
# # p03.setTitle('p03')
# p04 = pg.plot()
# p04.showGrid(x=True, y=True)
# p04.setTitle('p04')

bl, al = butter(3, 12.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
b, a = butter(2, 13.0, 'lp', fs=250)  # 3, 2.0, 'lp', fs=250
bh, ah = butter(1, 0.2, 'hp', fs=250)  # 3, 1.0, 'hp', fs=250
# print(f"bl: {bl}, al: {al}")
# print(f"b: {b}, a: {a}")



def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n

    return wrapper

def main():
    fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
    print(fdir)
    lead1 = np.load(fdir + "/clean_lead1.npy")
    lead2 = np.load(fdir + "/clean_lead2.npy")
    lead3 = np.load(fdir + "/clean_lead3.npy")
    mask_pseudo_fibr = np.load(fdir + '/mask_pseudo_fibr.npy')
    p.plot(mask_pseudo_fibr, pen="w")
    get_Q(fdir)
    r_pos, intervals, chars, forms = parse_B1_txt(fdir)
    plot_fragment_ecg(lead1, lead2, lead3, r_pos, k, 5000, ind_plot=1)
    plot_fragment_ecg(lead1, lead2, lead3, r_pos, 29560, 5000, ind_plot=2)
    p.plot(intervals, pen="b")
    # fintervals = del_V_A(intervals, chars)
    fintervals = del_V_S(intervals, chars)
    fintervals = del_artifacts(fintervals, intervals)
    p.plot(fintervals, pen="m")
    # mean_intervals = np.mean(fintervals)
    # fintervals[fintervals < 75] = mean_intervals
    # fintervals[fintervals > 370] = mean_intervals
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
    clean_fintervals = moving_average(clean_fintervals, 12)
    p.plot(clean_fintervals, pen="g")

    inds_min = get_inds_min_diff(intervals)
    mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1, mean_PR2, mean_PR3 = get_p_pos(lead1, lead2, lead3, intervals,
                                                                                    r_pos, chars, inds_min)
    coef_p = get_P(lead1, lead2, lead3, intervals, r_pos, chars, mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1,
                   mean_PR2, mean_PR3)
    coef_p *= mask_pseudo_fibr
    coef_p = step_moving_average(coef_p, 12)
    coef_p = step_moving_average(coef_p, 8)
    coef_p = moving_average(coef_p, 12)
    # p.plot(coef_p, pen='y')
    coef_p = truncate_win2(coef_p, 0.85, 160)
    coef_p = truncate_win2(coef_p, 0.75, 80)
    p.plot(coef_p, pen='y')

    coef_fibr = get_scatter_coef1(fintervals)
    coef_fibr *= mask_pseudo_fibr
    coef_fibr = step_moving_average(coef_fibr, 12)
    coef_fibr = step_moving_average(coef_fibr, 8)
    coef_fibr = moving_average(coef_fibr, 12)
    # p.plot(coef_fibr, pen='c')
    coef_fibr = truncate_win2(coef_fibr, 0.85, 160)
    coef_fibr = truncate_win2(coef_fibr, 0.75, 80)
    p.plot(coef_fibr, pen='c')
    # print(f"div = {coef_fibr[71090] / coef_fibr[46420]:.2f}")
    # coef_p = coef_p - 0.0
    # coef_p[coef_p < 0.0] = 0.0
    # coef_fibr = coef_fibr - 0.5
    # coef_fibr[coef_fibr < 0.0] = 0.0
    # e_coef_p = coef_p * coef_fibr
    e_coef_p = coef_p * coef_fibr * (0.5 + 100 / clean_fintervals)
    # e_coef_p[e_coef_p < 0.0] = 0.0
    e_coef_p = step_moving_average(e_coef_p, 12)
    e_coef_p = step_moving_average(e_coef_p, 8)
    e_coef_p = moving_average(e_coef_p, 12)
    # p.plot(e_coef_p, pen='w')
    e_coef_p = truncate_win2(e_coef_p, 0.85, 160)
    e_coef_p = truncate_win2(e_coef_p, 0.75, 80)
    # fibr_num_sampl = e_coef_p[e_coef_p > clean_fintervals].size
    # print(f"fibr_num_sampl = {fibr_num_sampl}")
    # count_trans_coef = count_trans(e_coef_p, clean_fintervals)
    # print(f"count_trans_coef = {count_trans_coef}")
    # if fibr_num_sampl / e_coef_p.size > 0.9:
    #     e_coef_p = truncate_ch(e_coef_p, 0.75)
    start_ind_arr, stop_ind_arr, diff_stop_start = get_fibr_num_samples(e_coef_p, clean_fintervals)
    print(f"start_ind_arr = {start_ind_arr}")
    print(f"stop_ind_arr = {stop_ind_arr}")
    print(f"diff_stop_start = {diff_stop_start}")
    print(f"number of fibr = {diff_stop_start.size}")
    s_over =  np.sum(e_coef_p[e_coef_p > clean_fintervals] - clean_fintervals[e_coef_p > clean_fintervals])
    print(f"s_over = {s_over:.1f}")
    s_under = np.sum(clean_fintervals[e_coef_p < clean_fintervals] - e_coef_p[e_coef_p < clean_fintervals])
    print(f"s_under = {s_under:.1f}")

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
