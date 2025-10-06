from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, argrelmax
from functions import parse_B1_txt, get_Q, moving_average, del_V_S, truncate_win2, get_disp_coef1, \
    step_moving_average, get_fibr_num_samples, del_artifacts, get_inds_min_diff, get_p_pos, get_P, k, plot_select_p
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


bl, al = butter(3, 12.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
b, a = butter(2, 13.0, 'lp', fs=250)  # 3, 2.0, 'lp', fs=250
# print(f"b = {b}\na = {a}")
bh, ah = butter(1, 0.2, 'hp', fs=250)  # 3, 1.0, 'hp', fs=250

def time_fun(func):
    def wrapper(*args, **kwargs):
        start = time()
        n = func(*args, **kwargs)
        stop = time()
        print(func.__name__, stop - start)
        return n

    return wrapper

def main():
    fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar/fibr")
    print(fdir)
    lead1 = np.load(fdir + "/clean_lead1.npy")
    lead2 = np.load(fdir + "/clean_lead2.npy")
    lead3 = np.load(fdir + "/clean_lead3.npy")

    try:
        r_pos = np.load(fdir + "/r_pos1.npy")
        intervals = np.load(fdir + "/intervals1.npy")
        chars = np.load(fdir + "/chars1.npy")
        forms = np.load(fdir + "/forms1.npy")
    except FileNotFoundError:
        get_Q(fdir)
        r_pos, intervals, chars, forms = parse_B1_txt(fdir)
        np.save(fdir + "/r_pos.npy", r_pos)
        np.save(fdir + "/intervals.npy", intervals)
        np.save(fdir + "/chars.npy", chars)
        np.save(fdir + "/forms.npy", forms)

    try:
        mask_pseudo_fibr = np.load(fdir + '/mask_pseudo_fibr1.npy')
        print(f"mask == 1: {mask_pseudo_fibr[mask_pseudo_fibr == 1].size}  mask == 0: {mask_pseudo_fibr[mask_pseudo_fibr == 0].size}")
        p.plot(mask_pseudo_fibr * 10 - 12, pen="w")
    except FileNotFoundError:
        mask_pseudo_fibr = np.ones(intervals.size)
        np.save(fdir + '/mask_pseudo_fibr.npy', mask_pseudo_fibr)

    plot_fragment_ecg(lead1, lead2, lead3, r_pos, k, 5000, ind_plot=1)
    try:
        fintervals = np.load(fdir + "/fintervals1.npy")
    except FileNotFoundError:
        fintervals = del_V_S(intervals, chars)
        fintervals = del_artifacts(fintervals, intervals)
        np.save(fdir + "/fintervals.npy", fintervals)

    try:
        clean_fintervals = np.load(fdir + "/clean_fintervals1.npy")
    except FileNotFoundError:
        clean_fintervals = step_moving_average(fintervals, 4)
        clean_fintervals = step_moving_average(clean_fintervals, 6)
        clean_fintervals = step_moving_average(clean_fintervals, 4)
        clean_fintervals = moving_average(clean_fintervals, 12)
        np.save(fdir + "/clean_fintervals.npy", clean_fintervals)
    p.plot(intervals, pen=(90, 90, 210))
    # p.plot(fintervals, pen="m")
    p.plot(clean_fintervals, pen="g")

    try:
        coef_p = np.load(fdir + "/coef_p1.npy")
    except FileNotFoundError:
        inds_min = get_inds_min_diff(intervals, chars)
        mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1, mean_PR2, mean_PR3 = get_p_pos(lead1, lead2, lead3, intervals,
                                                                                    r_pos, chars, inds_min)
        plot_select_p(lead1, lead2, lead3, intervals, r_pos, mean_PR1, mean_PR2, mean_PR3, k)
        coef_p = get_P(lead1, lead2, lead3, intervals, r_pos, chars, mean_amp_p1, mean_amp_p2, mean_amp_p3, mean_PR1,
                       mean_PR2, mean_PR3)
        # p.plot(coef_p, pen='g')
        coef_p = step_moving_average(coef_p, 20)
        # p.plot(coef_p, pen='r')
        coef_p = moving_average(coef_p, 40)
        coef_p = moving_average(coef_p, 20)
        # coef_p = truncate_win2(coef_p, 0.85, 160)
        # coef_p = truncate_win2(coef_p, 0.75, 80)
        np.save(fdir + "/coef_p.npy", coef_p)
        print(f"mean_coef_p = {coef_p.mean()}")
    p.plot(coef_p, pen='y')
    # coef_p = truncate_win2(coef_p, 0.75, 80)
    # p.plot(coef_p, pen='r')
    try:
        coef_disp = np.load(fdir + "/coef_disp1.npy")
    except FileNotFoundError:
        coef_disp = get_disp_coef1(fintervals)
        # p.plot(coef_disp, pen='m')
        # coef_disp = truncate_win2(coef_disp, 0.8, 70)
        # coef_disp = step_moving_average(coef_disp, 12)
        # coef_disp = step_moving_average(coef_disp, 8)
        # coef_disp = moving_average(coef_disp, 12)
        np.save(fdir + "/coef_disp.npy", coef_disp)
    # coef_disp = truncate_win2(coef_disp, 0.85, 160)
    # coef_disp = truncate_win2(coef_disp, 0.75, 80)
    p.plot(coef_disp, pen='c')
    try:
        coef_p = np.load(fdir + "/coef_fibr1.npy")
    except FileNotFoundError:
        coef_fibr = coef_p * coef_disp * (0.5 + 100 / clean_fintervals)
        # p.plot(coef_fibr, pen='w')
        coef_fibr = truncate_win2(coef_fibr, 0.85, 160)
        coef_fibr = truncate_win2(coef_fibr, 0.75, 80)
        np.save(fdir + "/coef_fibr.npy", coef_fibr)

    try:
        start_ind_arr = np.load(fdir + "/start_ind_arr1.npy")
        stop_ind_arr = np.load(fdir + "/stop_ind_arr1.npy")
        diff_stop_start = np.load(fdir + "/diff_stop_start1.npy")
    except FileNotFoundError:
        start_ind_arr, stop_ind_arr, diff_stop_start = get_fibr_num_samples(coef_fibr, clean_fintervals, mask_pseudo_fibr)
        np.save(fdir + "/start_ind_arr.npy", start_ind_arr)
        np.save(fdir + "/stop_ind_arr.npy", stop_ind_arr)
        np.save(fdir + "/diff_stop_start.npy", diff_stop_start)

        print(f"Количество эпизодов фибрилляции: = {diff_stop_start.size}")

    norm_coef = 1.0
    print(f"norm_coef = {norm_coef:.2f}")
    coef_fibr *= norm_coef
    # p.plot(coef_fibr, pen='r', fillLevel=0.0, brush=(60, 180, 60, 140))
    p.plot(coef_fibr, pen='r')


if __name__ == "__main__":
    main()

sys.exit(app.exec())
