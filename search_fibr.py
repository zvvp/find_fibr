from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg
from functions import get_r_pos, get_start_shift, clean_ch
from scipy.signal import butter, filtfilt

color_back = (40, 40, 40, 250)
pg.setConfigOption('background', color_back)

def main():
    """
    Основная функция для запуска приложения PyQt6.
    Эта функция загружает данные ЭКГ из файлов numpy, обрабатывает их и отображает на графике с помощью pyqtgraph.
    """
    app = QApplication(sys.argv)

    p = pg.plot()
    color_line = (60, 250, 180, 250)
    p.showGrid(x=True, y=True)

    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")

    r_pos = get_r_pos()
    start = r_pos[29079-13]
    stop = r_pos[29080-13]
    fragment1 = lead1[start-120:stop+120]
    fragment2 = lead2[start-120:stop+120]
    fragment3 = lead3[start-120:stop+120]
    fragment1 = clean_ch(fragment1)
    fragment2 = clean_ch(fragment2)
    fragment3 = clean_ch(fragment3)
    fragment1 = fragment1[120:-120]
    fragment2 = fragment2[120:-120]
    fragment3 = fragment3[120:-120]

    p.plot(fragment1, pen=color_line)
    p.plot(fragment2-1, pen=color_line)
    p.plot(fragment3-2, pen=color_line)

    fragment1[:30] = 0
    fragment1[-17:] = 0
    fragment1[fragment1 < 0] = 0.0
    fragment2[:30] = 0
    fragment2[-17:] = 0
    fragment2[fragment2 < 0] = 0.0
    fragment3[:30] = 0
    fragment3[-17:] = 0
    fragment3[fragment3 < 0] = 0.0

    # p.plot(fragment1, pen=color_line)
    # p.plot(fragment2-1, pen=color_line)
    # p.plot(fragment3-2, pen=color_line)

    b, a = butter(1, 4.3, 'lp', fs=250) # 1, 4, 'lp', fs=250
    fragment1 = filtfilt(b, a, fragment1**2)
    fragment2 = filtfilt(b, a, fragment2**2)
    fragment3 = filtfilt(b, a, fragment3**2)
    # fragment = filtfilt(b, a, fragment1 + fragment2 + fragment3)

    p.plot(fragment1*20, pen='r')
    p.plot(fragment2*20-1, pen='r')
    p.plot(fragment3*20-2, pen='r')
    p.plot((fragment1+fragment2+fragment3)*20*3-2, pen='y')
    # p.plot(fragment*20*3-2, pen='y')


    # n = get_start_shift(20, 5, 44)
    # print(n)
    # t = 1_000_000
    #
    # p.plot(lead1[n:n + t], pen=color_line)
    # p.plot(lead2[n:n + t] - 3, pen=color_line)
    # p.plot(lead3[n:n + t] - 6, pen=color_line)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
