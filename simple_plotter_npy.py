from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from functions import moving_average,truncate_win2
from scipy.signal import butter, filtfilt


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # b, a = butter(3, 12.0, 'lp', fs=250)  # 2, 13.0, 'lp', fs=250
    # print(f"b = {b}\na = {a}")


    p = pg.plot()
    p.showGrid(x=True, y=True)

    fname = QFileDialog.getOpenFileName()[0]
    graph1 = np.load(fname)
    fname = QFileDialog.getOpenFileName()[0]
    graph2 = np.load(fname)
    fname = QFileDialog.getOpenFileName()[0]
    graph3 = np.load(fname)
    # fname = QFileDialog.getOpenFileName()[0]
    # graph4 = np.load(fname)

    p.plot(graph1, pen='g')
    p.plot(graph2, pen='r')
    p.plot(graph3, pen='c')
    # p.plot(graph4, pen='w')

    sys.exit(app.exec())
