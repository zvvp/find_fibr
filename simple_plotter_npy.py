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
    # graph1 = filtfilt(b, a, graph1)
    fname = QFileDialog.getOpenFileName()[0]
    graph2 = np.load(fname)
    fname = QFileDialog.getOpenFileName()[0]
    graph3 = np.load(fname)

    p.plot(graph1, pen='g', name='graph1',)
    p.plot(graph2, pen='r', name='graph2',)
    p.plot(graph3, pen='c', name='graph3',)
    # p.plot(graph1 * graph2, pen='w', name='graph3',)


    sys.exit(app.exec())
