from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from functions import moving_average,truncate_win2
from scipy.signal import butter, filtfilt


def get_median_graph(graph1, graph2, graph3):
    out = np.array([])
    len1 = len(graph1)
    len2 = len(graph2)
    len3 = len(graph3)
    len_g = min(len1, len2, len3)
    for i in range(len_g):
        out = np.append(out, np.median([graph1[i], graph2[i], graph3[i]]))
    return out

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

    # med_graph = get_median_graph(graph1, graph2, graph3)
    p.plot(graph1, pen='g', name='graph1',)
    p.plot(graph2, pen='y', name='graph2',)
    p.plot(graph3, pen='r', name='graph3',)
    # p.plot(med_graph, pen='r',name='med_graph')
    # p.plot(graph1 * graph2, pen='w', name='graph3',)


    sys.exit(app.exec())
