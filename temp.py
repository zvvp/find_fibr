from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import pyqtgraph as pg
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)
# p1 = pg.plot()
# p1.showGrid(x=True, y=True)

x = np.arange(0, 100)
x = x
# p.plot(x, x**2, pen='g')

x1 = np.arange(50, 350)
x1 = x1
p.plot(x1, (x1 * 1)**0.5, pen='y')

sys.exit(app.exec())