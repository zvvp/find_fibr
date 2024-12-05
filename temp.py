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

x = np.arange(0, 100) *3
y = x**2 / 100
p.plot(x, y, pen='g')

# x1 = np.arange(0, 300)
y1 = 1000 / x**0.5
p.plot(x, y1, pen='y')
p.plot(x, y*y1/10, pen='r')

sys.exit(app.exec())