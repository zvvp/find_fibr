from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import pyqtgraph as pg
# from scipy.signal import butter
from scipy.stats import pearsonr


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

x = np.arange(100, 250)
y = 0.5 + 100 / x
p.plot(x, y)
sys.exit(app.exec())

# b, a = butter(3, 15, "lp", fs=250)
# print(b)
# print(a)

