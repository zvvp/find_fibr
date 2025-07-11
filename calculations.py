from PyQt6.QtWidgets import QApplication
import sys
import numpy as np
import pyqtgraph as pg
# from scipy.signal import butter
from scipy.stats import pearsonr


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

x = np.arange(40, 350)
y = 150/(x + 790) * x + 25
p.plot(x, y)
sys.exit(app.exec())

# b, a = butter(3, 15, "lp", fs=250)
# print(b)
# print(a)

