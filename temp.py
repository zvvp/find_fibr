from PyQt6.QtWidgets import QApplication
import sys
import pyqtgraph as pg
import numpy as np


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

x = np.arange(100, 300)
y = (1 + 10000 / x**2)
p.plot(x, y)

sys.exit(app.exec())