from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

f_name = QFileDialog.getOpenFileName()[0]
print(f_name)
ecg = np.fromfile(f_name, dtype=np.int16, offset=1024, count=900_000)

ch1 = (-ecg[::3] + 1024) / 100
ch2 = (-ecg[1::3] + 1024) / 100
ch3 = (-ecg[2::3] + 1024) / 100
ch1 = ch1 - np.mean(ch1)
ch2 = ch2 - np.mean(ch2)
ch3 = ch3 - np.mean(ch3)
ch = ch3.copy()
n = 5000
x = np.arange(n)
nous50 = np.ones(n)*np.sin(2*np.pi*50*x/250)
ch[:n] += nous50 * 0.2
def filt50(ch):
    out = np.zeros(ch.size)
    for i in np.arange(ch.size-2):
        out[i] = 0.1382*ch[i - 2] + 0.3618*ch[i - 1] + 0.3618*ch[i + 1] + 0.1382*ch[i + 2]
    return out

p.plot(ch, pen='m')

fch = filt50(ch)


p.plot(fch, pen='y')

sys.exit(app.exec())