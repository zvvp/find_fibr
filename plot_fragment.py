from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg


app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
print(fdir)
lead1 = np.load(fdir + "/clean_lead1.npy")
lead2 = np.load(fdir + "/clean_lead2.npy")
lead3 = np.load(fdir + "/clean_lead3.npy")

try:
    r_pos = np.load(fdir + "/r_pos.npy")
    k = 61000
    p.plot(lead1[r_pos[k]-2000:r_pos[k]+2000])
    p.plot(lead2[r_pos[k]-2000:r_pos[k]+2000]-2)
    p.plot(lead3[r_pos[k]-2000:r_pos[k]+2000]-4)
except:
    pass

sys.exit(app.exec())
