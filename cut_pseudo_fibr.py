from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np


app = QApplication(sys.argv)

fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
print(fdir)
try:
    mask_pseudo_fibr = np.load(fdir + '/mask_pseudo_fibr.npy')
    intervals = np.load(fdir + '/intervals.npy')
except FileNotFoundError:
    mask_pseudo_fibr = np.ones(intervals.size)
mask_pseudo_fibr[:] = 1
# mask_pseudo_fibr[:30000] = 0
# mask_pseudo_fibr[100000:] = 0
# mask_pseudo_fibr[29300:74000] = 0 # борщ
# mask_pseudo_fibr[100035:] = 0      # борщ
np.save(fdir + '/mask_pseudo_fibr.npy', mask_pseudo_fibr)

sys.exit()