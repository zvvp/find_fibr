from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np


app = QApplication(sys.argv)

fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
print(fdir)

mask_pseudo_fibr = np.load(fdir + '/mask_pseudo_fibr.npy')
mask_pseudo_fibr[82560:] = 0

np.save(fdir + '/mask_pseudo_fibr.npy', mask_pseudo_fibr)

sys.exit()