from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
from functions import parse_B1_txt


app = QApplication(sys.argv)

fdir = QFileDialog.getExistingDirectory(parent=None, directory="C:/EcgVar")
print(fdir)
r_pos, intervals, chars, forms = parse_B1_txt(fdir)
mask_pseudo_fibr = np.ones(len(intervals))
np.save(fdir + '/mask_pseudo_fibr.npy', mask_pseudo_fibr)
sys.exit()
# sys.exit(app.exec())