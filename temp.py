from PyQt6.QtWidgets import QFileDialog, QApplication 
import sys


def get_time_qrs(addr, fname):  
    with open(fname, "rb") as f:
        f.seek(151)
        dlmt = f.read(1)
        if dlmt == b":":
            f.seek(150)
            start_h = int(f.read(1))
            f.seek(152)
            start_m = int(f.read(2))
            f.seek(155)
            start_s = int(f.read(2))
        else:
            f.seek(150)
            start_h = int(f.read(2))
            f.seek(153)
            start_m = int(f.read(2))
            f.seek(156)
            start_s = int(f.read(2))
            # start_h = int(head[150:152])
            # start_m = int(head[153:155])
            # start_s = int(head[156:158])
    s = addr * 4 // 1000
    m = s // 60
    s = s % 60
    s = s + start_s
    if s >= 60:
        s = s - 60
        m = m + 1
    h = m // 60
    m = m % 60
    m = m + start_m
    if m >= 60:
        m = m - 60
        h = h + 1
    d = h // 24
    h = h % 24
    h = h + start_h
    if h >= 24:
        h = h - 24
        d = d + 1

    print(f" {d + 1} день {h}:{m}:{s}")

if __name__ == "__main__":

    app = QApplication(sys.argv)

    fname = QFileDialog.getOpenFileName()[0]

    get_time_qrs(1490, fname)

    sys.exit()