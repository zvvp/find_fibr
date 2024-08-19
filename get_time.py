from PyQt6.QtWidgets import QFileDialog, QApplication 
import sys


def get_time_qrs(addr, fname):  
    with open(fname, "rb") as f:
        head = f.read(1024)
        if head[151] == ":":
            start_h = int(head[150])
            start_m = int(head[152:154])
            start_s = int(head[155:157])
        else:
            start_h = int(head[150:152])
            start_m = int(head[153:155])
            start_s = int(head[156:158])
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

    print(f"{d} день {h}:{m}:{s}")
    # return start_h, start_m, start_s

if __name__ == "__main__":

    app = QApplication(sys.argv)

    fname = QFileDialog.getOpenFileName()[0]

    get_time_qrs(5310980, fname)

    sys.exit()

