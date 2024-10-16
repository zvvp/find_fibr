import numpy as np
from functions import get_S


def main():
    lead1 = np.load("d:/Kp_01/clean_lead1.npy")
    lead2 = np.load("d:/Kp_01/clean_lead2.npy")
    lead3 = np.load("d:/Kp_01/clean_lead3.npy")

    get_S(lead1, lead2, lead3)

if __name__ == "__main__":
    main()
