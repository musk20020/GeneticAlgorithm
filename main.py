from model import Genetic
import numpy as np

def main():
    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    x = np.load('/AudioProject/trainingData/noisy2.npy')
    x_tmp = np.reshape(np.transpose(x, [0, 2, 1, 3]), [-1, 60, 257])
    y_tmp = np.load('/AudioProject/trainingData/clean2.npy')
    y = y_tmp/x_tmp
    y[y>1] = 1
    del x_tmp, y_tmp

    G = Genetic()
    G.trainModel(x, y)

if __name__ == '__main__':
    main()