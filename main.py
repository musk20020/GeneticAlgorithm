from model import Genetic
import numpy as np

def main():
    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    x = np.load('/Users/musk/dataset/trainingData/noisy0.npy')
    x_tmp = np.reshape(np.transpose(x, [0, 2, 1, 3]), [-1, 60, 257])
    y_tmp = np.load('/Users/musk/dataset/trainingData/clean0.npy')
    y = y_tmp/x_tmp

    G = Genetic()
    genetic = G.initialPopulation()

    G.trainModel([genetic], x[:64], y[:64])
    breakpoints = 0


if __name__ == '__main__':
    main()