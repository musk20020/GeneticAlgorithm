from model import Genetic
import numpy as np
import os

def main():
    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    # x = np.load('/AudioProject/trainingData/noisy2.npy')
    # x_tmp = np.reshape(np.transpose(x, [0, 2, 1, 3]), [-1, 60, 257])
    # y_tmp = np.load('/AudioProject/trainingData/clean2.npy')
    # y = y_tmp/x_tmp
    # # y[y>1] = 1
    # del x_tmp, y_tmp

    fileNum = 15
    noisy0 = np.load('/AudioProject/trainingDataNorm/regular/noisy0.npy')
    noisyNorm0 = np.load('/AudioProject/trainingDataNorm/regular/noisy_norm0.npy')
    clean0 = np.load('/AudioProject/trainingDataNorm/regular/clean0.npy')
    for i in range(1, fileNum):
        noisy = np.load('/AudioProject/trainingDataNorm/regular/noisy{}.npy'.format(i))
        noisyNorm = np.load('/AudioProject/trainingDataNorm/regular/noisy_norm{}.npy'.format(i))
        clean = np.load('/AudioProject/trainingDataNorm/regular/clean{}.npy'.format(i))

        noisy0 = np.concatenate((noisy0, noisy), axis=0)
        noisyNorm0 = np.concatenate((noisyNorm0, noisyNorm), axis=0)
        clean0 = np.concatenate((clean0, clean), axis=0)

    targetMask = clean0/np.transpose(noisy0[:,:,:,0], [0,2,1])

    noisy_dev = np.load('/AudioProject/trainingDataNorm/regular/noisy20.npy')
    noisyNorm_dev = np.load('/AudioProject/trainingDataNorm/regular/noisy_norm20.npy')
    clean_dev = np.load('/AudioProject/trainingDataNorm/regular/clean20.npy')
    targetMask_dev = clean_dev / np.transpose(noisy_dev[:, :, :, 0], [0, 2, 1])

    del noisy0, clean0, noisy, noisyNorm, clean, clean_dev, noisy_dev



    G = Genetic()
    G.trainModel(noisyNorm0, targetMask, noisyNorm_dev, targetMask_dev)

if __name__ == '__main__':
    main()