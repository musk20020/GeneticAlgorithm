import tensorflow as tf
import random
import numpy as np

class Genetic:

    def __init__(self):
        self.population = 30
        self.generation = 5
        self.survival = 15
        self.phase = 6

        ## ============= genetic setting ===============
        self.activation = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.selu, tf.nn.relu6, tf.nn.tanh]
        self.skipLayer = [True, False]

        self.outputChannal = [32, 64, 96, 128, 256]
        self.kernelSzie_init = [3, 5, 7]
        self.BN = [True, False]

        self.kernelSzie = [3, 5, 7]
        self.dilation = [(1,1), (1,2), (1,4), (1,8), (1,16), (1,32)]
        self.twoLayer = [True, False]

    def initialPopulation(self):
        activation = np.random.choice(self.activation )
        skipLayer = np.random.choice(self.skipLayer)
        outputChannal_a = np.random.choice(self.outputChannal)
        kernelSzie_a = np.random.choice(self.kernelSzie_init)
        BN_a = np.random.choice(self.BN)
        outputChannal_b = np.random.choice(self.outputChannal)
        kernelSzie_b = np.random.choice(self.kernelSzie_init)
        BN_b = np.random.choice(self.BN)
        dilation_a = (1, 1)
        dilation_b = (1, 1)
        twoLayer = True
        return [outputChannal_a, (kernelSzie_a,kernelSzie_a), BN_a, dilation_a,
                outputChannal_b, (kernelSzie_b,kernelSzie_b), BN_b, dilation_b,
                skipLayer, activation, twoLayer]

    def populationGenerator(self, activation):
        skipLayer = np.random.choice(self.skipLayer)
        outputChannal_a = np.random.choice(self.outputChannal)
        kernelSzie_a = np.random.choice(self.kernelSzie)
        BN_a = np.random.choice(self.BN)
        dilation_a = np.random.choice(self.dilation)
        outputChannal_b = np.random.choice(self.outputChannal)
        kernelSzie_b = np.random.choice(self.kernelSzie)
        BN_b = np.random.choice(self.BN )
        dilation_b = np.random.choice(self.dilation)
        twoLayer = np.random.choice(self.twoLayer)
        return [outputChannal_a, (kernelSzie_a,1), BN_a, dilation_a,
                outputChannal_b, (kernelSzie_b,1), BN_b, dilation_b,
                skipLayer, activation, twoLayer]

    # def mutation(self, genetic):
    #
    # def crossover(self, gen1, gen2):
    #
    # def geneticControl(self, genetic):

    def genModel(self, genetic):
        ## genetic is a 2D array, each row is a one/two layer model structure
        kerasModel = []
        for l in genetic:
            twoLayer = l[-1]
            if twoLayer:
                batchNorm_a = l[2]
                batchNorm_b = l[6]

                conv1 = tf.keras.layers.Conv2D(filters=l[0], kernel_size=l[1], padding='same', activation=l[9],
                                              dilation_rate=l[3])
                kerasModel.append(conv1)
                if batchNorm_a:
                    BN1 = tf.keras.layers.BatchNormalization()
                    kerasModel.append(BN1)

                conv2 = tf.keras.layers.Conv2D(filters=l[4], kernel_size=l[5], padding='same', activation=l[9],
                                              dilation_rate=l[7])
                kerasModel.append(conv2)
                if batchNorm_b:
                    BN2 = tf.keras.layers.BatchNormalization()
                    kerasModel.append(BN2)
            else:
                batchNorm_a = l[2]

                conv1 = tf.keras.layers.Conv2D(filters=l[0], kernel_size=l[1], padding='same', activation=l[9],
                                              dilation_rate=l[3])
                kerasModel.append(conv1)
                if batchNorm_a:
                    BN1 = tf.keras.layers.BatchNormalization()
                    kerasModel.append(BN1)

        if twoLayer:
            denseUnits = l[4]
        else:
            denseUnits = l[0]
        kerasModel.append(tf.keras.layers.Permute((2, 1, 3)))
        kerasModel.append(tf.keras.layers.Reshape([60, denseUnits*257]))
        kerasModel.append(tf.keras.layers.Dense(257, activation='sigmoid'))
        return kerasModel

    def modelFit(self, geneticList):
        keras1 = self.genModel(geneticList)
        model1 = tf.keras.Sequential(keras1)
        model1.compile(optimizer='adam', loss='mse')
        history = model1.fit(x, y, batch_size=32, epochs=10)
        return history.history['loss'][-1]

    def trainModel(self, geneticList, x, y):
        keras1 = self.genModel(geneticList)
        model1 = tf.keras.Sequential(keras1)
        model1.compile(optimizer='adam', loss='mse')
        hist = model1.fit(x, y, batch_size=32, epochs=3)

        for p in range(self.phase):
            for g in range(self.generation):
                for N in range(0, 3, self.population):
                    L1 = modelFit(geneticList[N])
                    L2 = modelFit(geneticList[N]+1)
                    L3 = modelFit(geneticList[N]+2)


