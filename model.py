import tensorflow as tf
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Genetic:

    def __init__(self):
        self.population = 6
        self.survival = 3
        self.generation = 5
        self.phase = 6
        self.survivalRate = 0.2
        self.mutateRate = 0.2
        self.epoch = 10
        self.batchSize = 64

        ## ============= genetic setting ===============
        self.activation = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.selu, tf.nn.relu6, tf.nn.tanh]
        self.skipLayer = [True, False]

        self.outputChannal = [8, 16, 32, 64, 96, 128]
        self.kernelSzie_init = [3, 5, 7]
        self.BN = [True, False]

        self.kernelSzie = [3, 5, 7]
        self.dilation = [1, 2, 4, 8, 16, 32]
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
        dilation_a = [1, 1]
        dilation_b = [1, 1]
        twoLayer = True
        return [outputChannal_a, [kernelSzie_a,kernelSzie_a], BN_a, dilation_a,
                outputChannal_b, [kernelSzie_b,kernelSzie_b], BN_b, dilation_b,
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
        return [outputChannal_a, [kernelSzie_a,1], BN_a, [dilation_a,1],
                outputChannal_b, [kernelSzie_b,1], BN_b, [dilation_b,1],
                skipLayer, activation, twoLayer]

    def mutation(self, genetic, phase):
        layerN = len(genetic)
        layer = np.random.randint(0, len(genetic))
        param = np.random.randint(0, len(genetic[layer]))

        if param == 0 or param == 4:
            genetic[layer][param] = np.random.choice(self.outputChannal)
        elif param == 1 or param == 5:
            kernelSize = np.random.choice(self.kernelSzie)
            if phase:
                genetic[layer][param] = [kernelSize, 1]
            else:
                genetic[layer][param] = [kernelSize, kernelSize]
        elif param == 2 or param == 6:
            genetic[layer][param] = not genetic[layer][param]
        elif param == 3 or param == 7:
            genetic[layer][param] = [np.random.choice(self.dilation), 1]
        elif param == 8:
            genetic[layer][param] = not genetic[layer][param]
        elif param == 9:
            for l in range(layerN):
                genetic[l][param] = np.random.choice(self.activation)
        elif param == 10:
            genetic[layer][param] = not genetic[layer][param]
        return genetic

    def crossover(self, gen1, gen2):
        child = gen1.copy()
        for l in range(len(gen1)):
            for genN in range(len(gen1[0])):
                p1 = np.random.choice([0,1])
                if p1:
                    child[l][genN] = gen1[l][genN]
                else:
                    child[l][genN] = gen2[l][genN]
        return child

    def geneticControl(self, geneticList, phase):
        survival = len(geneticList)
        for genN in range(survival):
            p = np.random.random()
            # p = 0.1
            if p<self.mutateRate:
                geneticList[genN] = self.mutation(geneticList[genN], phase)

        restN = self.population-survival
        for i in range(restN):
            gen1N = np.random.randint(0,len(geneticList))
            gen2N = np.random.randint(0,len(geneticList))
            child = self.crossover(geneticList[gen1N], geneticList[gen2N])
            geneticList.append(child)
        return geneticList

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

    def modelFit(self, genetic, x, y):
        keras1 = self.genModel(genetic)
        model1 = tf.keras.Sequential(keras1)
        model1.compile(optimizer=tf.optimizers.Adam(), loss='mse')
        history = model1.fit(x, y, batch_size=self.batchSize, epochs=self.epoch)
        loss = history.history['loss'][-1]
        del model1, history, keras1
        return loss

    def trainModel(self, x, y):
        # keras1 = self.genModel(geneticList)
        # model1 = tf.keras.Sequential(keras1)
        # model1.compile(optimizer='adam', loss='mse')
        # hist = model1.fit(x, y, batch_size=32, epochs=3)

        geneticList = []
        for population in range(self.population):
            geneticList.append([self.initialPopulation()])
        for phase in range(self.phase):
            print('==================  phase{} start  =================='.format(phase))
            for g in range(self.generation):
                print('==================  generation{} start  =================='.format(g))

                lossList = np.zeros([self.population])
                for N in range(0, self.population, 3):
                    L1 = self.modelFit(geneticList[N], x, y)
                    print('==========================================================')
                    L2 = self.modelFit(geneticList[N+1], x, y)
                    print('==========================================================')
                    L3 = self.modelFit(geneticList[N+2], x, y)
                    print('==========================================================')
                    lossList[N] = L1
                    lossList[N+1] = L2
                    lossList[N+2] = L3

                sortLoss = np.sort(lossList)
                T = sortLoss[int(self.population/2)]
                for i in range(self.population-1, -1, -1):
                    p = np.random.random()
                    if lossList[i]>T:
                        if p>self.survivalRate:
                            geneticList.pop(i)

                geneticList = self.geneticControl(geneticList, phase) # mutate & generate child

            for population in range(self.population):
                activation = geneticList[population][0][9]
                newGen = self.populationGenerator(activation)
                geneticList[population].append(newGen)
