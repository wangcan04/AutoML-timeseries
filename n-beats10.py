import warnings
import tensorflow as tf

import numpy as np

from nbeats_keras.model import NBeatsNet as NBeatsKeras
#from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
import numpy as np
import os
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import sklearn
from tqdm import tqdm
import sklearn.model_selection
import sklearn.metrics
warnings.filterwarnings(action='ignore', message='Setting attributes')
import math

def main():
    # https://keras.io/layers/recurrent/
    # At the moment only Keras supports input_dim > 1. In the original paper, input_dim=1.
    data=np.loadtxt('timeseries(not real)/Autoregressive with noise/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Correlated noise/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Diffusionless Lorenz Attractor/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Driven pendulum with dissipation/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Driven van der Pol oscillator/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Duffing two-well oscillator/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Duffing-van der Pol Oscillator/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Moving average process/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Nonstationary autoregressive/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(not real)/Random walk/comp-engine-export-datapoints.txt',delimiter=',')

    #data=np.loadtxt('timeseries(real)/Crude oil prices/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/ECG/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Exchange rate/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Gas prices/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/human speech/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Macroeconomics/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Micoeconomics/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/music/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Tropical forest soundscape(animal sound)/comp-engine-export-datapoints.txt',delimiter=',')
    #data=np.loadtxt('timeseries(real)/Zooplankton growth/comp-engine-export-datapoints.txt',delimiter=',')
    time_steps, input_dim, output_dim = 10, 1, 1
    i=int(sys.argv[1])+1
    seed = i
    tf.random.set_seed(i)


    # This example is for both Keras and Pytorch. In practice, choose the one you prefer.
    for BackendType in [NBeatsKeras]:
        backend = BackendType(
            backcast_length=time_steps, forecast_length=output_dim,
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),

        )

        # Definition of the objective function and the optimizer.
        backend.compile(loss='mse', optimizer='adam')

        # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
        # where f = np.mean.
        window_size=time_steps
        windowsize = 200
        y = data
        y = y[windowsize:]
        X = []
        for i in range(len(data)-windowsize):
            x = data[i:i+windowsize]
            X.append(x)
        X = np.array(X)

        X_trainlist = X[:int(len(X)*0.67),-window_size:]
        y_trainlist = y[:int(len(X)*0.67)]

        X_train = X_trainlist[:int(len(X)*0.67)]
        y_train = y_trainlist[:int(len(X)*0.67)]
        X_valid = X_trainlist[int(len(X)*0.67):]
        y_valid = y_trainlist[int(len(X)*0.67):]

        X_test = X[int(len(X)*0.67):,-window_size:]
        y_test = y[int(len(X)*0.67):]
        test_size = len(X_test)
        print (y_test.shape)
        # Train the model.
        print('Training...')
        backend.fit(X_train, y_train, validation_data=(X_valid, y_valid),epochs=500)
        print ('training done')
        predictions = backend.predict(X_train)
        predictions =  predictions.reshape(-1,1)
        score = sklearn.metrics.mean_squared_error(y_train, predictions)
        MAE = sklearn.metrics.mean_absolute_error(y_train, predictions)
        RMSEscore = math.sqrt(score)
        R2 = sklearn.metrics.r2_score(y_train, predictions)
        print("ERRORS TRAINING SET:")
        print ('RMSE: ', str(RMSEscore))
        print ('MAE: ', str(MAE))
        print('R2:', str(R2))

        # Predict on the testing set (forecast).
        predictions = backend.predict(X_test)


        predictions =  predictions.reshape(-1,1)
        score = sklearn.metrics.mean_squared_error(y_test, predictions)
        MAE = sklearn.metrics.mean_absolute_error(y_test, predictions)
        RMSEscore = math.sqrt(score)
        R2 = sklearn.metrics.r2_score(y_test, predictions)
        print("ERRORS TEST SET:")
        print ('RMSE: ', str(RMSEscore))
        print ('MAE: ', str(MAE))
        print('R2:', str(R2))
if __name__ == '__main__':
    main()
