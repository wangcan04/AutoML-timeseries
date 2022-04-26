# AutoML-timeseries

AutoML-timeseries is an automated model for time series forecasting, including automated window size selection and automated feature extraction. Here we show two kinds of baselines, including simple baselines(ARIMA, Moving average, Support vector machine, Random forest, gradient boosting machine, and eXtreme Gradient Boosting), deep neural network baselines (Long short-term memory, Gated recurrent unit, and N-BEATS), AutoML baselines (Auto-Keras, and auto-sklearn), and our AutoML variants for time series forecasting. 

## Requirements
some python packages are needed:

auto-sklearn  0.8.0 

tsfresh  0.16.1 

autokeras  1.0.12

pmdarima  1.8.0 


## Simple baselines
Simple stastical models and machine learning models are used.

ARIMA: 
* arima.py

Moving average: 
* mv1.py 
* mv10.py 
* mv200.py

SVM: 
* svm10.py 
* svm200.py

RF:
 * rf10.py 
 * rf200.py

GBM: 
* gbm10.py 
* gbm200.py

XGBoost: 
* xgbboost10.py 
* xgboost200.py

Random seed for SVM, RF, GBM experiments is all 10.
Random seed for XGBoost is 1.

## Deep neural network baselines

LSTM: 
* lstm10.py 
* lstm200.py

GRU: 
* gru10.py 
* gru200.py

N-BEATS: 
* n-beats10.py 
* n-beats200.py

## AutoML baselines
Auto-Keras: 
* autokeras10.py 
* autokeras200.py

The number of models that can be evaluated with 2 hours time budget for every dataset is in autokerasmodelcount.txt

Auto-sklearn: 
* autosklearn10.py 
* autosklearn200.py 


## AutoML variants for time series forecasting:
Auto-sklearn with automated window size selection (W):
* autosklearn_w_ens.py
* autosklearn_w_single.py

Auto-sklearn with tsfresh features (T):
* autosklearn_t10.py
* autosklearn_t200.py

Auto-sklearn with automated window size selection and tsfresh features (WT):
* autosklearn_wt.py

Random seeds for autosklearn experiments are 1 to 25.

## Citation
```
@article{AutoML-timeseries,
  author = {Can Wang, Mitra Baratchi, Thomas Bäck, Holger H. Hoos, Steffen Limmer, Markus Olhofer},
  title = {Towards time-series-specific feature engineering in automated machine learning frameworks},
  year = {2022},
  journal = {under review},
}
```
