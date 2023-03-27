# python_ml
# A repository for machine learning in python that covers topics in supervised learning and time series using sklearn and Keras. 
### This repository covers topics in fundamental statistical modeling using standard modules. 
For data preprocessing, the repo "demo" contains scripts illustrating how to conduct sampling and hypothesis testing. Also, it provides simulations of LLN and CLT. 
For supervised learning, a few short scripts demonstrate how to load, extract, and transform datasets for later modeling. And also I have written several programs on benchmark models with feature engineering, model estimation, and cross-validation. 
Time series models are paired with a rolling window estimation framework, by which the models can be evaluated out-of-sample. A customized ensemble framework is available. 

Regarding deep learning, I have included several Keras scripts to study the time series model using commodity futures prices. Further experimentation might consist of more detailed Keras functional API model development/TensorFlow and large-scale model development.
Modules used

1) Numpy
2) Sklearn
3) Statsmodels
4) Keras(Tensorflow Backend)



```bash
├── README.md
├── demo
│   ├── LLN_CLT_demo.py  Law of Large Numbers and Central Limit Theorem 
│   ├── contour.png
│   ├── date_format_demo.py parse and format Date using datatime module
│   ├── feature_scaling_demo.ipynb  min_max/standardizer/normalizer - feature scaling
│   ├── iterator_generator_demo.ipynb 
│   ├── kernel_density.py use a Gaussian kernel to estimate density
│   ├── kernel_density_figure.png
│   ├── onehot_encoding_demo.ipynb  use one-hot encoding to handle categorical features
│   ├── stat_test.ipynb random sampling/bootstrap and hypothesis testing
│   └── str_format_demo.py  Python f-string format (since Python 3.6)
├── ensemble
│   └── ensemble_voting_clf2.py use a voting classifier to collect results from SVM/Decision Tree/KNN
├── keras misc
│   ├── keras_CLF_multinomial.py  multinomial classification using a keras sequential model 
│   ├── keras_LSTM.py keras Long short-Term Memory model
│   ├── keras_MNIST_CLF.py 
│   ├── keras_MNIST_CLF_tuning.py Tuning example on MINST dataset
│   ├── keras_func_api.py Kera\'s functional API demo
│   ├── keras_seq_demo.py
│   └── sklearn_mlp_demo.py Multi-Layer Perceptron Model from Sklearn
├── supervised learning
│   ├── KNN.ipynb K Nearest Neighbors model 
│   ├── LinearModel_CrossValidation.py  Regularized linear model with cross-validation
│   ├── Logit_demo.ipynb Logistic regression
│   ├── OLS_Ridge_demo.ipynb L2-regularized Linear model
│   ├── lasso.png 
│   ├── lasso.py  L1-regularized linear model
│   └── mse and timing.png
├── \time series
│   ├── ARIMA_RollingWindow.py Auto Regressive Integrated Moving Average model with rolling window estimation
│   ├── GARCH_Gridsearch_Estimation.py Generalized AutoRegressive Conditional Heteroskedasticity using a grid search tuning strategy 
│   └── VAR_RollingWindow.py Vector Auto Regressive 
├── \time series LSTM
│   ├── HistoricalQuotes SoybeanNasdaq.csv
│   ├── HistoricalQuotesCornNasdaq.csv
│   ├── LSTM_CT_backtest.png
│   ├── LSTM_SP500_backtest.png
│   ├── LSTM_ZW_backtest.png
│   ├── MNIST_CLF.py
│   ├── MNIST_CLF_tune.py
│   ├── VAR.py
│   ├── backtest.py A Trading strategy implementation that tracks Daily Profit and Loss 
│   ├── futures_price_forecast_lstm.py An LSTM model that forecasts futures prices 
│   ├── gridsearch_LSTM.py  grid search script - LSTM model hyperparameter tuning
│   └── rolling_LSTM.py Model evaluation(Accuracy + Profit and Loss)

```
### Author Yang Song 
