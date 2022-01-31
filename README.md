# Predicting realized volatility

## description:
TODO
replace this text with description (short description, not more than 200 characters)

LSTM notebook: 
Our goal is to use lstm network architectures to perform the volatility prediction. 
We try three different architecture 1.) a simple one layer lstm, 2.) a two layer lstm with additioanl dense layers, 
3.) a model with a funcitonal API that combines a lstm layer, appropriate for features with time lags, and a dense layers which
are appropriate for features without time lags.
We check the performance based on the train data and create prediction results for the validation and test set.


### authors:
Erik Senn, Jule Schüttler, Leszek Wächter, Leon Wolf

### course:
Machine Learning with Tensorflow

### semester:
WiSe2122

### data:
TODO
[Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
