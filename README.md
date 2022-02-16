# Predicting realized volatility
This ReadMe gives a quick overview of the project.
The detailed report can be found in the notebook at code/Data_Description_and_Model_Evaluation.ipynb.

## Project Idea: 
Forecast stock volatility tomorrow with stock and macroeconomic data from today and the past.

## What is financial (realized) volatility?
Realized (return) volatility is the standard deviation of (daily) returns. Realized volatility describes the risk connected to a financial asset.

## Data:
*   Source: Trade and Quote Database from New York Stock Exchange and Chicago Stock Exchange
*   Stock: BAC (Bank of America)
* Timeframe: 01/2010 to 12/2020 (11 years, 2768 trading days after removing non-business days and missing data)
* 	Additional features: open-close return (sign), VIX, Trading volume.

## Notebooks
* Data_Description_and_Model_Evaluation.ipynb - Main report notebook.
Describes the project idea, literature and data. Compares predictions of the different models (from notebooks below) and concludes.
* [data_preprocessing.ipynb](https://github.com/fogx/predicting-financial-volatility-project/blob/main/code/data_preprocessing.ipynb): data preprocessing.
* [FFN.ipynb](https://github.com/fogx/predicting-financial-volatility-project/blob/main/code/FFN.ipynb): Trains feed-forward neural network model configurations.
* [LSTM.ipynb](LSTM.ipynb): lstm network architectures to perform the volatility prediction. 
We try three different architectures 
  1. a simple one layer lstm,
  2. a two layer lstm with additional dense layers, 
  3. a model with a functional API that combines an lstm layer, appropriate for features with time lags, and dense layers which
 are appropriate for features without time lags.
* [GRU.ipynb](https://github.com/fogx/predicting-financial-volatility-project/blob/main/code/GRU.ipynb): similar to the LSTM notebook, but with GRU cells instead of LSTM cells.


## Authors:
Erik Senn, Jule Schüttler, Leszek Wächter, Leon Wolf

## Course:
Machine Learning with Tensorflow, opencampus.sh, Winter Term 2122

