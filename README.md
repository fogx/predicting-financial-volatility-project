# Predicting realized volatility
The readme gives a quick overview of the project.
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
* FFN.ipynb: SHORT DESCRIPTION MISSING
* LSTM.ipynb: LSTM notebook
The goal is to use lstm network architectures to perform the volatility prediction. 
We try three different architecture 1.) a simple one layer lstm, 2.) a two layer lstm with additioanl dense layers, 
3.) a model with a funcitonal API that combines a lstm layer, appropriate for features with time lags, and a dense layers which
are appropriate for features without time lags.
We check the performance based on the train data and create prediction results for the validation and test set.
* * GRU.ipynb: as LSTM notebook with GRU cells instead of LSTM cells.

## Authors:
Erik Senn, Jule Schüttler, Leszek Wächter, Leon Wolf

## Course:
Machine Learning with Tensorflow, opencampus.sh, Winter Term 2122

