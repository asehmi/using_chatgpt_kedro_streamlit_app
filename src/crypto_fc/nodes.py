from typing import List
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .constants import (
    TRAIN_INDEX,
    TEST_INDEX,
)

# COMMENT:
# ChatGPT doesn't suggest the argument to train_model() should be the training data split of the source data set.
# My "target" column is the close price. You'd need to supply non-"target" columns. i.e. feature columns, in your dataset.

def get_symbol(data) -> str:
    symbol = data['symbol'].unique()[0].upper()
    return symbol

def train_model(data):
	# Split the data into features and targets
	X = data.drop(['symbol', 'open', 'close', 'low', 'high', 'close_t1', 'series_id'], axis=1)
	y = data['close_t1']
	# Create a LinearRegression model and fit it to the data
	model = LinearRegression()
	model.fit(X, y)
	return model

def evaluate_model(model, data):
	X = data.drop(['symbol', 'open', 'close', 'low', 'high', 'close_t1', 'series_id'], axis=1)
	y = data['close_t1']
	# Make predictions using the model and calculate the mean squared error
	y_pred = model.predict(X)
	mse = mean_squared_error(y, y_pred)
	return y, y_pred, mse


def plot_metric(symbol, y, y_pred, mse):
	result_df = pd.DataFrame(y)
	result_df['close_pred_t1'] = y_pred
	result_df['close_pred_t1'] = result_df['close_pred_t1'].shift(-1)
	result_df.dropna(inplace=True)
 
	print(f'Saving {symbol} predictions for display in Streamlit...')	
	result_df.to_csv(f'./data/{symbol.lower()}_predictions.csv', index=False, encoding='utf-8')

	print('\n\n', '🤒 Mean Square Error (MSE)', f'{round(mse * 100, 3)}%', '\n\n')
	print(result_df)

def train_test_split_2(data: pd.DataFrame) -> List[pd.DataFrame]:
    df_features = data.copy()
    df_features.drop('Timestamp', inplace=True)
    return np.array_split(df_features, 2)

def train_test_split(data: pd.DataFrame, train_index=TRAIN_INDEX, test_index=TEST_INDEX) -> List[pd.DataFrame]:
    df_features = data.copy()
    df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], dayfirst=True)
    df_features.set_index('Timestamp', inplace=True)
    return [df_features[train_index], df_features[test_index]]