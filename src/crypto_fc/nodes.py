import os
import psutil
from typing import List
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# These nodes can be run from Kedro's CLI, so don't import interactive GUI stuff
ppid = os.getppid() # Get parent process id
print(psutil.Process(ppid).name().lower())
IS_STREAMLIT = (psutil.Process(ppid).name().lower() == 'streamlit.exe')
# This will work for Streamlit (interactive mode), but not from Kedro's CLI
print("🎈 Running in Streamlit") if IS_STREAMLIT else print("🥁 Running from Kedro's CLI")
if IS_STREAMLIT:
	import plotly.express as px
	import streamlit as st
	import st_functions
	state = st.session_state

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
	X = data.drop(['symbol', 'close_t1', 'series_id'], axis=1)
	y = data['close_t1']
	# Create a LinearRegression model and fit it to the data
	model = LinearRegression()
	model.fit(X, y)
	return model

def evaluate_model(model, data):
	X = data.drop(['symbol', 'close_t1', 'series_id'], axis=1)
	y = data['close_t1']
	# Make predictions using the model and calculate the mean squared error
	y_pred = model.predict(X)
	mse = mean_squared_error(y, y_pred)
	return y, y_pred, mse


def plot_metric(symbol, y, y_pred, mse, show_pipeline=True):
	result_df = pd.DataFrame(y)
	result_df['close_pred_t1'] = y_pred
	result_df['close_pred_t1'] = result_df['close_pred_t1'].shift(-1)

	if not IS_STREAMLIT:
		print('\n\n', '🤒 Mean Square Error (MSE)', f'{round(mse * 100, 3)}%', '\n\n')
		print(result_df)
		return

	st.markdown(f'### PREDICTIONS for {symbol}')
	st.write('')
	c1, c2, _ = st.columns([1,1,3])
	with c1:
		st.markdown('##### 🤒 Mean Square Error (MSE)')
		st.metric('Mean Square Error (MSE)', f'{round(mse * 100, 3)}%' , f'{round((0.05 - mse) * 100, 3)}%', label_visibility='collapsed')
	with c2:
		if show_pipeline:
			# Launch button will only work locally
			if not st.secrets['IS_ST_CLOUD']:
				st.markdown('##### ⚙️ Pipeline visualization')
				st_functions.st_button('kedro', 'http://127.0.0.1:4141/', 'Launch Kedro-Viz', 40)
			else:
				st.markdown('##### ⚙️ Pipeline specification')
				st.caption('_Please [clone the app](https://github.com/asehmi/using_chatgpt_kedro_streamlit_app) and run it locally to get an interactive pipeline visualization._')
    
			if st.checkbox('Show specification', False):
				with open(f'./data/{symbol.lower()}_pipeline.json', 'rt', encoding='utf-8') as fp:
					pipeline_json = fp.read()
				st.json(pipeline_json, expanded=True)
  
	if state['show_table']:
		st.markdown('---')
		st.subheader('Data')
		st.write(result_df)

	st.markdown('---')
	st.subheader('Chart')
	fig = px.line(
		result_df,
		x=result_df.index, y=['close_t1', 'close_pred_t1'],
		labels={result_df.index.name: 'T', 'close_t1': f'{symbol} Price ($)', 'close_pred_t1': f'{symbol} Price Prediction ($)'},
		title=f'Price Prediction: {symbol}',
		width=1200, height=800,
		**state['chart_kwargs']
	)
	st.plotly_chart(fig, theme=state['chart_theme'])

def train_test_split_2(data: pd.DataFrame) -> List[pd.DataFrame]:
    df_features = data.copy()
    df_features.drop('Timestamp', inplace=True)
    return np.array_split(df_features, 2)

def train_test_split(data: pd.DataFrame, train_index=TRAIN_INDEX, test_index=TEST_INDEX) -> List[pd.DataFrame]:
    df_features = data.copy()
    df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], dayfirst=True)
    df_features.set_index('Timestamp', inplace=True)
    return [df_features[train_index], df_features[test_index]]