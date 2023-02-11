import time
import pandas as pd
import numpy as np
import datetime as dt
from PIL import Image
import requests

from sklearn.metrics import mean_squared_error

import streamlit as st
import streamlit.components.v1 as components

from crypto_fc.constants import (
    SYMBOL_DEFAULT,
    OCLH_PERIOD,
    TRAIN_INDEX,
    TEST_INDEX,
    SPLIT_DATE,
    FORECAST_HORIZON
)

from crypto_fc.data import MyDataCatalog
from crypto_fc.nodes import train_model, evaluate_model, plot_metric
from crypto_fc.pipeline import create_pipeline, run_pipeline 

# https://plotly.com/python/plotly-express/#gallery
# https://plotly.com/python/creating-and-updating-figures/
# https://plotly.com/python/templates/
import plotly.graph_objects as go
import plotly.express as px
px_templates = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'presentation', 'none']

st.set_page_config(page_title="Kedro Streamlit App!", page_icon='🤑', layout='wide')

import st_functions
st_functions.load_css()

import streamlit_debug
streamlit_debug.set(flag=True, wait_for_client=False, host='localhost', port=3210)

#----------------------------------------------------------------------------
# KEDRO CONFIG

from pathlib import Path
from kedro.framework.project import configure_project

package_name = Path(__file__).parent.name
configure_project(package_name)

KEDRO_VIZ_SERVER_URL = 'http://127.0.0.1:4141/'

#----------------------------------------------------------------------------

state = st.session_state

if 'kedro_viz_started' not in state:
    state['kedro_viz_started'] = False

if 'chart_theme' not in state:
    state['chart_theme'] = None
if 'chart_kwargs' not in state:
    state['chart_kwargs'] = {}
if 'chart_template' not in state:
    state['chart_template'] = 'plotly_dark'

if 'show_table' not in state:
    state['show_table'] = False

if 'menu_choice' not in state:
    state['menu_choice'] = 0

def _set_chart_theme_cb():
    if state['key_chart_theme']:
        state['chart_theme'] = 'streamlit'
        state['chart_kwargs'] = {}
    else:
        state['chart_theme'] = None
        state['chart_kwargs'] = {'template': state['chart_template']}

def _charts_selectbox_cb(menu_map):
    state['menu_choice'] = list(menu_map.keys()).index(state['charts_selectbox'])

def _set_chart_template_cb():
    state['chart_template'] = state['key_chart_template']
    state['chart_kwargs'] = {'template': state['chart_template']}

def _show_table_checkbox_cb():
    state['show_table'] = state['show_table_checkbox']

# -----------------------------------------------------------------------------
# DATA WRAPPERS (uses latest Streamlit 1.18 @st.cache_data)

@st.cache_data(ttl=dt.timedelta(hours=1), show_spinner="Building data catalog")
def data_catalog() -> MyDataCatalog:
    catalog = MyDataCatalog()
    datasets = catalog.build_data_catalog()
    print('Available datasets:', datasets)
    return catalog
    
@st.cache_data(ttl=dt.timedelta(hours=1), show_spinner="Loading crypto candles data")
def load_data(symbol):
    data = data_catalog().load('crypto_candles_data')
    df_oclh = data.copy().query(f"symbol == '{symbol}' and period == '{OCLH_PERIOD}'")
    df_oclh['Timestamp'] = pd.to_datetime(df_oclh['Timestamp'], dayfirst=True)
    df_oclh.set_index('Timestamp', inplace=True)
    return df_oclh

@st.cache_data(ttl=dt.timedelta(hours=1), show_spinner="Loading model features data")
def load_features(symbol):
    data = data_catalog().load(f'{symbol.lower()}_crypto_features_data')
    df_features = data.copy()
    df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], dayfirst=True)
    df_features.set_index('Timestamp', inplace=True)
    return df_features

@st.cache_data(ttl=dt.timedelta(hours=1), show_spinner="Converting data to CSV")
def _convert_df_to_csv(df: pd.DataFrame, index=False, name=None):
    return df.to_csv(index=index, encoding='utf-8')

#----------------------------------------------------------------------------
# KEDRO VIZ SERVER

def launch_kedro_viz_server(reporter):

    if not state['kedro_viz_started']:
        import os
        import subprocess
        import threading

        def _run_job(job):
            print (f"\nRunning job: {job}\n")
            proc = subprocess.Popen(job)
            proc.wait()
            return proc

        if st.secrets['OS'] == 'windows':
            job = [os.path.join('.\\', 'kedro_viz.cmd')]
        else:
            job = [os.path.join('./', 'kedro_viz.sh')]

        reporter.warning('Starting visualization server...')
        time.sleep(3)
        # server thread will remain active as long as streamlit thread is running, or is manually shutdown
        thread = threading.Thread(name='Kedro-Viz', target=_run_job, args=(job,), daemon=True)
        thread.start()
        reporter.info('Waiting for server response...')
        time.sleep(3)

        retries = 5
        while True:
            reporter.info('Waiting for server response...')
            # give it time to start
            resp = None
            try:
                resp = requests.get(KEDRO_VIZ_SERVER_URL)
            except:
                pass
            if resp and resp.status_code == 200:
                state['kedro_viz_started'] = True
                reporter.empty()
                break
            else:
                time.sleep(1)
            retries -= 1
            if retries < 0:
                reporter.info('Right click on the empty iframe and select "Reload frame"')
                break

#----------------------------------------------------------------------------
# 
# PAGE DISPLAY FUNCTIONS
#
#----------------------------------------------------------------------------
# CANDLESTICKS

def page_candlesticks(symbol, df_oclh: pd.DataFrame):
    st.markdown(f'### CANDLESTICKS for {symbol}')

    layout = {
        'title': f'{symbol} Price and Volume Chart',
        'xaxis': {'title': 'T'},
        'yaxis': {'title': 'Closing Price ($)'},
        'xaxis_rangeslider_visible': True,
        'width': 1200,
        'height': 800,
    }
    if state['chart_kwargs']:
        layout['template'] = state['chart_kwargs']['template']

    fig = go.Figure(
        data = [
            go.Candlestick(
                x=df_oclh.index,
                open=df_oclh['open'],
                high=df_oclh['high'],
                low=df_oclh['low'],
                close=df_oclh['close'],
                increasing_line_color='green',
                decreasing_line_color='#FF4B4B',
            )
        ],
        layout = layout,
    )
    # fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, theme=state['chart_theme'])
    
    df_oclh_copy = df_oclh.copy()
    df_oclh_copy['up_down'] = np.where(df_oclh_copy['close'] >= df_oclh_copy['open'], 'up', 'down')
    print(df_oclh_copy.head())
    fig = px.bar(
        df_oclh_copy, 
        x=df_oclh_copy.index, y='volume', 
        labels={df_oclh_copy.index.name: 'T', 'volume': 'Volume'},
        color='up_down',
        color_discrete_sequence=['green', '#FF4B4B'],
        opacity = 0.6,
        width=1200, height=250,
        **state['chart_kwargs']
    )
    st.plotly_chart(fig, theme=state['chart_theme'])

#----------------------------------------------------------------------------
# INDICATORS

def page_price_indicators(symbol, df_features: pd.DataFrame):
    st.markdown(f'### CLOSE PRICE & INDICATORS for {symbol}')

    fig = px.line(
        df_features,
        x=df_features.index, y='close',
        labels={'Timestamp': 'T', 'close': 'Close'},
        color='series_id',
        title=f'{OCLH_PERIOD} Frequency Close Prices ({symbol})',
        width=1200, height=800,
        **state['chart_kwargs']
    )
    st.plotly_chart(fig, theme=state['chart_theme'])
    
    columns = [col for col in df_features.columns if not col in [
        'Timestamp', 'symbol', 'period', 'series_id', 
        'open', 'low', 'high', f'close_t{FORECAST_HORIZON}'
    ]]
    indicators = st.multiselect('Select indicator series', options=columns, default=['close', 'ema_short', 'ema_long'], max_selections=5)

    fig = px.line(
        df_features[indicators],
        x=df_features.index, y=indicators,
        labels={'Timestamp': 'T'},
        # color=indicators,
        title=f'{OCLH_PERIOD} Frequency ({symbol})',
        width=1200, height=800,
        **state['chart_kwargs']
    )
    st.plotly_chart(fig, theme=state['chart_theme'])

#----------------------------------------------------------------------------
# TRAIN / TEST

def page_train_test(symbol, df_oclh: pd.DataFrame):
    st.markdown(f'### TRAIN & TEST DATA SPLITS for {symbol}')

    train_df = df_oclh[TRAIN_INDEX].copy()
    if not train_df.empty:
        train_df['split_id'] = 'train'
    else:
        st.error(
            f'Training data set is not in display window. '
            f'Increase number of days data in window (split_date = {SPLIT_DATE}).'
        )
    test_df = df_oclh[TEST_INDEX].copy()
    test_df['split_id'] = 'test'
    
    train_test_df = pd.concat([train_df, test_df], axis=0)
        
    fig = px.line(
        train_test_df,
        x=train_test_df.index, y='close', 
        labels={train_test_df.index.name: 'T', 'close': f'{symbol} Price ($)'},
        color='split_id',
        # color_discrete_sequence=['blue','green'],
        title=f'Train / Test Split: {symbol}',
        width=1200, height=800,
        **state['chart_kwargs']
    )
    st.plotly_chart(fig, theme=state['chart_theme'])

# -----------------------------------------------------------------------------
# PREDICTIONS

def page_predictions(symbol):
    st.markdown(f'### PREDICTIONS for {symbol}')
    st.write('')

    reporter = st.empty()
    
    result_df = pd.read_csv(f'./data/{symbol.lower()}_predictions.csv', encoding='utf-8', keep_default_na=True)

    c1, c2, _ = st.columns([1,1,3])
    with c1:
        y, y_pred = result_df['close_t1'], result_df['close_pred_t1']
        mse = mean_squared_error(y, y_pred)
        st.markdown('##### 🤒 Mean Square Error (MSE)')
        st.metric('Mean Square Error (MSE)', f'{round(mse * 100, 3)}%' , f'{round((0.05 - mse) * 100, 3)}%', label_visibility='collapsed')
    with c2:
        # Launch button will only work locally
        if not st.secrets['IS_ST_CLOUD']:
            st.markdown('##### ⚙️ Pipeline visualization')
            launch_kedro_viz_server(reporter)
            if state['kedro_viz_started']:
                reporter.empty()
                st_functions.st_button('kedro', KEDRO_VIZ_SERVER_URL, 'Launch Kedro-Viz', 40)
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

# -----------------------------------------------------------------------------
# SETTINGS and MENU

def sidebar_menu():
    with st.sidebar:
        c1, _ = st.columns([1,1])
        with c1:
            st.image(Image.open('./images/a12i_logo.png'))
        st.header('Kedro ML Pipeline')
        menu_selection = st.radio('What would you like to do?', [
            '📈 View source data charts', 
            '👣 Run model (manual)', 
            '🥁 Run model (pipeline orchestration)',
            '❤️ Pipeline visualization (embedded)',
            '🙋 About',
        ], horizontal=False)
    return menu_selection


def sidebar_settings():
    with st.sidebar:
        st.subheader('Settings')
        with st.form(key='settings_form'):
            options = ['LTC', 'SOL', 'UNI', 'DOT']
            symbol = st.selectbox('💰 Select coin', options=options, index=1)
            st.form_submit_button('Apply', type='primary')
        st.checkbox('🔢 Show source data table', state['show_table'], key='show_table_checkbox', on_change=_show_table_checkbox_cb)
    return symbol


def sidebar_chart_style_and_other_settings():
    with st.sidebar:
        st.subheader('Chart style')
        c1, c2 = st.columns(2)
        with c1:
            st.caption('🎈 Theme')
            st.checkbox('Streamlit', value=state['chart_theme'], on_change=_set_chart_theme_cb, key='key_chart_theme')
        with c2:
            if not state['chart_theme']:
                st.caption('🌈 Template')
                st.selectbox(
                    'Label should not be visible', options=px_templates, index=px_templates.index(state['chart_template']), 
                        label_visibility='collapsed', on_change=_set_chart_template_cb, key='key_chart_template'
                )

        st.markdown('---')
        if st.button('🧹 Clear cache', type='primary', help='Refresh source data and data catalog for this application'):
            data_catalog.clear()
            load_data.clear()
            load_features.clear()
            _convert_df_to_csv.clear()
            st.experimental_rerun()

# -----------------------------------------------------------------------------
# TOP LEVEL MENU ACTIONS

def view_source_data_charts(symbol):

    df_oclh = load_data(symbol)
    df_features = load_features(symbol)
    menu_map = {
        'Candlesticks': (page_candlesticks, [symbol, df_oclh]),
        'Price & Indicators': (page_price_indicators, [symbol, df_features]),
        'Train | Test Split': (page_train_test, [symbol, df_oclh]),
    }

    with st.sidebar:
        st.subheader('Charts and Tables')
        menu_choice = st.radio(
            'Charts',
            label_visibility='collapsed',
            options=menu_map.keys(), 
            index=state['menu_choice'], 
            key='charts_selectbox', 
            on_change=_charts_selectbox_cb,
            args=(menu_map,)
        )
        
    if state['show_table']:
        with st.expander(f'Data Tables ({symbol})', expanded=True):
            tab1, tab2 = st.tabs(['OCLH Data', 'Features Data'])
            
            with tab1:
                st.markdown(f'### OCLH Data for {symbol}')
                c1, c2 = st.columns([3,1])
                with c1:
                    st.write(df_oclh.sort_values(by=df_oclh.index.name, ascending=False))
                    st.caption(f'Size {df_oclh.shape}')
                    file_name=f'{symbol.lower()}_oclh.csv'
                    st.download_button( 
                        label='📥 Download OCLH Data',
                        help=file_name,
                        data=_convert_df_to_csv(df_oclh, index=True, name=file_name),
                        file_name=file_name,
                        mime='text/csv',
                    )
                with c2:
                    st.write(df_oclh.shape)
                    st.json(list(df_oclh.dtypes), expanded=False)
                    
            with tab2:
                st.markdown(f'### Features Data for {symbol}')
                c1, c2 = st.columns([3,1])
                with c1:
                    st.write(df_features.sort_values(by=df_features.index.name, ascending=False))
                    st.caption(f'Size {df_features.shape}')
                    file_name=f'{symbol.lower()}_features.csv'
                    st.download_button( 
                        label='📥 Download Features Data',
                        help=file_name,
                        data=_convert_df_to_csv(df_features, index=True, name=file_name),
                        file_name=file_name,
                        mime='text/csv',
                    )
                with c2:
                    st.write(df_features.shape)
                    st.json(list(df_features.dtypes), expanded=False)

    fn = menu_map[menu_choice][0]
    args = menu_map[menu_choice][1]
    fn(*args)


def run_model_manual(symbol):
    df_features = load_features(symbol)
    model = train_model(df_features[TRAIN_INDEX])
    y, y_pred, mse = evaluate_model(model, df_features[TEST_INDEX])
    plot_metric(symbol, y, y_pred, mse)
    page_predictions(symbol)
    

def run_model_pipeline(symbol):
    pipeline_json = create_pipeline(**{'symbol': symbol}).to_json()
    with open(f'./data/{symbol.lower()}_pipeline.json', 'wt', encoding='utf-8') as fp:
        fp.write(pipeline_json)
    run_pipeline(symbol, data_catalog())
    page_predictions(symbol)


def show_pipeline_viz(symbol):
    # Render the pipeline graph (cool demo here: https://demo.kedro.org/)
    st.subheader('KEDRO PIPELINE VISUALIZATION')
    
    reporter = st.empty()
    
    if st.secrets['IS_ST_CLOUD']:
        st.markdown('**_The interactive pipeline visualization is only available when running this app on your local computer. Please [clone the app](https://github.com/asehmi/using_chatgpt_kedro_streamlit_app) and run it locally._**')
        st.write("Here's a preview image of what you will see:")
        st.image(Image.open('./images/kedro_viz.png'))
        return

    launch_kedro_viz_server(reporter)
    
    if state['kedro_viz_started']:
        st.caption(f'This interactive pipeline visualization is for {SYMBOL_DEFAULT} but is the same for all coins.')
        components.iframe(KEDRO_VIZ_SERVER_URL, width=1500, height=800)
   

def show_about():
    c1, _ = st.columns([1,2])
    with c1:
        st.markdown("""
            ## Using ChatGPT to build a Kedro ML pipeline

            Hi community! 👋

            My name is Arvindra Sehmi, and I'm an active member of the Streamlit Creators group. I’m on a break from a 35-year-long career in tech 
            (currently advising [Auth0.com](http://auth0.com/), [Macrometa.com](http://macrometa.com/), [Tangle.io](http://tangle.io/), 
            [Crowdsense.ai](https://crowdsense.ai/), and [DNX ventures](https://www.dnx.vc/)) and am taking the opportunity to learn new software development tools.

            I recently came across an open-source Python DevOps framework [Kedro](https://kedro.org/) and thought, "Why not  have [ChatGPT](https://chat.openai.com/chat) 
            teach me how to use it to build some ML/DevOps automation?"

            The idea was to:
            1. Ask ChatGPT some basic questions about Kedro.
            2. Ask it to use more advanced features in the Kedro framework.
            3. Write my questions with hints and phrases that encouraged explanations of advanced Kedro features (to evolve incrementally as if I were taught by a teacher).

            Kedro has some pipeline visualization capabilities, so I wondered:
            - Could ChatGPT show me how to display pipeline graphs in Streamlit?
            - Could ChatGPT build me an example ML model and explicitly refer to it in the Kedro pipeline?
            - What does it take to scale the pipeline, and perform pipeline logging, monitoring, and error handling?
            - Could I connect Kedro logs to a cloud-based logging service?
            - Could ChatGPT contrast Kedro with similar (competing) products and services and show me how the pipeline it developed earlier could be implemented in one of them?

            I wrote a blog post (link pending) with annotated responses to the answers I got to my questions plan. I was super impressed and decided to implement the 
            Kedro pipeline and Streamlit application from what I learned. My [GitHub](https://github.com/asehmi/using_chatgpt_kedro_streamlit_app) repository 
            contains the code for the application and details of installing and running it yourself. 
            
            The code is released to the public under MIT License. 
            
            Happy Streamlit-ing! 🎈
        """)
    c1, _ = st.columns([1,5])
    with c1:
        st_functions.st_button('twitter', 'https://twitter.com/asehmi/', 'Follow me on Twitter', 20)
        st_functions.st_button('linkedin', 'https://www.linkedin.com/in/asehmi/', 'Follow me on LinkedIn', 20)
        st_functions.st_button('cup', 'https://www.buymeacoffee.com/asehmi', 'Buy me a Coffee', 20)

# -----------------------------------------------------------------------------
# TOP LEVEL MENU ACTIONS DISPATCHER

menu_selection = sidebar_menu()
if menu_selection == '📈 View source data charts':
    symbol = sidebar_settings()
    view_source_data_charts(symbol)
    sidebar_chart_style_and_other_settings()
if menu_selection == '👣 Run model (manual)':
    symbol = sidebar_settings()
    run_model_manual(symbol)
    sidebar_chart_style_and_other_settings()
if menu_selection == '🥁 Run model (pipeline orchestration)':
    symbol = sidebar_settings()
    run_model_pipeline(symbol)
    sidebar_chart_style_and_other_settings()
if menu_selection == '❤️ Pipeline visualization (embedded)':
    symbol = sidebar_settings()
    show_pipeline_viz(symbol)
    sidebar_chart_style_and_other_settings()
if menu_selection == '🙋 About':
    show_about()
