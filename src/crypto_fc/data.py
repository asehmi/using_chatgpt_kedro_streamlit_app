from typing import List
import pandas as pd
# https://github.com/twopirllc/pandas-ta
import pandas_ta as ta
import numpy as np

from .constants import (
    OCLH_PERIOD,
    TIME_PERIOD,
    SHORT_PERIOD,
    LONG_PERIOD,
    FORECAST_HORIZON,
)

from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import CSVDataSet, GenericDataSet

# -----------------------------------------------------------------------------
# LOW LEVEL DATA CATALOG FUNCTIONS (not used)

def __exists(var: str):
     return var in globals()

if not __exists('__DATA_CATALOG__'):
    # Create a DataCatalog to manage the data for project
    global __DATA_CATALOG__
    __DATA_CATALOG__: DataCatalog = DataCatalog()

def catalog_add_dataset(name, dataset):
    global __DATA_CATALOG__
    if not __DATA_CATALOG__:
        raise Exception('Data catalog has not been initialised.')
    else:
        __DATA_CATALOG__.add(name, dataset)

def catalog_load_dataset_data(name):
    global __DATA_CATALOG__
    if not __DATA_CATALOG__:
        raise Exception('Data catalog has not been initialised.')
    else:
        # Load the data registered with catalog
        df_data = __DATA_CATALOG__.load(name)
    return df_data

# -----------------------------------------------------------------------------
# DATA CATALOG WRAPPER

class MyDataCatalog(DataCatalog):
    
    def load_crypto_candles(self) -> pd.DataFrame:
        # COMMENT: Data loading code written by ChatGPT was incorrect
        crypto_ds = CSVDataSet(filepath="./data/crypto_candles_data.csv", load_args=None, save_args={'index': False})
        self.add('crypto_candles_data', crypto_ds)
        df_data: pd.DataFrame = self.load('crypto_candles_data')
        return df_data

    def filter_data_and_build_features(
        self,
        symbol, df_oclh,
        period=OCLH_PERIOD, timeperiod=TIME_PERIOD,
        ema_short_period=SHORT_PERIOD, ema_long_period=LONG_PERIOD,
        forecast_horizon=FORECAST_HORIZON,
    ) -> pd.DataFrame:
        if not isinstance(df_oclh, pd.DataFrame):
            raise Exception('ERROR: filter_data_and_build_features null data')

        # FILTER OCLH DATA FRAME

        # Key for the OCLH series data set. Each symbol's data set contains:
        #   'Timestamp', 'open', 'close', 'low', 'high', 'volume',
        df_oclh_filtered = df_oclh[
            df_oclh['symbol'] == symbol
        ].copy()

        # backfill OCLH dataframe for good measure
        df_oclh_filtered.fillna(method='bfill', inplace=True)

        # BUILD FILTERED FEATURE DATA SERIES
        
        # Price OCLH series
        open_price = df_oclh_filtered['open']
        close_price = df_oclh_filtered['close']
        low_price = df_oclh_filtered['low']
        high_price = df_oclh_filtered['high']
        # Volume series
        volume = df_oclh_filtered['volume']

        # CREATE NEW INDICATORS (using the above series)

        # SecurityPrice indicators
        close_price_pct_change = close_price.pct_change(periods=1, fill_method='bfill')

        rsi = ta.rsi(close_price, length=timeperiod)

        ema_short = ta.ema(close_price, length=ema_short_period) 
        ema_long = ta.ema(close_price, length=ema_long_period) 

        obv = ta.obv(close_price, volume)

        obv_ema_short = ta.ema(obv, length=ema_short_period)
        obv_ema_long = ta.ema(obv, length=ema_long_period)

        # Volatility measures
        
        close_off_high_temp = ( ((high_price - close_price) * 2)
                        / (high_price - low_price - 1) )
        close_off_high = np.nan_to_num(close_off_high_temp, copy=True, posinf=0.0001, neginf=-0.0001)
        
        volat_zero_base_temp = (high_price - low_price) / open_price
        volat_zero_base = np.nan_to_num(volat_zero_base_temp, copy=True, posinf=0.0001, neginf=-0.0001)

        # BUILD INDICATORS DATA FRAME

        INDICATORS_DICT = {
            'close_pct_change': close_price_pct_change,
            'close_off_high': close_off_high,
            
            'rsi': rsi,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'obv': obv,
            'obv_ema_short': obv_ema_short,
            'obv_ema_long': obv_ema_long,

            'volat_zero_base': volat_zero_base,
        }
        
        # print([(k,len(v)) for k,v in INDICATORS_DICT.items()])
        
        indicators_df = pd.DataFrame(INDICATORS_DICT)
        
        # print('Indicators dataframe shape:', indicators_df.shape)

        # combine into final data frame
        df_features = df_oclh_filtered.join(indicators_df, how='inner')

        # add prediction target to final dataframe
        
        forecast_horizon_label = 'close' + f'_t{forecast_horizon}'
        df_features[forecast_horizon_label] = df_features['close'].shift(-forecast_horizon).fillna(method='ffill') # used as target prediction
        
        # FINALLY, LIMIT FULL FEATURE SET TO REQUIRED COLUMNS
        
        oclh_series_id = f'{symbol.upper()}#{period}'
        df_features['series_id'] = oclh_series_id

        # df_features_final is a copy not a view!
        df_features_final = df_features[['Timestamp', 'symbol', 'open' ,'close', 'low', 'high'] + list(INDICATORS_DICT.keys()) + ['series_id', forecast_horizon_label]].copy()
        
        # TAKE CARE TO DEAL PROPERLY WITH NANs PRODUCED FROM AVERAGING

        # The mean calc window is equivalent to 3 * longest averaging period (LONG_PERIOD)
        # We backfill with the mean of the head data so it is more reflective of the amplitude in that window only
        mean_calculation_window = (3 * LONG_PERIOD)
        for col in INDICATORS_DICT.keys():
            if ('short' in col) or ('long' in col) or ('rsi' in col):
                df_features_final[col].fillna(df_features_final[col].head(mean_calculation_window).mean(), inplace=True)

        df_features_final.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features_final.fillna(method='bfill', inplace=True)

        print(f'#### OCLH and Model Features Data | Series ID: {oclh_series_id} ####')
        print('Features shape:', df_features_final.shape)
        print('Features columns:', list(df_features_final.columns))
        # print('Features dtypes:', df_features_final.dtypes)
        # print(df_features_final.head(2))
        print(df_features_final.tail(2))
        
        features_ds = GenericDataSet(filepath=f'./data/{symbol.lower()}_crypto_features_data.csv', file_format='csv', load_args=None, save_args={'index': False})
        features_ds.save(df_features_final)
        self.add(f'{symbol.lower()}_crypto_features_data', features_ds)
        df_features_final_reloaded = self.load(f'{symbol.lower()}_crypto_features_data')

        return df_features_final_reloaded

    def build_data_catalog(self) -> List[str]:
        df_data = self.load_crypto_candles()
        symbols = df_data['symbol'].unique()
        for symbol in symbols:
            self.filter_data_and_build_features(symbol, df_data)
        catalog_datasets = self.list()
        print(catalog_datasets)
        return catalog_datasets
