# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt

# @Rallipanos

# Buy hyperspace params:
buy_params = {
      "base_nb_candles_buy": 14,
      "ewo_high": 2.327,
      "ewo_low": -19.988,
      "low_offset": 0.975,
      "rsi_buy": 69
    }

# Sell hyperspace params:
sell_params = {
      "base_nb_candles_sell": 24,
      "high_offset": 0.991,
      "high_offset_2": 0.997
    }

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        return dataframe

class ElliotV7(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.051,
        "10": 0.031,
        "22": 0.018,
        "66": 0
    }

    # Stoploss:
    stoploss = -0.32

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)        

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 39

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < -0.1 and current_time - timedelta(minutes=720) > trade.open_date_utc:
            return -0.01

        return -0.99


    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        
        informative_1h['ema_fast'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_slow'] = ta.EMA(informative_1h, timeperiod=25)

        informative_1h['uptrend'] = (
            (informative_1h['ema_fast'] > informative_1h['ema_slow'])
        ).astype('int')
        informative_1h['rsi_100'] = ta.RSI(informative_1h, timeperiod=100)
        return informative_1h



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']       
        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        
        #dataframe['hma_50']=hmao
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']       
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_100'] = ta.RSI(dataframe, timeperiod=100)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   
                (dataframe['uptrend_1h'] > 0)&
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
                

            )
        )

        conditions.append(
            (   
                
                (dataframe['uptrend_1h'] > 0)&
                (dataframe['rsi_fast'] < 35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
                
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   (dataframe['sma_9']>dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['sma_9']<dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )    
            
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe