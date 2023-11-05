# --- Do not remove these libs ---
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
      "ewo_high_2": -2.327,
      "ewo_low": -20.988,
      "low_offset": 0.975,
      "low_offset_2": 0.955,
      "rsi_buy": 69
    }

# Sell hyperspace params:
sell_params = {
      "base_nb_candles_sell": 24,
      "high_offset": 0.998,
      "high_offset_2": 1
    }

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class NotAnotherSMAOffSetStrategy_V2(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.032,
        "87": 0.016,
        "201": 0
    }

    # Stoploss:
    stoploss = -0.35

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)        
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

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)       
    
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.025
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
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]


        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50'] > last_candle['ema_100']) and (last_candle['rsi'] < 45): #*1.2
                    return False

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951): #*1.2
                    return False

        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)          

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['vol_7_max'] = dataframe['volume'].rolling(window=20).max()
        dataframe['vol_14_max'] = dataframe['volume'].rolling(window=14).max()
        dataframe['vol_7_min'] = dataframe['volume'].rolling(window=20).min()
        dataframe['vol_14_min'] = dataframe['volume'].rolling(window=14).min()
        dataframe['roll_7'] = 100*((dataframe['volume']-dataframe['vol_7_max'])/(dataframe['vol_7_max']-dataframe['vol_7_min']))
        dataframe['vol_base']=ta.SMA(dataframe['roll_7'], timeperiod=5)
        dataframe['vol_ma_26'] = ta.SMA(dataframe['volume'], timeperiod=26)
        dataframe['vol_ma_200'] = ta.SMA(dataframe['volume'], timeperiod=100)
      

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        dataframe.loc[
        (
                (dataframe['vol_base']>-96)&    
                (dataframe['vol_base']<-77)&
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        ),
        ['buy', 'buy_tag']] = (1, 'ewo1')
                
        dataframe.loc[
        (
                (dataframe['vol_base']>-96)&
                (dataframe['vol_base']> -20)&
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        ),
        ['buy', 'buy_tag']] = (1, 'ewo3')


        dataframe.loc[
        (       (dataframe['vol_base']>-96)&
                (dataframe['vol_base']<-77)&
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
                (dataframe['rsi']<25)
        ),
        ['buy', 'buy_tag']] = (1, 'ewo2')

    
        dataframe.loc[
        (
                (dataframe['vol_base']>-96)&
                (dataframe['vol_base']<-77)&
                (dataframe['rsi_fast'] < 35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        ),
        ['buy', 'buy_tag']] = (1, 'ewolow')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   (dataframe['close']>dataframe['sma_9'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )
            |
            (
                (dataframe['close']<dataframe['hma_50'])&
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