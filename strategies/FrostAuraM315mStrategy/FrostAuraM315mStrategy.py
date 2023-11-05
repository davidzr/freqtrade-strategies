import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class FrostAuraM315mStrategy(IStrategy):
    """
    This is FrostAura's mark 3 strategy which aims to make purchase decisions
    based on the BB, RSI and Stochastic.
    
    Last Optimization:
        Sharpe Ratio    : 6.23754
        Profit %        : 1311.96%
        Optimized for   : Last 115+ days, 15m
        Avg             : 986.6m
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.18418,
        "36": 0.10736,
        "91": 0.02998,
        "438": 0
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.41114

    # Trailing stoploss
    trailing_stop = False

    # Optimal ticker interval for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals.
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # Stochastic Slow
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Bollinger Bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']
        
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        
        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['bb_upperband4'] = bollinger4['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        minimum_coin_price = 0.0000015
        
        dataframe.loc[
            (
                #(dataframe['slowd'] > 30) &
                #(dataframe['slowk'] > 30) &
                (dataframe['rsi'] > 22) &
                (dataframe['slowk'] < dataframe['slowd']) &
                (dataframe["close"] < dataframe['bb_lowerband3']) &
                (dataframe["close"] > minimum_coin_price)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['slowk'] < dataframe['slowd']) &
                (dataframe['rsi'] > 72) &
                (dataframe["close"] > dataframe['bb_middleband1'])
            ),
            'sell'] = 1
        
        return dataframe