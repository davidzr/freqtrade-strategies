from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from technical.indicators import ichimoku

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Ichimoku(IStrategy):
    """
    Ichimoku Strategy
    """

    minimal_roi = {
        "0":  1
    }

    stoploss = -0.1

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # run "populate_indicators" only for new candle
    ta_on_candle = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        """
        """

        return [(f"{self.config['stake_currency']}/USDT", self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        ichi = ichimoku(dataframe)
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        dataframe.loc[
            (
                (dataframe['tenkan'].shift(1)<dataframe['kijun'].shift(1)) &
                (dataframe['tenkan']>dataframe['kijun']) &
                (dataframe['cloud_red']==True)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
