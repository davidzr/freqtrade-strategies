import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy

class SRsi(IStrategy):

    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.012
    }

    stoploss = -0.15

    timeframe = '1m'
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    startup_candle_count: int = 120

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def informative_pairs(self):

        return []

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        p = 14
        d = 3
        k = 3

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=30)
        srsi  = (dataframe['rsi'] - dataframe['rsi'].rolling(p).min()) / (dataframe['rsi'].rolling(p).max() - dataframe['rsi'].rolling(p).min())
        dataframe['k'] = srsi.rolling(k).mean() * 100
        dataframe['d'] = dataframe['k'].rolling(d).mean()


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
        (
                (dataframe['k'] < 15) &
                (dataframe['k'] >= dataframe['d'])
                
	    ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['k'] > 75) &
                (dataframe['d'] >= dataframe['k'])
            ),
            'sell'] = 1
        return dataframe
