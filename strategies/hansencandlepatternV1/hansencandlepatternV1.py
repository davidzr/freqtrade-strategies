# --- Do not remove these libs --- freqtrade backtesting --strategy SmoothScalp --timerange 20210110-20210410
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class hansencandlepatternV1(IStrategy):
    """
        This strategy is only an experiment using candlestick pattern to be used as buy or sell indicator. Do not use this strategy live.
    """

    timeframe = '1h'
    minimal_roi = {
        "0": 10,
    }
    stoploss = -0.1
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:   
        dataframe['3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['EVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['ABANDONEDBABY'] = ta.CDLEVENINGSTAR(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['HARAMI'] = ta.CDLHARAMI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['INVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['ENGULFING'] = ta.CDLENGULFING(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['hclose']=(dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['hopen']= ((dataframe['open'].shift(2) + dataframe['close'].shift(2))/ 2)
        dataframe['hhigh']=dataframe[['open','close','high']].max(axis=1)
        dataframe['hlow']=dataframe[['open','close','low']].min(axis=1)
        dataframe['emac'] = ta.SMA(dataframe['hclose'], timeperiod=6)
        dataframe['emao'] = ta.SMA(dataframe['hopen'], timeperiod=6)
        return dataframe
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe['3LINESTRIKE'] < 0)|(dataframe['EVENINGSTAR'] > 0)|(dataframe['ABANDONEDBABY'] > 0)|(dataframe['HARAMI'] > 0)|(dataframe['ENGULFING'] > 0))&
                (dataframe['emao'] < dataframe['emac'])
                
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['emao'] > dataframe['emac'])
            ),
            'sell'] = 1
        return dataframe
