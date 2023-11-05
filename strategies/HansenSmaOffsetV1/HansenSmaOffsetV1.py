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
""" use 15 open trade, unlimited stake. 

pairlist setting:

"pairlists": [
        {
            "method": "VolumePairList",
            "number_assets": 50,
            "sort_key": "quoteVolume",
            "refresh_period": 1800
        }
    ],

"""

class HansenSmaOffsetV1(IStrategy):
    timeframe = '15m'
    #I haven't found the optimal ROI yet
    minimal_roi = {
        "0": 10,
    }
    stoploss = -99
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:   
        dataframe['smau1'] = ta.SMA(dataframe['close'], timeperiod=20)+0.05*ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['smad1'] = ta.SMA(dataframe['close'], timeperiod=20)-0.05*ta.SMA(dataframe['close'], timeperiod=20)
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
                (dataframe['high']<dataframe['smad1'])&
                (dataframe['hopen'] < dataframe['hclose'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['low']>dataframe['smau1'])&
                (dataframe['hopen'] > dataframe['hclose'])
            ),
            'sell'] = 1
        return dataframe
