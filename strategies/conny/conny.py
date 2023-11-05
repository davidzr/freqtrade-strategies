import numpy as np # noqa
import pandas as pd # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.consensus import Consensus

class conny(IStrategy):

    minimal_roi = {
        "0": 0.025,
        "10": 0.015,
        "20": 0.01,
        "30": 0.005,
        "120": 0
    }

    stoploss = -0.0203

    timeframe = '15m'

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 30


    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Consensus strategy
        # add c.evaluate_indicator bellow to include it in the consensus score (look at
        # consensus.py in freqtrade technical)
        # add custom indicator with c.evaluate_consensus(prefix=<indicator name>)
        c = Consensus(dataframe)
        c.evaluate_rsi()
        c.evaluate_stoch()
        c.evaluate_macd_cross_over()
        c.evaluate_macd()
        c.evaluate_hull()
        c.evaluate_vwma()
        c.evaluate_tema(period=12)
        c.evaluate_ema(period=24)
        c.evaluate_sma(period=12)
        c.evaluate_laguerre()
        c.evaluate_osc()
        c.evaluate_cmf()
        c.evaluate_cci()
        c.evaluate_cmo()
        c.evaluate_ichimoku()
        c.evaluate_ultimate_oscilator()
        c.evaluate_williams()
        c.evaluate_momentum()
        c.evaluate_adx()
        dataframe['consensus_buy'] = c.score()['buy']
        dataframe['consensus_sell'] = c.score()['sell']


        print(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['consensus_buy'] > 45) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['consensus_sell'] > 88) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe