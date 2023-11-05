# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AlligatorStrat(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        buys and sells on crossovers - doesn't really perfom that well and its just a proof of concept
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.1
        # "53": 0.06157,
        # "93": 0.0518,
        # "187": 0.03,
    }


    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.2

    # Optimal ticker interval for the strategy
    ticker_interval = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # CCIPeriod = 14
        # T3Period = 5
        # b = 0.618

        dataframe['SMAShort'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['SMAMedium'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['SMALong'] = ta.SMA(dataframe, timeperiod=13)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # b2 = b*b2b3 = b2 * b2c1 = -b3
        # c2 = (3 * (b2+b3))
        # c3 = -3 * (2*b2+b+b3)
        # c4 = (1+3*b+b3+3*b2)

        # nr = 1 + 0.5 * (T3period - 1)
        # w1 = 2 / (nr + 1)
        # w2 = 1 - w1

        # xcci = ta.CCI(CCIPeriod)




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                #or cross above SMALong to be more conservative
                (
                qtpylib.crossed_above(dataframe['SMAShort'], dataframe['SMAMedium']) &
                ((dataframe['macd'] > -0.00001)) &
                (dataframe['macd'] > dataframe['macdsignal'])
                )
                |
                qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
                # |
                # (
                # (dataframe['SMAShort'] > dataframe['SMAMedium']) &
                # ((dataframe['macd'] > -0.00006)) &
                # qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
                # )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                # qtpylib.crossed_below(dataframe['SMAShort'], dataframe['SMALong']) &
                ((dataframe['close'] < dataframe['SMAMedium']) &
                (dataframe['macd'] < dataframe['macdsignal'])
                )
                |
                qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
                # (dataframe['cci'] >= 100.0)
            ),
            'sell'] = 1
        return dataframe
