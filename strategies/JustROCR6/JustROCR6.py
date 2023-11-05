from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class JustROCR6(IStrategy):
    minimal_roi = {
        "0": 0.05
    }

    stoploss = -0.01
    trailing_stop = True
    ticker_interval = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rocr_499'] = ta.ROCR(dataframe, timeperiod=499)
        dataframe['rocr_200'] = ta.ROCR(dataframe, timeperiod=200)
        dataframe['rocr_100'] = ta.ROCR(dataframe, timeperiod=100)
        dataframe['rocr_50'] = ta.ROCR(dataframe, timeperiod=50)
        dataframe['rocr_10'] = ta.ROCR(dataframe, timeperiod=10)
        dataframe['rocr_5'] = ta.ROCR(dataframe, timeperiod=5)
        dataframe['rocr_2'] = ta.ROCR(dataframe, timeperiod=2)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr_499'] > 1.20) &
                (dataframe['rocr_200'] > 1.15) &
                (dataframe['rocr_100'] > 1.125) &
                (dataframe['rocr_50'] > 1.10) &
                (dataframe['rocr_10'] > 1.075) &
                (dataframe['rocr_5'] > 1.05) &
                (dataframe['rocr_2'] > 1.01)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
