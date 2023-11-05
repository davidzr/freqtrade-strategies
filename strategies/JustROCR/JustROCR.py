from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class JustROCR(IStrategy):
    minimal_roi = {
        "0": 0.20
    }

    stoploss = -0.20
    trailing_stop = True
    ticker_interval = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rocr'] = ta.ROCR(dataframe, period=499)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['rocr'] > 1.10
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
