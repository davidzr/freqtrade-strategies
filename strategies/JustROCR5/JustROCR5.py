from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class JustROCR5(IStrategy):
    minimal_roi = {
        "0": 0.05
    }

    stoploss = -0.01
    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rocr'] = ta.ROCR(dataframe, timeperiod=5)
        dataframe['rocr_2'] = ta.ROCR(dataframe, timeperiod=2)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr'] > 1.10) &
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
