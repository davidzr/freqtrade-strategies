from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from technical.indicators import ichimoku
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Stavix2(IStrategy):
    minimal_roi = {
        "0": 0.15
    }
    stoploss = -0.10

    ticker_interval = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cloud = ichimoku(dataframe, conversion_line_period=200, base_line_periods=350, laggin_span=150, displacement=75)
        dataframe['tenkan_sen'] = cloud['tenkan_sen']
        dataframe['kijun_sen'] = cloud['kijun_sen']
        dataframe['senkou_span_a'] = cloud['senkou_span_a']
        dataframe['senkou_span_b'] = cloud['senkou_span_b']
        dataframe['chikou_span'] = cloud['chikou_span']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
                (
                    (dataframe['close'] > dataframe['senkou_span_a']) &
                    (dataframe['close'] > dataframe['senkou_span_b']) & 
                    (qtpylib.crossed_above(dataframe['kijun_sen'], dataframe['tenkan_sen']))
                    ),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
                (
                    (dataframe['close'] < dataframe['senkou_span_a']) &
                    (dataframe['close'] < dataframe['senkou_span_b']) & 
                    (qtpylib.crossed_above(dataframe['tenkan_sen'], dataframe['kijun_sen']))
                    ),
                'sell'] = 1
        return dataframe
