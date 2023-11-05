# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class ADX_15M_USDT(IStrategy):
    ticker_interval = '15m'

    # ROI table:
    minimal_roi = {
        "0": 0.26552,
        "30": 0.10255,
        "210": 0.03545,
        "540": 0
    }

    # Stoploss:
    stoploss = -0.1255

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        dataframe['sell-adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['sell-plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['sell-minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe['sell-sar'] = ta.SAR(dataframe)
        dataframe['sell-mom'] = ta.MOM(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 16) &
                    (dataframe['minus_di'] > 4) &
                    (dataframe['plus_di'] > 20) &
                    (qtpylib.crossed_above( dataframe['plus_di'],dataframe['minus_di']))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 43) &
                    (dataframe['minus_di'] > 22) &
                    (dataframe['plus_di'] > 20) &
                    (qtpylib.crossed_above(dataframe['sell-minus_di'], dataframe['sell-plus_di']))

            ),
            'sell'] = 1
        return dataframe
