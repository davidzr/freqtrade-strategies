# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class StrategyScalpingFast(IStrategy):

    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.5
    timeframe = '1m'
    timeframe_support = '5m'
    timeframe_main = '5m'

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    ignore_buying_expired_candle_after = 0
    trailing_stop = False

    startup_candle_count: int = 20


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['open'] < dataframe['ema_low']) &
                    (dataframe['adx'] > 30) &
                    (dataframe['mfi'] < 30) &
                    (
                        (dataframe['fastk'] < 30) &
                        (dataframe['fastd'] < 30) &
                        (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                    ) &
                    (dataframe['cci'] < -150)
                )

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (
                        (dataframe['open'] >= dataframe['ema_high'])

                    ) |
                    (
                        (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                        (qtpylib.crossed_above(dataframe['fastd'], 70))

                    )
                ) & (dataframe['cci'] > 150)
            ),
            'sell'] = 1
        return dataframe
