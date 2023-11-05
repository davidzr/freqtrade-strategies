# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class BbRoi(IStrategy):
    minimal_roi = { 
        "0": 0.17552,
        "53": 0.11466,
        "226": 0.06134,
        "400": 0
    }   

    # Stoploss:
    stoploss = -0.23701

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01007
    trailing_stop_positive_offset = 0.01821
    trailing_only_offset_is_reached = True

    ticker_interval = '15m'

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_middleband']) &
                    (dataframe['close'] < dataframe['bb_upperband']) &
                    (dataframe['close'] > dataframe['ema9']) &
                    (dataframe['close'] > dataframe['ema200']) &
                    (dataframe['ema20'] > dataframe['ema200'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 75) |
                    (dataframe['close'] < dataframe['bb_middleband'] * 0.97) &
                    (dataframe['open'] > dataframe['close'])  # red bar

            ),
            'sell'] = 1
        return dataframe

