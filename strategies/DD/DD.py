# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class DD(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.11078,
        "33": 0.06816,
        "68": 0.02844,
        "165": 0
    }

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0102
    trailing_stop_positive_offset = 0.03701
    trailing_only_offset_is_reached = False
    # Optimal stoploss designed for the strategy
    stoploss = -0.32745

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include buy
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband']) &
             #   (dataframe['mfi'] < 16) &
              #  (dataframe['adx'] > 25) &
                (dataframe['rsi'] < 40)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include sell
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband']) 
             #   (dataframe['rsi'] > 21)
                
                
                
          #      (qtpylib.crossed_above(
           #         dataframe['macdsignal'], dataframe['macd']
          #      )) &
          #      (dataframe['fastd'] > 54)
            ),
            'sell'] = 1
        return dataframe
