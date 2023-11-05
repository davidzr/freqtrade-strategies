# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open

# --------------------------------

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

class adaptive(IStrategy):
    """

    author: @jilv220

    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.02
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.109

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # MAMA, FAMA, KAMA
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.25, 0.025)
        dataframe['mama_diff'] = ( ( dataframe['mama'] - dataframe['fama'] ) / dataframe['hl2'] )
        dataframe['kama'] = ta.KAMA(dataframe['close'], 84)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['kama'] > dataframe['fama']) &
                (dataframe['fama'] > dataframe['mama'] * 0.981) &
                (dataframe['r_14'] < -61.3) &
                (dataframe['mama_diff'] < -0.025) &
                (dataframe['cti'] < -0.715)
                &
                (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.05 )) &
                ( (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.125 )) )
                &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
