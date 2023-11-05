# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class StochRSITEMA(IStrategy):
    """
    author@: werkkrew
    github@: https://github.com/werkkrew/freqtrade-strategies

    Reference: Strategy #1 @ https://tradingsim.com/blog/5-minute-bar/

    Trade entry signals are generated when the stochastic oscillator and relative strength index provide confirming signals.

    Buy:
        - Stoch slowd and slowk below lower band and cross above
        - Stoch slowk above slowd
        - RSI below lower band and crosses above

    You should exit the trade once the price closes beyond the TEMA in the opposite direction of the primary trend.
    There are many cases when candles are move partially beyond the TEMA line. We disregard such exit points and we exit the market when the price fully breaks the TEMA.

    Sell:
        - Candle closes below TEMA line (or open+close or average of open/close)
        - ROI, Stoploss, Trailing Stop
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    # 47/50:     19 trades. 7/6/6 Wins/Draws/Losses. Avg profit  -0.35%. Median profit   0.00%. Total profit -0.00006706 BTC (  -6.69Î£%). Avg duration  80.3 min. Objective: 1.98291

    # Buy hyperspace params:
    buy_params = {
     'rsi-lower-band': 36, 'rsi-period': 15, 'stoch-lower-band': 48
    }

    # Sell hyperspace params:
    sell_params = {
     'tema-period': 5, 'tema-trigger': 'close'
    }

    # ROI table:
    minimal_roi = {
        "0": 0.19503,
        "13": 0.09149,
        "36": 0.02891,
        "64": 0
    }

    # Stoploss:
    stoploss = -0.02205

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.17251
    trailing_stop_positive_offset = 0.2516
    trailing_only_offset_is_reached = False


    """
    END HYPEROPT
    """

    # Ranges for dynamic indicator periods
    rsiStart = 5
    rsiEnd = 30
    temaStart = 5
    temaEnd = 50

    # Stochastic Params
    fastkPeriod = 14
    slowkPeriod = 3
    slowdPeriod = 3

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    # Set this to the highest period value in the indicator_params dict or highest of the ranges in the hyperopt settings (default: 72)
    startup_candle_count: int = 50

    """
    Populate all of the indicators we need (note: indicators are separate for buy/sell)
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for rsip in range(self.rsiStart, (self.rsiEnd + 1)):
            dataframe[f'rsi({rsip})'] = ta.RSI(dataframe, timeperiod=rsip)

        for temap in range(self.temaStart, (self.temaEnd + 1)):
            dataframe[f'tema({temap})'] = ta.TEMA(dataframe, timeperiod=temap)

        # Stochastic Slow
        # fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        stoch_slow = ta.STOCH(dataframe, fastk_period=self.fastkPeriod,
                              slowk_period=self.slowkPeriod, slowd_period=self.slowdPeriod)
        dataframe['stoch-slowk'] = stoch_slow['slowk']
        dataframe['stoch-slowd'] = stoch_slow['slowd']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []

        conditions.append(
            dataframe[f"rsi({params['rsi-period']})"] > params['rsi-lower-band'])
        conditions.append(qtpylib.crossed_above(
            dataframe['stoch-slowd'], params['stoch-lower-band']))
        conditions.append(qtpylib.crossed_above(
            dataframe['stoch-slowk'], params['stoch-lower-band']))
        conditions.append(qtpylib.crossed_above(
            dataframe['stoch-slowk'], dataframe['stoch-slowd']))

        # Check that the candle had volume
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []
        
        if params.get('tema-trigger') == 'close':
            conditions.append(
                dataframe['close'] < dataframe[f"tema({params['tema-period']})"])
        if params.get('tema-trigger') == 'both':
            conditions.append((dataframe['close'] < dataframe[f"tema({params['tema-period']})"]) & (
                dataframe['open'] < dataframe[f"tema({params['tema-period']})"]))
        if params.get('tema-trigger') == 'average':
            conditions.append(
                ((dataframe['close'] + dataframe['open']) / 2) < dataframe[f"tema({params['tema-period']})"])

        # Check that the candle had volume
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
