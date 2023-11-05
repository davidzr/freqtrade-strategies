# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401


import numpy as np  # noqa
import pandas as pd  # noqa

pd.options.mode.chained_assignment = None
from collections import deque

import talib.abstract as ta
from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame
from scipy.signal import argrelextrema


class RaposaDivergenceV1(IStrategy):
    """
    Divergence strategy
    - By alb#1349

    NOTES:
    - "argrelextrema" might have look-ahead bias so results in backtest will not be reliable
    - sell signal seems like garbage

    Based on:
    - https://raposa.trade/higher-highs-calculate-python/
    - https://medium.com/raposa-technologies/test-and-trade-rsi-divergence-in-python-34a11c1c4142

    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "k_value": 2,
        "order": 5,
        "rsi_buy": 50,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_sell": 50,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "60": 0.02,
        "120": 0.01,
        "180": 0.001
    }

    # Stoploss:
    stoploss = -0.3

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 40

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    rsi_buy = IntParameter(20, 80, default=buy_params['rsi_buy'], space='buy', optimize=True)
    order = IntParameter(1, 32, default=buy_params['order'], space='buy', optimize=True)
    k_value = IntParameter(1, 32, default=buy_params['k_value'], space='buy', optimize=True)

    rsi_sell = IntParameter(20, 80, default=sell_params['rsi_sell'], space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        dataframe = getPeaks(dataframe, key='close', order=int(self.order.value), K=int(self.k_value.value))
        dataframe = getPeaks(dataframe, key='rsi', order=int(self.order.value), K=int(self.k_value.value))

        dataframe.loc[
            (
                (dataframe['close_lows'] == -1) &
                (dataframe['rsi_lows'] == -1) &
                (dataframe['rsi'] < int(self.rsi_buy.value)) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close_highs'] == 1) &
                (dataframe['rsi_highs'] == -1) &
                (dataframe['rsi'] > int(self.rsi_sell.value)) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1

        return dataframe


def getHigherLows(data: np.array, order=5, K=2):
    """
    Finds consecutive higher lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be higher.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are higher than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] < lows[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getLowerHighs(data: np.array, order=5, K=2):
    """
    Finds consecutive lower highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be lower.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are lower than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] > highs[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getHigherHighs(data: np.array, order=5, K=2):
    """
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=5)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getLowerLows(data: np.array, order=5, K=2):
    """
    Finds consecutive lower lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be lower.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are lower than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getHHIndex(data: np.array, order=5, K=2):
    extrema = getHigherHighs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getLHIndex(data: np.array, order=5, K=2):
    extrema = getLowerHighs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getLLIndex(data: np.array, order=5, K=2):
    extrema = getLowerLows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getHLIndex(data: np.array, order=5, K=2):
    extrema = getHigherLows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getPeaks(data, key='close', order=5, K=2):
    vals = data[key].values
    hh_idx = getHHIndex(vals, order, K)
    lh_idx = getLHIndex(vals, order, K)
    ll_idx = getLLIndex(vals, order, K)
    hl_idx = getHLIndex(vals, order, K)
    data[f'{key}_highs'] = np.nan
    data[f'{key}_highs'][hh_idx] = 1
    data[f'{key}_highs'][lh_idx] = -1
    data[f'{key}_highs'] = data[f'{key}_highs'].ffill().fillna(0)
    data[f'{key}_lows'] = np.nan
    data[f'{key}_lows'][ll_idx] = 1
    data[f'{key}_lows'][hl_idx] = -1
    data[f'{key}_lows'] = data[f'{key}_highs'].ffill().fillna(0)
    return data
