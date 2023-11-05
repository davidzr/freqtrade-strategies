# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

from technical.util import resample_to_interval, resampled_merge

from functools import reduce
from datetime import datetime, timedelta

# ObeliskRSI v6.1 - 2021-03-06
#
# by Obelisk 
# https://twitter.com/brookmiles
#
# Run at your own risk.
# I don't know what I'm doing.
# Backtests quite a bit better than it actually works.
# Let me know if you manage to improve it!
#
# Buys when RSI crosses the low threshold
# Doesn't wait for reversal, trying to anticipate the "bottom" of the dip.
#
# Sells on ROI (often), Stoploss (sometimes), or RSI high (rarely)
#
# Custom Stoploss starts wide and then narrows over time
# Combined with ROI, every trade will be forced to close in custom_stop_ramp_minutes, win or lose
#
# Bull/Bear trends
#
# Uses two different sets of buy/sell thresholds, which are used depending on the longer RSI trend
# Enter earlier and leave later when bullish; enter later and leave earlier when bearish
#
# Strengths
#
# Choppy markets - requires strong enough dips to trigger a buy, then we want a bounce back up
#
# Weaknesses
#
# Steady bull runs - no dips to buy, missed opportunity
# Sudden ongoing bear runs - if the market keeps dipping and dropping repeatedly that's bad,
# so use protections, eg.
#
# "protections": [
#     {
#         "method": "StoplossGuard",
#         "lookback_period": 720,
#         "trade_limit": 2,
#         "stop_duration": 720,
#         "only_per_pair": true
#     },
# ],  
# 
# Works best on high volume pairs with decent volatility eg.
#
# "SXP/USD",
# "MATIC/USD",
# "SUSHI/USD",
# "CHZ/USD",
#
# and not ETH/USD or BTC/USD


def easeInCubic(t):
    return t * t * t

def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)

def clamp01(num):
    return clamp(num, 0, 1)


class ObeliskRSI_v6_1(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '5m'

    startup_candle_count = 240
    process_only_new_candles = True

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "35": 0.04,
        "65": 0.01,
        "115": 0
    }

    # Buy hyperspace params:
    buy_params = {
     'bear-buy-rsi-value': 21, 
     'bull-buy-rsi-value': 35
    }

    # Sell hyperspace params:
    sell_params = {
     'bear-sell-rsi-value': 55, 
     'bull-sell-rsi-value': 69 
    }

    # Stoploss:
    stoploss = -0.30

    use_custom_stoploss = True
    custom_stop_ramp_minutes = 110

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # longer RSI used for determining trend
        resample_rsi_indicator = self.get_ticker_indicator() * 12
        resample_rsi_key = 'resample_{}_rsi'.format(resample_rsi_indicator)

        dataframe_long = resample_to_interval(dataframe, resample_rsi_indicator)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)
        dataframe = resampled_merge(dataframe, dataframe_long)
        dataframe[resample_rsi_key].fillna(method='ffill', inplace=True)

        # bull used to select between two different sets of buy/sell threshold values
        # based on long RSI
        dataframe['bull'] = dataframe[resample_rsi_key].gt(60).astype('int')

        # normal rsi acts mainly as the buy trigger
        # used for sell as well, but more likely to ROI or stop out
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    # buy low
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []
        conditions.append(dataframe['volume'] > 0)

        conditions.append(
            ((dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bull-buy-rsi-value'])) |
            (~(dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bear-buy-rsi-value']))
            )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    # sell high
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []
        conditions.append(dataframe['volume'] > 0)

        conditions.append(
            ((dataframe['bull'] > 0) & (dataframe['rsi'] > params['bull-sell-rsi-value'])) |
            (~(dataframe['bull'] > 0) & (dataframe['rsi'] > params['bear-sell-rsi-value']))
            )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe

    # Custom stoploss starts at the basic stoploss, and ramps towards zero in a curve, 
    # narrowing the trailing stoploss until forcing the trade to stop out after custom_stop_ramp_minutes

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        since_open = current_time - trade.open_date
        sl_pct = 1 - easeInCubic( clamp01( since_open / timedelta(minutes=self.custom_stop_ramp_minutes) ) )
        sl_ramp = abs(self.stoploss) * sl_pct

        return sl_ramp + 0.001 # we can't go all the way to zero
