# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

import technical.indicators as ftt
from technical.util import resample_to_interval, resampled_merge

from functools import reduce
from datetime import datetime, timedelta

# Obelisk_TradeProIM v1 - 2021-03-25
#
# by Obelisk 
# https://twitter.com/brookmiles
#
# Based on "Crazy Results Best Ichimoku Cloud Trading Strategy Proven 100 Trades" by Trade Pro
# https://www.youtube.com/watch?v=8gWIykJgMNY
#
# Does not attempt to emulate the risk/reward take-profit/stop-loss, so the sell criteria are mine.

class Obelisk_TradePro_Ichi_v1_1(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '1h'

    startup_candle_count = 120
    process_only_new_candles = True

    # no ROI
    minimal_roi = {
        "0": 10,
    }

    # Stoploss:
    stoploss = -0.015

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(0,0,0,0.2)',
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {
                'color': 'red',
            },
            'tenkan_sen': { 'color': 'orange' },
            'kijun_sen': { 'color': 'blue' },

            'chikou_span': { 'color': 'lightgreen' },
        },
        'subplots': {
            "Signals": {
                'go_long': {'color': 'blue'},
                'future_green': {'color': 'green'},
                'chikou_high': {'color': 'lightgreen'},
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # # Standard Settings
        # displacement = 26
        # ichimoku = ftt.ichimoku(dataframe, 
        #     conversion_line_period=9, 
        #     base_line_periods=26,
        #     laggin_span=52, 
        #     displacement=displacement
        #     )

        # Crypto Settings
        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        # cross indicators
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        # DANGER ZONE START

        # NOTE: Not actually the future, present data that is normally shifted forward for display as the cloud
        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2

        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.
        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['senkou_a']) &
                (dataframe['chikou_span'] > dataframe['senkou_b'])
            ).shift(displacement).fillna(0).astype('int')

        # DANGER ZONE END

        dataframe['go_long'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
                (dataframe['close'] > dataframe['senkou_a']) &
                (dataframe['close'] > dataframe['senkou_b']) &
                (dataframe['future_green'] > 0) &
                (dataframe['chikou_high'] > 0)
                ).astype('int') * 3


        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

            qtpylib.crossed_above(dataframe['go_long'], 0),

        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

            qtpylib.crossed_below(dataframe['tenkan_sen'], dataframe['kijun_sen']) 
            | 
            qtpylib.crossed_below(dataframe['close'], dataframe['kijun_sen']),

        'sell'] = 1

        return dataframe

#
# Fun with outliers.
#
# ============================================================= BACKTESTING REPORT =============================================================
# |        Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USD |   Tot Profit % |    Avg Duration |   Wins |   Draws |   Losses |
# |-------------+--------+----------------+----------------+------------------+----------------+-----------------+--------+---------+----------|
# |     BTC/USD |     12 |           0.92 |          11.02 |           55.129 |           2.20 |  1 day, 1:10:00 |      5 |       0 |        7 |
# |     ETH/USD |     12 |           2.13 |          25.50 |          127.549 |           5.10 |  1 day, 7:35:00 |      5 |       0 |        7 |
# |    PERP/USD |     11 |          -0.76 |          -8.35 |          -41.752 |          -1.67 |         6:27:00 |      3 |       0 |        8 |
# |     FTT/USD |     11 |           7.37 |          81.09 |          405.519 |          16.22 | 1 day, 15:11:00 |      5 |       0 |        6 |
# |     MOB/USD |     13 |           0.59 |           7.67 |           38.371 |           1.53 |        23:42:00 |      1 |       0 |       12 |
# |     RAY/USD |      5 |          -0.88 |          -4.42 |          -22.129 |          -0.88 |        13:12:00 |      1 |       0 |        4 |
# |    BULL/USD |      8 |          -0.18 |          -1.40 |           -7.003 |          -0.28 |         8:30:00 |      2 |       0 |        6 |
# |     SRM/USD |      6 |          -1.54 |          -9.24 |          -46.191 |          -1.85 |         4:30:00 |      0 |       0 |        6 |
# |    LINA/USD |     11 |          11.56 |         127.19 |          636.069 |          25.44 |         9:22:00 |      3 |       0 |        8 |
# | SXPBULL/USD |      9 |          -0.08 |          -0.75 |           -3.767 |          -0.15 |         3:00:00 |      1 |       0 |        8 |
# |    BTMX/USD |     12 |          38.27 |         459.20 |         2296.435 |          91.84 | 1 day, 23:15:00 |      5 |       0 |        7 |
# |    LINK/USD |     13 |          -0.68 |          -8.81 |          -44.053 |          -1.76 |        13:00:00 |      3 |       0 |       10 |
# |     LTC/USD |     12 |           0.46 |           5.54 |           27.728 |           1.11 |        11:40:00 |      5 |       0 |        7 |
# |     SXP/USD |     11 |           1.79 |          19.71 |           98.578 |           3.94 |        10:22:00 |      2 |       0 |        9 |
# |    AAVE/USD |      7 |          -1.38 |          -9.68 |          -48.425 |          -1.94 |         2:00:00 |      0 |       0 |        7 |
# |     BNB/USD |     12 |          -0.83 |          -9.95 |          -49.744 |          -1.99 |         4:30:00 |      2 |       0 |       10 |
# |     UNI/USD |     14 |          -0.62 |          -8.71 |          -43.580 |          -1.74 |         2:34:00 |      2 |       0 |       12 |
# | ADABULL/USD |      7 |          -1.54 |         -10.78 |          -53.889 |          -2.16 |         2:17:00 |      0 |       0 |        7 |
# |   SUSHI/USD |      9 |          -0.44 |          -4.00 |          -20.000 |          -0.80 |         5:40:00 |      1 |       0 |        8 |
# |     BCH/USD |     13 |          -0.10 |          -1.35 |           -6.744 |          -0.27 |         8:55:00 |      4 |       0 |        9 |
# |       TOTAL |    208 |           3.17 |         659.49 |         3298.103 |         131.90 |        14:42:00 |     50 |       0 |      158 |
# ====================================================== SELL REASON STATS ======================================================
# |   Sell Reason |   Sells |   Wins |   Draws |   Losses |   Avg Profit % |   Cum Profit % |   Tot Profit USD |   Tot Profit % |
# |---------------+---------+--------+---------+----------+----------------+----------------+------------------+----------------|
# |     stop_loss |     145 |      0 |       0 |      145 |          -1.54 |        -223.21 |         -1116.28 |         -44.64 |
# |   sell_signal |      63 |     50 |       0 |       13 |          14.01 |         882.7  |          4414.39 |         176.54 |
# ======================================================= LEFT OPEN TRADES REPORT ========================================================
# |   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USD |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |
# |--------+--------+----------------+----------------+------------------+----------------+----------------+--------+---------+----------|
# |  TOTAL |      0 |           0.00 |           0.00 |            0.000 |           0.00 |           0:00 |      0 |       0 |        0 |
# =============== SUMMARY METRICS ===============
# | Metric                | Value               |
# |-----------------------+---------------------|
# | Backtesting from      | 2021-02-01 00:00:00 |
# | Backtesting to        | 2021-03-24 22:00:00 |
# | Max open trades       | 5                   |
# |                       |                     |
# | Total trades          | 208                 |
# | Total Profit %        | 131.9%              |
# | Trades per day        | 4.08                |
# |                       |                     |
# | Best Pair             | BTMX/USD 459.2%     |
# | Worst Pair            | ADABULL/USD -10.78% |
# | Best trade            | BTMX/USD 307.52%    |
# | Worst trade           | AAVE/USD -1.54%     |
# | Best day              | 304.44%             |
# | Worst day             | -16.93%             |
# | Days win/draw/lose    | 20 / 4 / 28         |
# | Avg. Duration Winners | 2 days, 2:08:00     |
# | Avg. Duration Loser   | 3:30:00             |
# |                       |                     |
# | Abs Profit Min        | -60.400 USD         |
# | Abs Profit Max        | 3298.103 USD        |
# | Max Drawdown          | 39.83%              |
# | Drawdown Start        | 2021-02-27 01:00:00 |
# | Drawdown End          | 2021-03-10 02:00:00 |
# | Market change         | 229.59%             |
# ===============================================
