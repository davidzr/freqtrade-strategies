# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
# import talib.abstract as ta
import ta


# This class is a sample. Feel free to customize it.
class CrossEMAStrategy(IStrategy):
    """
    Cross EMA + stoch RSI

    By Crypto Robot : https://www.youtube.com/watch?v=z9dbgvAYDuA
    GitHub : https://github.com/CryptoRobotFr/TrueStrategy/

    freqtrade backtesting -s CrossEMAStrategy --timerange=20170817-20210808 --stake-amount unlimited -p ETH/USDT --config user_data/config_binance.json --enable-position-stacking --max-open-trades 1
    =============== SUMMARY METRICS ================
    | Metric                 | Value               |
    |------------------------+---------------------|
    | Backtesting from       | 2017-08-19 05:00:00 |
    | Backtesting to         | 2021-08-08 00:00:00 |
    | Max open trades        | 1                   |
    |                        |                     |
    | Total/Daily Avg Trades | 250 / 0.17          |
    | Starting balance       | 1000.000 USDT       |
    | Final balance          | 135855.953 USDT     |
    | Absolute profit        | 134855.953 USDT     |
    | Total profit %         | 13485.6%            |
    | Trades per day         | 0.17                |
    | Avg. daily profit %    | 9.31%               |
    | Avg. stake amount      | 17038.974 USDT      |
    | Total trade volume     | 4259743.407 USDT    |
    |                        |                     |
    | Best Pair              | ETH/USDT 628.7%     |
    | Worst Pair             | ETH/USDT 628.7%     |
    | Best trade             | ETH/USDT 65.48%     |
    | Worst trade            | ETH/USDT -10.02%    |
    | Best day               | 50942.482 USDT      |
    | Worst day              | -9651.772 USDT      |
    | Days win/draw/lose     | 88 / 1193 / 156     |
    | Avg. Duration Winners  | 6 days, 2:44:00     |
    | Avg. Duration Loser    | 1 day, 10:12:00     |
    | Rejected Buy signals   | 18454               |
    |                        |                     |
    | Min balance            | 1015.748 USDT       |
    | Max balance            | 135855.953 USDT     |
    | Drawdown               | 45.55%              |
    | Drawdown               | 40840.978 USDT      |
    | Drawdown high          | 131200.572 USDT     |
    | Drawdown low           | 90359.594 USDT      |
    | Drawdown Start         | 2021-05-13 06:00:00 |
    | Drawdown End           | 2021-07-19 04:00:00 |
    | Market change          | 948.1%              |
    ================================================

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100 # inactive
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99 # inactive

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    buy_stoch_rsi = DecimalParameter(0.5, 1, decimals=3, default=0.8, space="buy")
    sell_stoch_rsi = DecimalParameter(0, 0.5, decimals=3, default=0.2, space="sell")

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 49 # EMA 48 + 1

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'ema28': {},
            'ema48': {}
        },
        'subplots': {
            "RSI": {
                'stoch_rsi': {}
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Momentum Indicators
        # ------------------------------------

        # # Stochastic RSI
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(dataframe['close'])

        # Overlap Studies
        # ------------------------------------

        # # EMA - Exponential Moving Average
        dataframe['ema28']=ta.trend.ema_indicator(dataframe['close'], 28)
        dataframe['ema48']=ta.trend.ema_indicator(dataframe['close'], 48)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['ema28'] > dataframe['ema48']) &
                (dataframe['stoch_rsi'] < self.buy_stoch_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
            (
                (dataframe['ema28'] < dataframe['ema48']) &
                (dataframe['stoch_rsi'] > self.sell_stoch_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe