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
import ta


# This class is a sample. Feel free to customize it.
class AlligatorStrategy(IStrategy):
    """
    Sources
    Crypto Robot : https://www.youtube.com/watch?v=tHYs5135jUA
    Github : https://github.com/CryptoRobotFr/TrueStrategy/blob/main/AligatorStrategy/Aligator_Strategy_backtest.ipynb

    freqtrade backtesting -s AlligatorStrategy --timerange=20200903-20210826 --stake-amount unlimited -p EGLD/USDT --config user_data/config_binance.json --enable-position-stacking
    =============== SUMMARY METRICS ================
    | Metric                 | Value               |
    |------------------------+---------------------|
    | Backtesting from       | 2020-09-11 11:00:00 |
    | Backtesting to         | 2021-08-26 00:00:00 |
    | Max open trades        | 1                   |
    |                        |                     |
    | Total/Daily Avg Trades | 258 / 0.74          |
    | Starting balance       | 1000.000 USDT       |
    | Final balance          | 14245.054 USDT      |
    | Absolute profit        | 13245.054 USDT      |
    | Total profit %         | 1324.51%            |
    | Trades per day         | 0.74                |
    | Avg. daily profit %    | 3.81%               |
    | Avg. stake amount      | 829.268 USDT        |
    | Total trade volume     | 213951.164 USDT     |
    |                        |                     |
    | Best Pair              | EGLD/USDT 4287.66%  |
    | Worst Pair             | EGLD/USDT 4287.66%  |
    | Best trade             | EGLD/USDT 305.28%   |
    | Worst trade            | EGLD/USDT -10.69%   |
    | Best day               | 7078.313 USDT       |
    | Worst day              | -1052.860 USDT      |
    | Days win/draw/lose     | 8 / 275 / 19        |
    | Avg. Duration Winners  | 9 days, 18:58:00    |
    | Avg. Duration Loser    | 2 days, 9:00:00     |
    | Rejected Buy signals   | 3045                |
    |                        |                     |
    | Min balance            | 905.332 USDT        |
    | Max balance            | 16789.797 USDT      |
    | Drawdown               | 169.59%             |
    | Drawdown               | 2544.743 USDT       |
    | Drawdown high          | 15789.797 USDT      |
    | Drawdown low           | 13245.054 USDT      |
    | Drawdown Start         | 2021-08-12 18:00:00 |
    | Drawdown End           | 2021-08-24 18:00:00 |
    | Market change          | 497.97%             |
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
    buy_stoch_rsi = DecimalParameter(0.5, 1, decimals=3, default=0.82, space="buy")
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
    startup_candle_count = 200 # EMA200

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
            'ema7':{},
            'ema30':{},
            'ema50':{},
            'ema100':{},
            'ema121':{},
            'ema200':{}
        },
        'subplots': {
            "STOCH RSI": {
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

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # # Stochastic RSI
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(close=dataframe['close'], window=14, smooth1=3, smooth2=3) #Non moyenné 

        # Overlap Studies
        # ------------------------------------

        # # EMA - Exponential Moving Average
        dataframe['ema7']=ta.trend.ema_indicator(close=dataframe['close'], window=7)
        dataframe['ema30']=ta.trend.ema_indicator(close=dataframe['close'], window=30)
        dataframe['ema50']=ta.trend.ema_indicator(close=dataframe['close'], window=50)
        dataframe['ema100']=ta.trend.ema_indicator(close=dataframe['close'], window=100)
        dataframe['ema121']=ta.trend.ema_indicator(close=dataframe['close'], window=121)
        dataframe['ema200']=ta.trend.ema_indicator(close=dataframe['close'], window=200)

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
                (dataframe['ema7'] > dataframe['ema30']) &
                (dataframe['ema30'] > dataframe['ema50']) &
                (dataframe['ema50'] > dataframe['ema100']) &
                (dataframe['ema100'] > dataframe['ema121']) &
                (dataframe['ema121'] > dataframe['ema200']) &
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
                (dataframe['ema121'] > dataframe['ema7']) &
                (dataframe['stoch_rsi'] > self.sell_stoch_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe