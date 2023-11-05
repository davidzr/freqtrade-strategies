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
import pandas_ta as pda


# This class is a sample. Feel free to customize it.
class SupertrendStrategy(IStrategy):
    """
    Sources : 
    Cripto Robot : https://www.youtube.com/watch?v=rl00g3-Iv5A
    Github : https://github.com/CryptoRobotFr/TrueStrategy/tree/main/3SuperTrend

    freqtrade backtesting -s SupertrendStrategy --timerange=20180417-20210818 --stake-amount unlimited -p ADA/USDT --config user_data/config_binance.json --enable-position-stacking --max-open-trades 1
    =============== SUMMARY METRICS ================
    | Metric                 | Value               |
    |------------------------+---------------------|
    | Backtesting from       | 2018-04-18 10:00:00 |
    | Backtesting to         | 2021-08-18 00:00:00 |
    | Max open trades        | 1                   |
    |                        |                     |
    | Total/Daily Avg Trades | 202 / 0.17          |
    | Starting balance       | 1000.000 USDT       |
    | Final balance          | 41040.541 USDT      |
    | Absolute profit        | 40040.541 USDT      |
    | Total profit %         | 4004.05%            |
    | Trades per day         | 0.17                |
    | Avg. daily profit %    | 3.29%               |
    | Avg. stake amount      | 7215.587 USDT       |
    | Total trade volume     | 1457548.658 USDT    |
    |                        |                     |
    | Best Pair              | ADA/USDT 540.88%    |
    | Worst Pair             | ADA/USDT 540.88%    |
    | Best trade             | ADA/USDT 148.95%    |
    | Worst trade            | ADA/USDT -18.24%    |
    | Best day               | 17171.430 USDT      |
    | Worst day              | -7253.003 USDT      |
    | Days win/draw/lose     | 81 / 1013 / 121     |
    | Avg. Duration Winners  | 4 days, 10:40:00    |
    | Avg. Duration Loser    | 1 day, 21:07:00     |
    | Rejected Buy signals   | 14100               |
    |                        |                     |
    | Min balance            | 835.821 USDT        |
    | Max balance            | 41123.976 USDT      |
    | Drawdown               | 59.98%              |
    | Drawdown               | 9754.773 USDT       |
    | Drawdown high          | 39126.935 USDT      |
    | Drawdown low           | 29372.162 USDT      |
    | Drawdown Start         | 2021-05-16 22:00:00 |
    | Drawdown End           | 2021-07-12 21:00:00 |
    | Market change          | 624.61%             |
    ================================================

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

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
    startup_candle_count: int = 90

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
            'ema90':{},
            'supertrend_1':{},
            'supertrend_2':{},
            'supertrend_3':{},
        },
        'subplots': {
            "SUPERTREND DIRECTION": {
                'supertrend_direction_1': {},
                'supertrend_direction_2': {},
                'supertrend_direction_3': {},
            },
            "STOCH RSI": {
                'stoch_rsi': {},
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
        dataframe['stoch_rsi']=ta.momentum.stochrsi(dataframe['close'])

        # Overlap Studies
        # ------------------------------------

        # # EMA - Exponential Moving Average
        dataframe['ema90'] = ta.trend.ema_indicator(dataframe['close'], 90)

        # Supertrend
        supertrend_length = 20
        supertrend_multiplier = 3.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length, multiplier=supertrend_multiplier)
        dataframe['supertrend_1'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_1'] = superTrend['SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        
        supertrend_length = 20
        supertrend_multiplier = 4.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length, multiplier=supertrend_multiplier)
        dataframe['supertrend_2'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_2'] = superTrend['SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        
        supertrend_length = 40
        supertrend_multiplier = 8.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length, multiplier=supertrend_multiplier)
        dataframe['supertrend_3'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_3'] = superTrend['SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]

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
                ((dataframe['supertrend_direction_1'] + dataframe['supertrend_direction_2'] + dataframe['supertrend_direction_3']) >= 1) &
                (dataframe['stoch_rsi'] < self.buy_stoch_rsi.value) &
                (dataframe['close'] > dataframe['ema90']) &
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
                ((dataframe['supertrend_direction_1'] + dataframe['supertrend_direction_2'] + dataframe['supertrend_direction_3']) < 1) &
                (dataframe['stoch_rsi'] > self.sell_stoch_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe