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
class TrixStrategy(IStrategy):
    """
    Sources : 
    Cripto Robot : https://www.youtube.com/watch?v=uE04UROWkjs&list=PLpJ7cz_wOtsrqEQpveLc2xKLjOBgy4NfA&index=4
    Github : https://github.com/CryptoRobotFr/TrueStrategy/blob/main/TrixStrategy/Trix_Complete_backtest.ipynb

    freqtrade backtesting -s TrixStrategy --timerange=20170817-20210919 --stake-amount unlimited -p ETH/USDT --config user_data/config_binance.json --enable-position-stacking --max-open-trades 1
    =============== SUMMARY METRICS ================
    | Metric                 | Value               |
    |------------------------+---------------------|
    | Backtesting from       | 2017-08-18 01:00:00 |
    | Backtesting to         | 2021-09-19 00:00:00 |
    | Max open trades        | 1                   |
    |                        |                     |
    | Total/Daily Avg Trades | 910 / 0.61          |
    | Starting balance       | 1000.000 USDT       |
    | Final balance          | 455182.175 USDT     |
    | Absolute profit        | 454182.175 USDT     |
    | Total profit %         | 45418.22%           |
    | Trades per day         | 0.61                |
    | Avg. daily profit %    | 30.44%              |
    | Avg. stake amount      | 89051.996 USDT      |
    | Total trade volume     | 81037315.991 USDT   |
    |                        |                     |
    | Best Pair              | ETH/USDT 716.78%    |
    | Worst Pair             | ETH/USDT 716.78%    |
    | Best trade             | ETH/USDT 40.29%     |
    | Worst trade            | ETH/USDT -17.67%    |
    | Best day               | 69170.563 USDT      |
    | Worst day              | -49006.144 USDT     |
    | Days win/draw/lose     | 427 / 613 / 449     |
    | Avg. Duration Winners  | 23:59:00            |
    | Avg. Duration Loser    | 13:56:00            |
    | Rejected Buy signals   | 17052               |
    |                        |                     |
    | Min balance            | 1064.302 USDT       |
    | Max balance            | 466421.313 USDT     |
    | Drawdown               | 37.23%              |
    | Drawdown               | 149281.307 USDT     |
    | Drawdown high          | 453758.871 USDT     |
    | Drawdown low           | 304477.564 USDT     |
    | Drawdown Start         | 2021-05-15 02:00:00 |
    | Drawdown End           | 2021-07-02 08:00:00 |
    | Market change          | 1028.84%            |
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
    startup_candle_count: int = 21

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
            'trix': {},
        },
        'subplots': {
            "STOCH RSI": {
                'stoch_rsi': {},
            },
            "TRIX": {
                'trix_pct': {},
                'trix_signal': {},
                'trix_histo': {},
            },
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
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(close=dataframe['close'], window=14, smooth1=3, smooth2=3)

        # Overlap Studies
        # ------------------------------------

        # -- Trix Indicator --
        trixLength = 9
        trixSignal = 21
        dataframe['trix'] = ta.trend.ema_indicator(ta.trend.ema_indicator(ta.trend.ema_indicator(close=dataframe['close'], window=trixLength), window=trixLength), window=trixLength)
        dataframe['trix_pct'] = dataframe['trix'].pct_change() * 100
        dataframe['trix_signal'] = ta.trend.sma_indicator(dataframe['trix_pct'],trixSignal)
        dataframe['trix_histo'] = dataframe['trix_pct'] - dataframe['trix_signal']

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
                (dataframe['trix_histo'] > 0) &
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
                (dataframe['trix_histo'] < 0) &
                (dataframe['stoch_rsi'] > self.sell_stoch_rsi.value) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe