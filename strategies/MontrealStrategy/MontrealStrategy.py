# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MontrealStrategy(IStrategy):
    """
    Strategy developed in udemy course, really sucks actually
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 0.31651, "79": 0.07068, "147": 0.04049, "345": 0}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.28646

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal ticker interval for the strategy.
    timeframe = "15m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Optional order time in force.
    order_time_in_force = {"buy": "gtc", "sell": "gtc"}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),("BTC/USDT", "15m")]
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
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger1 = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=1
        )
        dataframe["bb1_lowerband"] = bollinger1["lower"]
        dataframe["bb1_middleband"] = bollinger1["mid"]
        dataframe["bb1_upperband"] = bollinger1["upper"]

        bollinger2 = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb2_lowerband"] = bollinger2["lower"]
        dataframe["bb2_middleband"] = bollinger2["mid"]
        dataframe["bb2_upperband"] = bollinger2["upper"]

        bollinger4 = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=4
        )
        dataframe["bb4_lowerband"] = bollinger4["lower"]
        dataframe["bb4_middleband"] = bollinger4["mid"]
        dataframe["bb4_upperband"] = bollinger4["upper"]

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
                (dataframe["rsi"] > 30)
                & (dataframe["close"] < dataframe["bb2_lowerband"])
                & (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1

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
                (dataframe["close"] > dataframe["bb2_upperband"])
                & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1

        return dataframe
