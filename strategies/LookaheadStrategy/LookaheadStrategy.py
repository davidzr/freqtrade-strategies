from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class LookaheadStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # Buy hyperspace params:
    buy_params = {
        "buy_fast": 2,
        "buy_push": 1.022,
        "buy_shift": -8,
        "buy_slow": 16,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_fast": 34,
        "sell_push": 0.458,
        "sell_shift": -8,
        "sell_slow": 44,
    }

    # ROI table:
    # fmt: off
    minimal_roi = {
        "0": 0.166,
        "44": 0.012,
        "59": 0
        }
    # fmt: on

    # Stoploss:
    stoploss = -0.194

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    # Buy hypers
    timeframe = "5m"
    # #################### END OF RESULT PLACE ####################

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        # EMA
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)

        return informative_1h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        dataframe["buy_ema_fast"] = ta.SMA(dataframe, timeperiod=self.buy_params["buy_fast"])
        dataframe["buy_ema_slow"] = ta.SMA(dataframe, timeperiod=self.buy_params["buy_slow"])

        dataframe["sell_ema_fast"] = ta.SMA(dataframe, timeperiod=self.sell_params["sell_fast"])
        dataframe["sell_ema_slow"] = ta.SMA(dataframe, timeperiod=self.sell_params["sell_slow"])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            qtpylib.crossed_above(
                dataframe["buy_ema_fast"].shift(self.buy_params["buy_shift"]),
                dataframe["buy_ema_slow"].shift(self.buy_params["buy_shift"])
                * self.buy_params["buy_push"],
            )
            & (dataframe["close"] > dataframe["ema_50_1h"])
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]] = (
                1,
                "buy_reason",
            )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            qtpylib.crossed_below(
                dataframe["sell_ema_fast"].shift(self.sell_params["sell_shift"]),
                dataframe["sell_ema_slow"].shift(self.sell_params["sell_shift"])
                * self.sell_params["sell_push"],
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]] = (
                1,
                "some_exit_tag",
            )
        return dataframe
