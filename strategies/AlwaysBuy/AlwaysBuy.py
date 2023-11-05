from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class AlwaysBuy(IStrategy):

    INTERFACE_VERSION = 3

    # ROI table:
    # fmt: off
    minimal_roi = {
        "0": 1,
        "100": 2,
        "200": 3,
        "300": -1
        }
    # fmt: on

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    # Buy hypers
    timeframe = "5m"
    use_exit_signal = False
    # #################### END OF RESULT PLACE ####################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, ["enter_long", "enter_tag"]] = (1, "entry_reason")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
