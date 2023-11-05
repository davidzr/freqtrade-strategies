import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import (
    merge_informative_pair,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)
from functools import reduce
import logging

# -------------------------------------------------------------------------------------------------
# --- logger for parameter merging output, only remove if you remove it further down too! ---------
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------------------------------------


class BcmbigzDevelop(IStrategy):
    INTERFACE_VERSION = 2

    # minimal_roi = {"0": 0.038, "20": 0.028, "40": 0.02, "60": 0.015, "180": 0.018, }
    # minimal_roi = {"0": 0.038, "20": 0.028, "40": 0.02, "60": 0.015, "180": 0.018, }
    minimal_roi = {"0": 0.20, "38": 0.074, "78": 0.025, "194": 0}

    stoploss = -0.99  # effectively disabled.

    timeframe = "5m"
    inf_1h = "1h"

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = (
        0.001  # it doesn't meant anything, just to guarantee there is a minimal profit.
    )
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "buy": "market",
        "sell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    buy_params = {
        "buy_minimum_conditions": 1,
        #############
        # Enable/Disable conditions
        "bzv7_buy_condition_0_enable": True,
        "bzv7_buy_condition_1_enable": True,
        "bzv7_buy_condition_2_enable": True,
        "bzv7_buy_condition_3_enable": True,
        "bzv7_buy_condition_4_enable": True,
        "bzv7_buy_condition_5_enable": True,
        "bzv7_buy_condition_6_enable": True,
        "bzv7_buy_condition_7_enable": True,
        "bzv7_buy_condition_8_enable": True,
        "bzv7_buy_condition_9_enable": True,
        "bzv7_buy_condition_10_enable": False,
        "bzv7_buy_condition_11_enable": False,
        "bzv7_buy_condition_12_enable": True,
        "bzv7_buy_condition_13_enable": False,
        "v6_buy_condition_0_enable": False,
        "v6_buy_condition_1_enable": True,
        "v6_buy_condition_2_enable": True,
        "v6_buy_condition_3_enable": True,
        "v8_buy_condition_0_enable": True,
        "v8_buy_condition_1_enable": False,
        "v8_buy_condition_2_enable": True,
        "v8_buy_condition_3_enable": False,
        "v8_buy_condition_4_enable": True,
    }
    sell_params = {
        #############
        # Enable/Disable conditions
        "bzv7_sell_condition_0_enable": False,
        "v8_sell_condition_0_enable": True,
        "v8_sell_condition_1_enable": True,
    }

    # if you want to see which buy conditions were met
    # or if there is an trade exit override due to high RSI set to True
    # logger will output the buy and trade exit conditions
    cust_log_verbose = False
    ############################################################################
    # minimum conditions to match in buy
    buy_minimum_conditions = IntParameter(
        1, 2, default=1, space="buy", optimize=False, load=True
    )
    #  Strategy: BigZ07
    # Buy HyperParam
    bzv7_buy_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_1_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_2_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_3_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_4_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_5_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_6_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_7_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_8_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_9_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_10_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_11_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_12_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_buy_condition_13_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    bzv7_sell_condition_0_enable = CategoricalParameter(
        [True, False], default=False, space="sell", optimize=False, load=True
    )

    bzv7_buy_bb20_close_bblowerband_safe_1 = DecimalParameter(
        0.7, 1.1, default=0.989, space="buy", optimize=False, load=True
    )
    bzv7_buy_bb20_close_bblowerband_safe_2 = DecimalParameter(
        0.7, 1.1, default=0.982, space="buy", optimize=False, load=True
    )

    bzv7_buy_volume_pump_1 = DecimalParameter(
        0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_volume_drop_1 = DecimalParameter(
        1, 10, default=3.8, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_volume_drop_2 = DecimalParameter(
        1, 10, default=3, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_volume_drop_3 = DecimalParameter(
        1, 10, default=2.7, space="buy", decimals=1, optimize=False, load=True
    )

    bzv7_buy_rsi_1h_1 = DecimalParameter(
        10.0, 40.0, default=16.5, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_1h_2 = DecimalParameter(
        10.0, 40.0, default=15.0, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_1h_3 = DecimalParameter(
        10.0, 40.0, default=20.0, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_1h_4 = DecimalParameter(
        10.0, 40.0, default=35.0, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_1h_5 = DecimalParameter(
        10.0, 60.0, default=39.0, space="buy", decimals=1, optimize=False, load=True
    )

    bzv7_buy_rsi_1 = DecimalParameter(
        10.0, 40.0, default=28.0, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_2 = DecimalParameter(
        7.0, 40.0, default=10.0, space="buy", decimals=1, optimize=False, load=True
    )
    bzv7_buy_rsi_3 = DecimalParameter(
        7.0, 40.0, default=14.2, space="buy", decimals=1, optimize=False, load=True
    )

    bzv7_buy_macd_1 = DecimalParameter(
        0.01, 0.09, default=0.02, space="buy", decimals=2, optimize=False, load=True
    )
    bzv7_buy_macd_2 = DecimalParameter(
        0.01, 0.09, default=0.03, space="buy", decimals=2, optimize=False, load=True
    )

    #  Strategy: CombinedBinHClucAndMADV6
    v6_buy_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v6_buy_condition_1_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v6_buy_condition_2_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v6_buy_condition_3_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    # Strategy:  CombinedBinHClucV8
    v8_buy_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v8_buy_condition_1_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v8_buy_condition_2_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v8_buy_condition_3_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v8_buy_condition_4_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    v8_sell_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=False, load=True
    )
    v8_sell_condition_1_enable = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=False, load=True
    )

    v8_sell_rsi_main = DecimalParameter(
        72.0, 90.0, default=80, space="sell", decimals=2, optimize=False, load=True
    )
    buy_dip_threshold_0 = DecimalParameter(
        0.001, 0.1, default=0.015, space="buy", decimals=3, optimize=False, load=True
    )
    buy_dip_threshold_1 = DecimalParameter(
        0.08, 0.2, default=0.12, space="buy", decimals=2, optimize=False, load=True
    )
    buy_dip_threshold_2 = DecimalParameter(
        0.02, 0.4, default=0.28, space="buy", decimals=2, optimize=False, load=True
    )
    buy_dip_threshold_3 = DecimalParameter(
        0.25, 0.44, default=0.36, space="buy", decimals=2, optimize=False, load=True
    )
    buy_bb40_bbdelta_close = DecimalParameter(
        0.005, 0.04, default=0.031, space="buy", optimize=False, load=True
    )
    buy_bb40_closedelta_close = DecimalParameter(
        0.01, 0.03, default=0.021, space="buy", optimize=False, load=True
    )
    buy_bb40_tail_bbdelta = DecimalParameter(
        0.2, 0.4, default=0.264, space="buy", optimize=False, load=True
    )
    buy_bb20_close_bblowerband = DecimalParameter(
        0.8, 1.1, default=0.992, space="buy", optimize=False, load=True
    )
    buy_bb20_volume = IntParameter(
        18, 36, default=29, space="buy", optimize=False, load=True
    )
    buy_rsi_diff = DecimalParameter(
        34.0, 60.0, default=50.48, space="buy", decimals=2, optimize=False, load=True
    )
    buy_min_inc = DecimalParameter(
        0.005, 0.05, default=0.01, space="buy", decimals=2, optimize=False, load=True
    )
    buy_rsi_1h = DecimalParameter(
        40.0, 70.0, default=67.0, space="buy", decimals=2, optimize=False, load=True
    )
    buy_rsi = DecimalParameter(
        30.0, 40.0, default=38.5, space="buy", decimals=2, optimize=False, load=True
    )
    buy_mfi = DecimalParameter(
        36.0, 65.0, default=36.0, space="buy", decimals=2, optimize=False, load=True
    )
    buy_volume_1 = DecimalParameter(
        1.0, 10.0, default=2.0, space="buy", decimals=2, optimize=False, load=True
    )
    buy_ema_open_mult_1 = DecimalParameter(
        0.01, 0.05, default=0.02, space="buy", decimals=3, optimize=False, load=True
    )
    sell_custom_roi_profit_1 = DecimalParameter(
        0.01, 0.03, default=0.01, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_rsi_1 = DecimalParameter(
        40.0, 56.0, default=50, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_profit_2 = DecimalParameter(
        0.01, 0.20, default=0.04, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_rsi_2 = DecimalParameter(
        42.0, 56.0, default=50, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_profit_3 = DecimalParameter(
        0.15, 0.30, default=0.08, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_rsi_3 = DecimalParameter(
        44.0, 58.0, default=56, space="sell", decimals=2, optimize=False, load=True
    )
    sell_custom_roi_profit_4 = DecimalParameter(
        0.3, 0.7, default=0.14, space="sell", decimals=2, optimize=True, load=True
    )
    sell_custom_roi_rsi_4 = DecimalParameter(
        44.0, 60.0, default=58, space="sell", decimals=2, optimize=False, load=True
    )
    sell_custom_roi_profit_5 = DecimalParameter(
        0.01, 0.1, default=0.04, space="sell", decimals=2, optimize=True, load=True
    )
    sell_trail_profit_min_1 = DecimalParameter(
        0.1, 0.25, default=0.1, space="sell", decimals=3, optimize=True, load=True
    )
    sell_trail_profit_max_1 = DecimalParameter(
        0.3, 0.5, default=0.4, space="sell", decimals=2, optimize=True, load=True
    )
    sell_trail_down_1 = DecimalParameter(
        0.04, 0.1, default=0.03, space="sell", decimals=3, optimize=True, load=True
    )
    sell_trail_profit_min_2 = DecimalParameter(
        0.01, 0.1, default=0.02, space="sell", decimals=3, optimize=True, load=True
    )
    sell_trail_profit_max_2 = DecimalParameter(
        0.08, 0.25, default=0.1, space="sell", decimals=2, optimize=True, load=True
    )
    sell_trail_down_2 = DecimalParameter(
        0.04, 0.2, default=0.015, space="sell", decimals=3, optimize=True, load=True
    )
    sell_custom_stoploss_1 = DecimalParameter(
        -0.15, -0.03, default=-0.05, space="sell", decimals=2, optimize=True, load=True
    )

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        **kwargs,
    ) -> bool:

        return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        last_candle_1 = dataframe.iloc[-2].squeeze()

        if sell_reason == "roi":
            # Looks like we can get a little have more
            if (last_candle["cmf"] < -0.1) & (
                last_candle["close"] > last_candle["ema_200_1h"]
            ):
                return False

        return True

    def custom_sell(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):

        # return False

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        last_candle_2 = dataframe.iloc[-2].squeeze()

        if last_candle is not None:
            # if (last_candle['high'] > last_candle['bb_upperband']) & (last_candle['volume'] > (last_candle_2['volume'] * 1.5)):
            # return 'sell_signal_1'

            if (current_profit > self.sell_custom_roi_profit_4.value) & (
                last_candle["rsi"] < self.sell_custom_roi_rsi_4.value
            ):
                return "roi_target_4"
            elif (current_profit > self.sell_custom_roi_profit_3.value) & (
                last_candle["rsi"] < self.sell_custom_roi_rsi_3.value
            ):
                return "roi_target_3"
            elif (current_profit > self.sell_custom_roi_profit_2.value) & (
                last_candle["rsi"] < self.sell_custom_roi_rsi_2.value
            ):
                return "roi_target_2"
            elif (current_profit > self.sell_custom_roi_profit_1.value) & (
                last_candle["rsi"] < self.sell_custom_roi_rsi_1.value
            ):
                return "roi_target_1"
            elif (
                (current_profit > 0)
                & (current_profit < self.sell_custom_roi_profit_5.value)
                & (last_candle["sma_200_dec"])
            ):
                return "roi_target_5"

            elif (
                (current_profit > self.sell_trail_profit_min_1.value)
                & (current_profit < self.sell_trail_profit_max_1.value)
                & (
                    ((trade.max_rate - trade.open_rate) / 100)
                    > (current_profit + self.sell_trail_down_1.value)
                )
            ):
                return "trail_target_1"
            elif (
                (current_profit > self.sell_trail_profit_min_2.value)
                & (current_profit < self.sell_trail_profit_max_2.value)
                & (
                    ((trade.max_rate - trade.open_rate) / 100)
                    > (current_profit + self.sell_trail_down_2.value)
                )
            ):
                return "trail_target_2"

        return False

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        # Manage losing trades and open room for better ones.

        if current_profit > 0:
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=240)

            # Trade open more then 60 minutes. For this strategy it's means -> loss
            # Let's try to minimize the loss

            if current_time > trade_time_50:

                try:
                    number_of_candle_shift = int(
                        (current_time - trade_time_50).total_seconds() / 300
                    )
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    # We are at bottom. Wait...
                    if candle["rsi_1h"] < 35:
                        return 0.99

                    if candle["open"] > candle["ema_200"]:
                        return 0.1

                    # Are we still sinking?
                    if current_rate * 1.025 < candle["open"]:
                        return 0.01

                except IndexError as error:

                    # Whoops, set stoploss at 10%
                    return 0.1

        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.inf_1h
        )
        # EMA
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        informative_1h["bb_lowerband"] = bollinger["lower"]
        informative_1h["bb_middleband"] = bollinger["mid"]
        informative_1h["bb_upperband"] = bollinger["upper"]

        # for BinClucMad
        informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["sma_200"] = ta.SMA(
            informative_1h, timeperiod=200
        )  # for BinClucMad
        informative_1h["sma_200_dec"] = informative_1h["sma_200"] < informative_1h[
            "sma_200"
        ].shift(
            20
        )  # for BinClucMad
        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h
        informative_1h["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=48).mean()

        # EMA
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)

        # MACD
        dataframe["macd"], dataframe["signal"], dataframe["hist"] = ta.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Chaikin A/D Oscillator
        dataframe["mfv"] = MFV(dataframe)
        dataframe["cmf"] = (
            dataframe["mfv"].rolling(20).sum() / dataframe["volume"].rolling(20).sum()
        )

        # for BinClucMad
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)  # for BinClucMad
        dataframe["sma_5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()
        dataframe["closedelta"] = (
            dataframe["close"] - dataframe["close"].shift()
        ).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True
        )

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        # reset additional dataframe rows
        dataframe.loc[:, "bzv7_buy_condition_0_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_1_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_2_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_3_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_4_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_5_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_6_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_7_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_8_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_9_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_10_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_11_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_12_enable"] = False
        dataframe.loc[:, "bzv7_buy_condition_13_enable"] = False
        dataframe.loc[:, "v6_buy_condition_0_enable"] = False
        dataframe.loc[:, "v6_buy_condition_1_enable"] = False
        dataframe.loc[:, "v6_buy_condition_2_enable"] = False
        dataframe.loc[:, "v6_buy_condition_3_enable"] = False
        dataframe.loc[:, "v8_buy_condition_0_enable"] = False
        dataframe.loc[:, "v8_buy_condition_1_enable"] = False
        dataframe.loc[:, "v8_buy_condition_2_enable"] = False
        dataframe.loc[:, "v8_buy_condition_3_enable"] = False
        dataframe.loc[:, "v8_buy_condition_4_enable"] = False
        dataframe.loc[:, "conditions_count"] = 0

        # Strategy: v8 BUY conditions
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_2.value
                )
                & dataframe["lower"].shift().gt(0)
                & dataframe["bbdelta"].gt(
                    dataframe["close"] * self.buy_bb40_bbdelta_close.value
                )
                & dataframe["closedelta"].gt(
                    dataframe["close"] * self.buy_bb40_closedelta_close.value
                )
                & dataframe["tail"].lt(
                    dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta.value
                )
                & dataframe["close"].lt(dataframe["lower"].shift())
                & dataframe["close"].le(dataframe["close"].shift())
                & (self.v8_buy_condition_0_enable.value == True)
            ),
            "v8_buy_condition_0_enable",
        ] = 1

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_2.value
                )
                & (dataframe["close"] < dataframe["ema_50"])
                & (
                    dataframe["close"]
                    < self.buy_bb20_close_bblowerband.value * dataframe["bb_lowerband"]
                )
                & (
                    dataframe["volume"]
                    < (
                        dataframe["volume_mean_slow"].shift(1)
                        * self.buy_bb20_volume.value
                    )
                )
                & (self.v8_buy_condition_1_enable.value == True)
            ),
            "v8_buy_condition_1_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["sma_5"])
                & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_2.value
                )
                & (
                    (
                        (dataframe["open"].rolling(144).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_3.value
                )
                & (dataframe["rsi"] < dataframe["rsi_1h"] - self.buy_rsi_diff.value)
                & (self.v8_buy_condition_2_enable.value == True)
            ),
            "v8_buy_condition_2_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["sma_200"] > dataframe["sma_200"].shift(20))
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(16))
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_2.value
                )
                & (
                    (
                        (dataframe["open"].rolling(144).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_3.value
                )
                & (
                    (
                        (dataframe["open"].rolling(24).min() - dataframe["close"])
                        / dataframe["close"]
                    )
                    > self.buy_min_inc.value
                )
                & (dataframe["rsi_1h"] > self.buy_rsi_1h.value)
                & (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["mfi"] < self.buy_mfi.value)
                & (self.v8_buy_condition_3_enable.value == True)
            ),
            "v8_buy_condition_3_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_100_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_2.value
                )
                & (
                    (
                        (dataframe["open"].rolling(144).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.buy_dip_threshold_3.value
                )
                & (
                    dataframe["volume"].rolling(4).mean() * self.buy_volume_1.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_1.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (self.v8_buy_condition_4_enable.value == True)
            ),
            "v8_buy_condition_4_enable",
        ] = 1

        # Strategy: v6 BUY conditions
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["close"] < 0.99 * dataframe["bb_lowerband"])
                & (
                    (
                        dataframe["volume"]
                        < (dataframe["volume_mean_slow"].shift(1) * 21)
                    )
                    | (
                        dataframe["volume_mean_slow"]
                        > (dataframe["volume_mean_slow"].shift(30) * 0.4)
                    )
                )
                & (self.v6_buy_condition_0_enable.value == True)
            ),
            "v6_buy_condition_0_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["close"] < 0.975 * dataframe["bb_lowerband"])
                & (
                    (
                        dataframe["volume"]
                        < (dataframe["volume_mean_slow"].shift(1) * 20)
                    )
                    | (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * 0.4
                    )
                )
                & (dataframe["rsi_1h"] < 15)  # Don't buy if someone drop the market.
                & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                & (self.v6_buy_condition_1_enable.value == True)
            ),
            "v6_buy_condition_1_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * 0.02)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (
                    (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                    | (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * 0.4
                    )
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (self.v6_buy_condition_2_enable.value == True)
            ),
            "v6_buy_condition_2_enable",
        ] = 1
        dataframe.loc[
            (
                (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * 0.03)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (self.v6_buy_condition_3_enable.value == True)
            ),
            "v6_buy_condition_3_enable",
        ] = 1

        # Strategy: BigZ07 BUY conditions
        dataframe.loc[
            (
                self.bzv7_buy_condition_0_enable.value &

                (dataframe['close'] > dataframe['ema_200']) &

                (dataframe['rsi'] < 30) &
                (dataframe['close'] * 1.024 < dataframe['open'].shift(3)) &
                (dataframe['rsi_1h'] < 71) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_0_enable",
        ] = 1
        dataframe.loc[
            (
                self.bzv7_buy_condition_1_enable.value &

                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['close'] <  dataframe['bb_lowerband'] * self.bzv7_buy_bb20_close_bblowerband_safe_1.value) &
                (dataframe['rsi_1h'] < 69) &
                (dataframe['open'] > dataframe['close']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_1_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_2_enable.value &

                (dataframe['close'] > dataframe['ema_200']) &

                (dataframe['close'] < dataframe['bb_lowerband'] *  self.bzv7_buy_bb20_close_bblowerband_safe_2.value) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_2_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_3_enable.value &

                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['rsi'] < self.bzv7_buy_rsi_3.value) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_3.value)) &

                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_3_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_4_enable.value &

                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_1.value) &

                (dataframe['close'] < dataframe['bb_lowerband']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_4_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_5_enable.value &

                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.bzv7_buy_macd_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'])) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            "bzv7_buy_condition_5_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_6_enable.value &

                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_5.value) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.bzv7_buy_macd_2.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'])) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_6_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_7_enable.value &

                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_2.value) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.bzv7_buy_macd_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_7_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_8_enable.value &

                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_3.value) &
                (dataframe['rsi'] < self.bzv7_buy_rsi_1.value) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_8_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_9_enable.value &

                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_4.value) &
                (dataframe['rsi'] < self.bzv7_buy_rsi_2.value) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_9_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_10_enable.value
                & (dataframe["rsi_1h"] < self.bzv7_buy_rsi_1h_4.value)
                & (dataframe["close_1h"] < dataframe["bb_lowerband_1h"])
                & (dataframe["hist"] > 0)
                & (dataframe["hist"].shift(2) < 0)
                & (dataframe["rsi"] < 40.5)
                & (dataframe["hist"] > dataframe["close"] * 0.0012)
                & (dataframe["open"] < dataframe["close"])
                & (dataframe["volume"] > 0)
            ),
            "bzv7_buy_condition_10_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_11_enable.value &

                ((dataframe['high'] - dataframe['low']) < dataframe['open']/100) &
                (dataframe['open'] < dataframe['close']) &
                ((dataframe['high'].shift() - dataframe['low'].shift()) < dataframe['open'].shift()/100) &
                ((dataframe['high'].shift(2) - dataframe['low'].shift(2)) < dataframe['open'].shift(2)/100) &
                ((dataframe['high'].shift(3) - dataframe['low'].shift(3)) < dataframe['open'].shift(3)/100) &
                ((dataframe['high'].shift(4) - dataframe['low'].shift(4)) < dataframe['open'].shift(4)/100) &
                ((dataframe['high'].shift(5) - dataframe['low'].shift(5)) < dataframe['open'].shift(5)/100) &
                ((dataframe['high'].shift(6) - dataframe['low'].shift(6)) < dataframe['open'].shift(6)/100) &
                ((dataframe['high'].shift(7) - dataframe['low'].shift(7)) < dataframe['open'].shift(7)/100) &
                ((dataframe['high'].shift(8) - dataframe['low'].shift(8)) < dataframe['open'].shift(8)/100) &
                ((dataframe['high'].shift(9) - dataframe['low'].shift(9)) < dataframe['open'].shift(9)/100) &
                (dataframe['bb_middleband'] > dataframe['bb_middleband'].shift(9) * 1.005) &
                (dataframe['rsi'] < 68) &

                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            "bzv7_buy_condition_11_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_12_enable.value &

                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['close'] < dataframe['bb_lowerband'] * 0.993) &
                (dataframe['low'] < dataframe['bb_lowerband'] * 0.985) &
                (dataframe['close'].shift() > dataframe['bb_lowerband']) &
                (dataframe['rsi_1h'] < 72.8) &
                (dataframe['open'] > dataframe['close']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

                (dataframe['volume'] > 0)
            ),
            "bzv7_buy_condition_12_enable",
        ] = 1

        dataframe.loc[
            (
                self.bzv7_buy_condition_13_enable.value
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["cmf"] < -0.435)
                & (dataframe["rsi"] < 22)
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(48)
                    * self.bzv7_buy_volume_pump_1.value
                )
                & (
                    dataframe["volume_mean_slow"] * self.bzv7_buy_volume_pump_1.value
                    < dataframe["volume_mean_slow"].shift(48)
                )
                & (dataframe["volume"] > 0)
            ),
            "bzv7_buy_condition_13_enable",
        ] = 1

        # count the amount of conditions met
        dataframe.loc[:, "conditions_count"] = (
            dataframe["bzv7_buy_condition_0_enable"].astype(int)
            + dataframe["bzv7_buy_condition_1_enable"].astype(int)
            + dataframe["bzv7_buy_condition_2_enable"].astype(int)
            + dataframe["bzv7_buy_condition_3_enable"].astype(int)
            + dataframe["bzv7_buy_condition_4_enable"].astype(int)
            + dataframe["bzv7_buy_condition_5_enable"].astype(int)
            + dataframe["bzv7_buy_condition_6_enable"].astype(int)
            + dataframe["bzv7_buy_condition_7_enable"].astype(int)
            + dataframe["bzv7_buy_condition_8_enable"].astype(int)
            + dataframe["bzv7_buy_condition_9_enable"].astype(int)
            + dataframe["bzv7_buy_condition_10_enable"].astype(int)
            + dataframe["bzv7_buy_condition_11_enable"].astype(int)
            + dataframe["bzv7_buy_condition_12_enable"].astype(int)
            + dataframe["bzv7_buy_condition_13_enable"].astype(int)
            + dataframe["v6_buy_condition_0_enable"].astype(int)
            + dataframe["v6_buy_condition_1_enable"].astype(int)
            + dataframe["v6_buy_condition_2_enable"].astype(int)
            + dataframe["v6_buy_condition_3_enable"].astype(int)
            + dataframe["v8_buy_condition_0_enable"].astype(int)
            + dataframe["v8_buy_condition_1_enable"].astype(int)
            + dataframe["v8_buy_condition_2_enable"].astype(int)
            + dataframe["v8_buy_condition_3_enable"].astype(int)
            + dataframe["v8_buy_condition_4_enable"].astype(int)
        )

        # append the minimum amount of conditions to be met
        conditions.append(
            dataframe["conditions_count"] >= self.buy_minimum_conditions.value
        )
        conditions.append(dataframe["volume"].gt(0))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1

        # verbose logging enable only for verbose information or troubleshooting
        if self.cust_log_verbose == True:
            for index, row in dataframe.iterrows():
                if row["buy"] == 1:
                    buy_cond_details = f"count={int(row['conditions_count'])}/bzv7_1={int(row['bzv7_buy_condition_1_enable'])}/bzv7_2={int(row['bzv7_buy_condition_2_enable'])}/bzv7_3={int(row['bzv7_buy_condition_3_enable'])}/bzv7_4={int(row['bzv7_buy_condition_4_enable'])}/bzv7_5={int(row['bzv7_buy_condition_5_enable'])}/bzv7_6={int(row['bzv7_buy_condition_6_enable'])}/bzv7_7={int(row['bzv7_buy_condition_7_enable'])}/bzv7_8={int(row['bzv7_buy_condition_8_enable'])}/bzv7_9={int(row['bzv7_buy_condition_9_enable'])}/bzv7_10={int(row['bzv7_buy_condition_10_enable'])}/bzv7_11={int(row['bzv7_buy_condition_11_enable'])}/bzv7_12={int(row['bzv7_buy_condition_12_enable'])}/bzv7_13={int(row['bzv7_buy_condition_13_enable'])}/bzv7_0={int(row['bzv7_buy_condition_0_enable'])}/v6_0={int(row['v6_buy_condition_0_enable'])}/v6_1={int(row['v6_buy_condition_1_enable'])}/v6_2={int(row['v6_buy_condition_2_enable'])}/v6_3={int(row['v6_buy_condition_3_enable'])}/v8_0={int(row['v8_buy_condition_0_enable'])}/v8_1={int(row['v8_buy_condition_1_enable'])}/v8_2={int(row['v8_buy_condition_2_enable'])}/v8_3={int(row['v8_buy_condition_3_enable'])}/v8_4={int(row['v8_buy_condition_4_enable'])}"

                    logger.info(
                        f"{metadata['pair']} - candle: {row['date']} - buy condition - details: {buy_cond_details}"
                    )

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        # dataframe.loc[:, "sell"] = 0

        if self.bzv7_sell_condition_0_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["bb_middleband"] * 1.01)
                    & (
                        dataframe["volume"] > 0
                    )  # Don't be gready, sell fast  # Make sure Volume is not 0
                )
            )

        if self.v8_sell_condition_0_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["bb_upperband"])
                    & (dataframe["close"].shift(1) > dataframe["bb_upperband"].shift(1))
                    & (dataframe["close"].shift(2) > dataframe["bb_upperband"].shift(2))
                    & (dataframe["close"].shift(2) > dataframe["bb_upperband"].shift(2))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_sell_condition_1_enable.value:
            conditions.append(
                (
                    (dataframe["rsi"] > self.v8_sell_rsi_main.value)
                    & (dataframe["volume"] > 0)
                )
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe


# SSL Channels
def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]


# Chaikin Money Flow Volume
def MFV(dataframe):
    df = dataframe.copy()
    N = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    M = N * df["volume"]
    return M
