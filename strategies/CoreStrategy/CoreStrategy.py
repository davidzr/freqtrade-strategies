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
    stoploss_from_open,
)
from functools import reduce
import logging
# -------------------------------------------------------------------------------------------------
# --- logger for parameter merging output, only remove if you remove it further down too! ---------
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------------------------------------


class CoreStrategy(IStrategy):
    INTERFACE_VERSION = 2

    # minimal_roi = {"0": 0.038, "20": 0.028, "40": 0.02, "60": 0.015, "180": 0.018, }
    # minimal_roi = {"0": 0.038, "20": 0.028, "40": 0.02, "60": 0.015, "180": 0.018, }
    minimal_roi = {"0": 0.20, "38": 0.074, "78": 0.025, "194": 0}
    stoploss = -0.228  # effectively disabled.

    timeframe = "5m"
    informative_timeframe = "1h"

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.049

    # Custom stoploss
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        "buy_minimum_conditions": 1,
        #############
        # Enable/Disable conditions
        "smaoffset_buy_condition_0_enable": True,
        "smaoffset_buy_condition_1_enable": True,
        "v6_buy_condition_0_enable": False, # avg 0.47 dd 27%
        "v6_buy_condition_1_enable": True, # no trade
        "v6_buy_condition_2_enable": True,  # avg 2.32
        "v6_buy_condition_3_enable": True, # avg 1.12 dd 6%
        "v8_buy_condition_0_enable": True, # avg 0.74
        "v8_buy_condition_1_enable": False,  # avg 0.41 dd 37%
        "v8_buy_condition_2_enable": True,   # avg 1.37
        "v8_buy_condition_3_enable": False,  # avg 0.41
        "v8_buy_condition_4_enable": True,   # avg 1.29
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": False,
        "v9_buy_condition_2_enable": False,
        "v9_buy_condition_3_enable": False,
        "v9_buy_condition_4_enable": False,
        "v9_buy_condition_5_enable": False,
        "v9_buy_condition_6_enable": False,
        "v9_buy_condition_7_enable": False,
        "v9_buy_condition_8_enable": False,
        "v9_buy_condition_9_enable": False,
        "v9_buy_condition_10_enable": False,
    }
    sell_params = {
        #############
        # Enable/Disable conditions
        "v9_sell_condition_0_enable": False,
        "v8_sell_condition_0_enable": True,
        "v8_sell_condition_1_enable": True,
        "smaoffset_sell_condition_0_enable": False,
    }
    plot_config = {
        'main_plot': {
             },
        'subplots': {
            "buy tag": {
                'buy_tag': {'color': 'green'}
            },
        }
    }

    # if you want to see which buy conditions were met
    # or if there is an trade exit override due to high RSI set to True
    # logger will output the buy and trade exit conditions
    cust_log_verbose = False

    ############################################################################
    # Buy SMAOffsetProtectOpt

    smaoffset_buy_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    smaoffset_buy_condition_1_enable = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=False, load=True
    )
    smaoffset_sell_condition_0_enable = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=False, load=True
    )
    # hyperopt parameters for SMAOffsetProtectOpt
    base_nb_candles_buy = IntParameter(
        5, 80, default=20, space="buy", optimize=False, load=True
    )
    base_nb_candles_sell = IntParameter(
        5, 80, default=24, space="sell", optimize=False, load=True
    )
    low_offset = DecimalParameter(
        0.9, 0.99, default=0.975, space="buy", optimize=True, load=True
    )
    high_offset = DecimalParameter(
        0.99, 1.1, default=1.012, space="sell", optimize=True, load=True
    )
    # Protection
    fast_ewo = IntParameter(10, 50, default=50, space="buy", optimize=False, load=True)
    slow_ewo = IntParameter(
        100, 200, default=200, space="buy", optimize=False, load=True
    )
    ewo_low = DecimalParameter(
        -20.0, -8.0, default=-19.881, space="buy", optimize=True, load=True
    )
    ewo_high = DecimalParameter(
        2.0, 12.0, default=5.499, space="buy", optimize=True, load=True
    )
    rsi_buy = IntParameter(30, 70, default=50, space="buy", optimize=True, load=True)
    # Buy CombinedBinHClucAndMADV6

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
    # Buy CombinedBinHClucV8
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
    # Buy  CombinedBinHClucAndMADV9
    v9_buy_condition_0_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_1_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_2_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_3_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_4_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_5_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_6_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_7_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_8_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_9_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )
    v9_buy_condition_10_enable = CategoricalParameter(
        [True, False], default=False, space="buy", optimize=False, load=True
    )

    # Sell
    v9_sell_condition_0_enable = CategoricalParameter(
        [True, False], default=False, space="sell", optimize=False, load=True
    )

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(
        0.7, 1.1, default=0.99, space="buy", optimize=False, load=True
    )
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(
        0.7, 1.1, default=0.982, space="buy", optimize=False, load=True
    )
    buy_volume_pump_1 = DecimalParameter(
        0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=False, load=True
    )
    buy_volume_drop_1 = DecimalParameter(
        1, 10, default=4, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_1h_1 = DecimalParameter(
        10.0, 40.0, default=16.5, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_1h_2 = DecimalParameter(
        10.0, 40.0, default=15.0, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_1h_3 = DecimalParameter(
        10.0, 40.0, default=20.0, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_1h_4 = DecimalParameter(
        10.0, 40.0, default=35.0, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_1 = DecimalParameter(
        10.0, 40.0, default=28.0, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_2 = DecimalParameter(
        7.0, 40.0, default=10.0, space="buy", decimals=1, optimize=False, load=True
    )
    buy_rsi_3 = DecimalParameter(
        7.0, 40.0, default=14.2, space="buy", decimals=1, optimize=False, load=True
    )
    buy_macd_1 = DecimalParameter(
        0.01, 0.09, default=0.02, space="buy", decimals=2, optimize=False, load=True
    )
    buy_macd_2 = DecimalParameter(
        0.01, 0.09, default=0.03, space="buy", decimals=2, optimize=False, load=True
    )
    # minimum conditions to match in buy
    buy_minimum_conditions = IntParameter(
        1, 2, default=1, space="buy", optimize=False, load=True
    )

    def custom_stoploss(
        self, pair: str, trade: "Trade", current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        # Manage losing trades and open room for better ones.

        if current_profit > 0:
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=240)
            # trade_time_240 = trade.open_date_utc + timedelta(minutes=240)
            # Trade open more then 60 minutes. For this strategy it's means -> loss
            # Let's try to minimize the loss

            if current_time > trade_time_50:

                try:
                    number_of_candle_shift = int((current_time - trade_time_50).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()
                    if (candle["sma_200_dec"]) & (candle["sma_200_dec_1h"]):
                        return 0.01
                    # We are at bottom. Wait...
                    if candle["rsi_1h"] < 30:
                        return 0.99

                    # Are we still sinking?
                    if candle["close"] > candle["ema_200"]:
                        if current_rate * 1.025 < candle["open"]:
                            return 0.01

                    if current_rate * 1.015 < candle["open"]:
                        return 0.01

                except IndexError as error:

                    # Whoops, set stoploss at 10%
                    return 0.1

        return 0.99



    def custom_sell(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if last_candle is not None:
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
        # Sell any positions at a loss if they are held for more than one day.
        # if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
        #     return 'unclog'

        return None

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        # EMA
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        informative_1h["sma_200"] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h["sma_200_dec"] = informative_1h["sma_200"] < informative_1h[
            "sma_200"
        ].shift(20)
        # RSI
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

        # SSL Channels
        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h
        informative_1h["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()
        dataframe["closedelta"] = (
            dataframe["close"] - dataframe["close"].shift()
        ).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()

        # EMA
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe["sma_5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        # MFI
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        # ------ ATR stuff
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # ------ SMAOffsetProtectOpt
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe["EWO"] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # reset additional dataframe rows
        dataframe.loc[:, "v9_buy_condition_1_enable"] = False
        dataframe.loc[:, "v9_buy_condition_2_enable"] = False
        dataframe.loc[:, "v9_buy_condition_3_enable"] = False
        dataframe.loc[:, "v9_buy_condition_4_enable"] = False
        dataframe.loc[:, "v9_buy_condition_5_enable"] = False
        dataframe.loc[:, "v9_buy_condition_6_enable"] = False
        dataframe.loc[:, "v9_buy_condition_7_enable"] = False
        dataframe.loc[:, "v9_buy_condition_8_enable"] = False
        dataframe.loc[:, "v9_buy_condition_9_enable"] = False
        dataframe.loc[:, "v9_buy_condition_10_enable"] = False
        dataframe.loc[:, "v6_buy_condition_0_enable"] = False
        dataframe.loc[:, "v6_buy_condition_1_enable"] = False
        dataframe.loc[:, "v6_buy_condition_2_enable"] = False
        dataframe.loc[:, "v6_buy_condition_3_enable"] = False
        dataframe.loc[:, "v8_buy_condition_0_enable"] = False
        dataframe.loc[:, "v8_buy_condition_1_enable"] = False
        dataframe.loc[:, "v8_buy_condition_2_enable"] = False
        dataframe.loc[:, "v8_buy_condition_3_enable"] = False
        dataframe.loc[:, "v8_buy_condition_4_enable"] = False
        dataframe.loc[:, "smaoffset_buy_condition_0_enable"] = False
        dataframe.loc[:, "smaoffset_buy_condition_1_enable"] = False
        dataframe.loc[:, "conditions_count"] = 0
        dataframe.loc[:, 'buy_tag'] = ''

        dataframe["ma_buy"] = (
            dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
            * self.low_offset.value
        )

        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ma_buy"])
                & (dataframe["EWO"] > self.ewo_high.value)
                & (dataframe["rsi"] < self.rsi_buy.value)
                & (self.smaoffset_buy_condition_0_enable.value == True)
            ),
            ['smaoffset_buy_condition_0_enable', 'buy_tag']] = (1, 'buy_signal_smaoffset_0')


        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ma_buy"])
                & (dataframe["EWO"] < self.ewo_low.value)
                & (self.smaoffset_buy_condition_1_enable.value == True)
            ),
            ['smaoffset_buy_condition_1_enable', 'buy_tag']] = (1, 'buy_signal_smaoffset_1')


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
            ['v8_buy_condition_0_enable', 'buy_tag']] = (1, 'buy_signal_v8_0')


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
            ['v8_buy_condition_1_enable', 'buy_tag']] = (1, 'buy_signal_v8_1')

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
            ['v8_buy_condition_2_enable', 'buy_tag']] = (1, 'buy_signal_v8_2')



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
            ['v8_buy_condition_3_enable', 'buy_tag']] = (1, 'buy_signal_v8_3')


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
            ['v8_buy_condition_4_enable', 'buy_tag']] = (1, 'buy_signal_v8_4')

        # start from here
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"]
                    * self.buy_bb20_close_bblowerband_safe_1.value
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["open"] - dataframe["close"]
                    < dataframe["bb_upperband"].shift(2)
                    - dataframe["bb_lowerband"].shift(2)
                )
                & (self.v9_buy_condition_1_enable.value == True)
            ),
            ['v9_buy_condition_1_enable', 'buy_tag']] = (1, 'buy_signal_v9_1')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"]
                    * self.buy_bb20_close_bblowerband_safe_2.value
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["open"] - dataframe["close"]
                    < dataframe["bb_upperband"].shift(2)
                    - dataframe["bb_lowerband"].shift(2)
                )
                & (self.v9_buy_condition_2_enable.value == True)
            ),
            ['v9_buy_condition_2_enable', 'buy_tag']] = (1, 'buy_signal_v9_2')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["close"] < dataframe["bb_lowerband"])
                & (dataframe["rsi"] < self.buy_rsi_3.value)
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (self.v9_buy_condition_3_enable.value == True)
            ),
            ['v9_buy_condition_3_enable', 'buy_tag']] = (1, 'buy_signal_v9_3')

        dataframe.loc[
            (
                (dataframe["rsi_1h"] < self.buy_rsi_1h_1.value)
                & (dataframe["close"] < dataframe["bb_lowerband"])
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (self.v9_buy_condition_4_enable.value == True)
            ),
            ['v9_buy_condition_4_enable', 'buy_tag']] = (1, 'buy_signal_v9_4')
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_macd_1.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (self.v9_buy_condition_5_enable.value == True)
            ),
            ['v9_buy_condition_5_enable', 'buy_tag']] = (1, 'buy_signal_v9_5')
        dataframe.loc[
            (
                (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_macd_2.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (self.v9_buy_condition_6_enable.value == True)
            ),
            ['v9_buy_condition_6_enable', 'buy_tag']] = (1, 'buy_signal_v9_6')
        dataframe.loc[
            (
                (dataframe["rsi_1h"] < self.buy_rsi_1h_2.value)
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_macd_1.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (self.v9_buy_condition_7_enable.value == True)
            ),
            ['v9_buy_condition_7_enable', 'buy_tag']] = (1, 'buy_signal_v9_7')
        dataframe.loc[
            (
                (dataframe["rsi_1h"] < self.buy_rsi_1h_3.value)
                & (dataframe["rsi"] < self.buy_rsi_1.value)
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (self.v9_buy_condition_8_enable.value == True)
            ),
            ['v9_buy_condition_8_enable', 'buy_tag']] = (1, 'buy_signal_v9_8')
        dataframe.loc[
            (
                (dataframe["rsi_1h"] < self.buy_rsi_1h_4.value)
                & (dataframe["rsi"] < self.buy_rsi_2.value)
                & (
                    dataframe["volume"]
                    < (dataframe["volume"].shift() * self.buy_volume_drop_1.value)
                )
                & (
                    dataframe["volume_mean_slow"]
                    > dataframe["volume_mean_slow"].shift(30)
                    * self.buy_volume_pump_1.value
                )
                & (self.v9_buy_condition_9_enable.value == True)
            ),
            ['v9_buy_condition_9_enable', 'buy_tag']] = (1, 'buy_signal_v9_9')
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["sma_5"])
                & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["rsi"] < dataframe["rsi_1h"] - 43.276)
                & (self.v9_buy_condition_10_enable.value == True)
            ),
            ['v9_buy_condition_10_enable', 'buy_tag']] = (1, 'buy_signal_v9_10')
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
            ['v6_buy_condition_0_enable', 'buy_tag']] = (1, 'buy_signal_v6_0')

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
            ['v6_buy_condition_1_enable', 'buy_tag']] = (1, 'buy_signal_v6_1')


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
            ['v6_buy_condition_2_enable', 'buy_tag']] = (1, 'buy_signal_v6_2')

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
            ['v6_buy_condition_3_enable', 'buy_tag']] = (1, 'buy_signal_v6_3')


        # count the amount of conditions met
        dataframe.loc[:, "conditions_count"] = (
            dataframe["v9_buy_condition_1_enable"].astype(int)
            + dataframe["v9_buy_condition_2_enable"].astype(int)
            + dataframe["v9_buy_condition_3_enable"].astype(int)
            + dataframe["v9_buy_condition_4_enable"].astype(int)
            + dataframe["v9_buy_condition_5_enable"].astype(int)
            + dataframe["v9_buy_condition_6_enable"].astype(int)
            + dataframe["v9_buy_condition_7_enable"].astype(int)
            + dataframe["v9_buy_condition_8_enable"].astype(int)
            + dataframe["v9_buy_condition_9_enable"].astype(int)
            + dataframe["v9_buy_condition_10_enable"].astype(int)
            + dataframe["v6_buy_condition_0_enable"].astype(int)
            + dataframe["v6_buy_condition_1_enable"].astype(int)
            + dataframe["v6_buy_condition_2_enable"].astype(int)
            + dataframe["v6_buy_condition_3_enable"].astype(int)
            + dataframe["v8_buy_condition_0_enable"].astype(int)
            + dataframe["v8_buy_condition_1_enable"].astype(int)
            + dataframe["v8_buy_condition_2_enable"].astype(int)
            + dataframe["v8_buy_condition_3_enable"].astype(int)
            + dataframe["v8_buy_condition_4_enable"].astype(int)
            + dataframe["smaoffset_buy_condition_0_enable"].astype(int)
            + dataframe["smaoffset_buy_condition_1_enable"].astype(int)
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
                    buy_cond_details = f"count={int(row['conditions_count'])}/v9_1={int(row['v9_buy_condition_1_enable'])}/v9_2={int(row['v9_buy_condition_2_enable'])}/v9_3={int(row['v9_buy_condition_3_enable'])}/v9_4={int(row['v9_buy_condition_4_enable'])}/v9_5={int(row['v9_buy_condition_5_enable'])}/v9_6={int(row['v9_buy_condition_6_enable'])}/v9_7={int(row['v9_buy_condition_7_enable'])}/v9_8={int(row['v9_buy_condition_8_enable'])}/v9_9={int(row['v9_buy_condition_9_enable'])}/v9_10={int(row['v9_buy_condition_10_enable'])}/v6_0={int(row['v6_buy_condition_0_enable'])}/v6_1={int(row['v6_buy_condition_1_enable'])}/v6_2={int(row['v6_buy_condition_2_enable'])}/v6_3={int(row['v6_buy_condition_3_enable'])}/v8_0={int(row['v8_buy_condition_0_enable'])}/v8_1={int(row['v8_buy_condition_1_enable'])}/v8_2={int(row['v8_buy_condition_2_enable'])}/v8_3={int(row['v8_buy_condition_3_enable'])}/v8_4={int(row['v8_buy_condition_4_enable'])}/sma_0={int(row['smaoffset_buy_condition_0_enable'])}/sma_1={int(row['smaoffset_buy_condition_1_enable'])}"

                    logger.info(
                        f"{metadata['pair']} - candle: {row['date']} - buy condition - details: {buy_cond_details}"
                    )

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe["ma_sell"] = (
            dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
            * self.high_offset.value
        )
        if self.smaoffset_sell_condition_0_enable.value:
            conditions.append(
                (
                    (qtpylib.crossed_below(dataframe["close"], dataframe["ma_sell"]))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_sell_condition_0_enable.value:
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
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_sell_condition_1_enable.value:
            conditions.append(
                (
                    (dataframe["rsi_1h"] > self.v8_sell_rsi_main.value)
                    & (dataframe["volume"] > 0)
                )
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe


# --- custom indicators ---------------------------------------------------------------------------


def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
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


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif


class BinClucMadv1(CoreStrategy):
    INTERFACE_VERSION = 2

    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        "buy_minimum_conditions": 1,
        "smaoffset_buy_condition_0_enable": False,
        "smaoffset_buy_condition_1_enable": False,
        "v6_buy_condition_0_enable": False, # avg 0.47 dd 27%
        "v6_buy_condition_1_enable": True, # no trade
        "v6_buy_condition_2_enable": True,  # avg 2.32
        "v6_buy_condition_3_enable": True, # avg 1.12 dd 6%
        "v8_buy_condition_0_enable": True, # avg 0.74
        "v8_buy_condition_1_enable": False,  # avg 0.41 dd 37%
        "v8_buy_condition_2_enable": True,   # avg 1.37
        "v8_buy_condition_3_enable": False,  # avg 0.41
        "v8_buy_condition_4_enable": True,   # avg 1.29
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": True,
        "v9_buy_condition_2_enable": True,
        "v9_buy_condition_3_enable": True,
        "v9_buy_condition_4_enable": False,
        "v9_buy_condition_5_enable": True,
        "v9_buy_condition_6_enable": True,
        "v9_buy_condition_7_enable": True,
        "v9_buy_condition_8_enable": False,
        "v9_buy_condition_9_enable": False,
        "v9_buy_condition_10_enable": False,

    }


class BinClucMadv2(CoreStrategy):
    INTERFACE_VERSION = 2

    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        "buy_minimum_conditions": 1,
        "smaoffset_buy_condition_0_enable": False,
        "smaoffset_buy_condition_1_enable": False,
        "v6_buy_condition_0_enable": False, # avg 0.47 dd 27%
        "v6_buy_condition_1_enable": True, # no trade
        "v6_buy_condition_2_enable": True,  # avg 2.32
        "v6_buy_condition_3_enable": True, # avg 1.12 dd 6%
        "v8_buy_condition_0_enable": True, # avg 0.74
        "v8_buy_condition_1_enable": False,  # avg 0.41 dd 37%
        "v8_buy_condition_2_enable": True,   # avg 1.37
        "v8_buy_condition_3_enable": False,  # avg 0.41
        "v8_buy_condition_4_enable": True,   # avg 1.29
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": False,
        "v9_buy_condition_2_enable": False,
        "v9_buy_condition_3_enable": False,
        "v9_buy_condition_4_enable": False,
        "v9_buy_condition_5_enable": False,
        "v9_buy_condition_6_enable": False,
        "v9_buy_condition_7_enable": False,
        "v9_buy_condition_8_enable": False,
        "v9_buy_condition_9_enable": False,
        "v9_buy_condition_10_enable": False,

    }



class BinClucMadSMAv1(CoreStrategy):

    INTERFACE_VERSION = 2


    stoploss = -0.228  # effectively disabled.
    # Custom stoploss
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        "buy_minimum_conditions": 1,
        "smaoffset_buy_condition_0_enable": True,
        "smaoffset_buy_condition_1_enable": True,
        "v6_buy_condition_0_enable": False, # avg 0.47 dd 27%
        "v6_buy_condition_1_enable": True, # no trade
        "v6_buy_condition_2_enable": True,  # avg 2.32
        "v6_buy_condition_3_enable": True, # avg 1.12 dd 6%
        "v8_buy_condition_0_enable": True, # avg 0.74
        "v8_buy_condition_1_enable": False,  # avg 0.41 dd 37%
        "v8_buy_condition_2_enable": True,   # avg 1.37
        "v8_buy_condition_3_enable": False,  # avg 0.41
        "v8_buy_condition_4_enable": True,   # avg 1.29
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": False,
        "v9_buy_condition_2_enable": False,
        "v9_buy_condition_3_enable": False,
        "v9_buy_condition_4_enable": False,
        "v9_buy_condition_5_enable": False,
        "v9_buy_condition_6_enable": False,
        "v9_buy_condition_7_enable": False,
        "v9_buy_condition_8_enable": False,
        "v9_buy_condition_9_enable": False,
        "v9_buy_condition_10_enable": False,


    }




class BinClucMadSMAv2(CoreStrategy):

    INTERFACE_VERSION = 2

    stoploss = -0.228  # effectively disabled.
    # Custom stoploss
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        "buy_minimum_conditions": 1,

        "smaoffset_buy_condition_0_enable": True,
        "smaoffset_buy_condition_1_enable": True,
        "v6_buy_condition_0_enable": False, # avg 0.47 dd 27%
        "v6_buy_condition_1_enable": True, # no trade
        "v6_buy_condition_2_enable": True,  # avg 2.32
        "v6_buy_condition_3_enable": True, # avg 1.12 dd 6%
        "v8_buy_condition_0_enable": True, # avg 0.74
        "v8_buy_condition_1_enable": False,  # avg 0.41 dd 37%
        "v8_buy_condition_2_enable": True,   # avg 1.37
        "v8_buy_condition_3_enable": False,  # avg 0.41
        "v8_buy_condition_4_enable": True,   # avg 1.29
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": True,
        "v9_buy_condition_2_enable": True,
        "v9_buy_condition_3_enable": True,
        "v9_buy_condition_4_enable": False,
        "v9_buy_condition_5_enable": True,
        "v9_buy_condition_6_enable": True,
        "v9_buy_condition_7_enable": True,
        "v9_buy_condition_8_enable": False,
        "v9_buy_condition_9_enable": False,
        "v9_buy_condition_10_enable": False,

    }



