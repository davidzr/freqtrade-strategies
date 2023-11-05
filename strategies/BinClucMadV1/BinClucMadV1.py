import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter
from functools import reduce


###############################################################################
#   this is the final adjustment version for all clucbinmad snip.
#  here i am goin to ad (v5&v6) & v8 &v9 lets see what we can achive
#   remember its production version no dev. should be lightweight
#
################################################################################

# SSL Channels
def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN))
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]


class BinClucMadV1(IStrategy):
    INTERFACE_VERSION = 2

    # minimal_roi = {
    #     "0": 0.028,  # I feel lucky!
    #     "10": 0.018,
    #     "40": 0.005,
    # }
    minimal_roi = {
        "0": 0.038,  # I feel lucky!
        "10": 0.028,
        "40": 0.015,
        "180": 0.018,  # We're going up?
    }

    stoploss = -0.99  # effectively disabled.

    timeframe = "5m"
    informative_timeframe = "1h"

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001  # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    buy_params = {
        #############
        # Enable/Disable conditions
        "v6_buy_condition_0_enable": True,
        "v6_buy_condition_1_enable": True,
        "v6_buy_condition_2_enable": True,
        "v6_buy_condition_3_enable": True,
        "v8_buy_condition_0_enable": True,
        "v8_buy_condition_1_enable": True,
        "v8_buy_condition_2_enable": True,
        "v8_buy_condition_3_enable": True,
        "v8_buy_condition_4_enable": True,
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
        "v8_sell_condition_1_enable": False,
    }
    ############################################################################

    # Buy CombinedBinHClucAndMADV6
    v6_buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_1_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_2_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_3_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    # Buy CombinedBinHClucV8
    v8_buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v8_buy_condition_1_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v8_buy_condition_2_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v8_buy_condition_3_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v8_buy_condition_4_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)

    v8_sell_condition_0_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)
    v8_sell_condition_1_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)

    v8_sell_rsi_main = DecimalParameter(72.0, 90.0, default=80, space="sell", decimals=2, optimize=False, load=True)
    buy_dip_threshold_0 = DecimalParameter(0.001, 0.1, default=0.015, space="buy", decimals=3, optimize=False, load=True)
    buy_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.12, space="buy", decimals=2, optimize=False, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.02, 0.4, default=0.28, space="buy", decimals=2, optimize=False, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.25, 0.44, default=0.36, space="buy", decimals=2, optimize=False, load=True)
    buy_bb40_bbdelta_close = DecimalParameter(0.005, 0.04, default=0.031, space="buy", optimize=False, load=True)
    buy_bb40_closedelta_close = DecimalParameter(0.01, 0.03, default=0.021, space="buy", optimize=False, load=True)
    buy_bb40_tail_bbdelta = DecimalParameter(0.2, 0.4, default=0.264, space="buy", optimize=False, load=True)
    buy_bb20_close_bblowerband = DecimalParameter(0.8, 1.1, default=0.992, space="buy", optimize=False, load=True)
    buy_bb20_volume = IntParameter(18, 36, default=29, space="buy", optimize=False, load=True)
    buy_rsi_diff = DecimalParameter(34.0, 60.0, default=50.48, space="buy", decimals=2, optimize=False, load=True)
    buy_min_inc = DecimalParameter(0.005, 0.05, default=0.01, space="buy", decimals=2, optimize=False, load=True)
    buy_rsi_1h = DecimalParameter(40.0, 70.0, default=67.0, space="buy", decimals=2, optimize=False, load=True)
    buy_rsi = DecimalParameter(30.0, 40.0, default=38.5, space="buy", decimals=2, optimize=False, load=True)
    buy_mfi = DecimalParameter(36.0, 65.0, default=36.0, space="buy", decimals=2, optimize=False, load=True)
    buy_volume_1 = DecimalParameter(1.0, 10.0, default=2.0, space="buy", decimals=2, optimize=False, load=True)
    buy_ema_open_mult_1 = DecimalParameter(0.01, 0.05, default=0.02, space="buy", decimals=3, optimize=False, load=True)

    sell_custom_roi_profit_1 = DecimalParameter(
        0.01, 0.03, default=0.01, space="sell", decimals=2, optimize=False, load=True
    )
    sell_custom_roi_rsi_1 = DecimalParameter(40.0, 56.0, default=50, space="sell", decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_2 = DecimalParameter(
        0.01, 0.20, default=0.04, space="sell", decimals=2, optimize=False, load=True
    )
    sell_custom_roi_rsi_2 = DecimalParameter(42.0, 56.0, default=50, space="sell", decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_3 = DecimalParameter(
        0.15, 0.30, default=0.08, space="sell", decimals=2, optimize=False, load=True
    )
    sell_custom_roi_rsi_3 = DecimalParameter(44.0, 58.0, default=56, space="sell", decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_4 = DecimalParameter(0.3, 0.7, default=0.14, space="sell", decimals=2, optimize=False, load=True)
    sell_custom_roi_rsi_4 = DecimalParameter(44.0, 60.0, default=58, space="sell", decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_5 = DecimalParameter(0.01, 0.1, default=0.04, space="sell", decimals=2, optimize=False, load=True)
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space="sell", decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space="sell", decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space="sell", decimals=3, optimize=False, load=True)
    sell_trail_profit_min_2 = DecimalParameter(0.01, 0.1, default=0.02, space="sell", decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.1, space="sell", decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space="sell", decimals=3, optimize=False, load=True)
    sell_custom_stoploss_1 = DecimalParameter(
        -0.15, -0.03, default=-0.05, space="sell", decimals=2, optimize=False, load=True
    )

    # Buy  CombinedBinHClucAndMADV9
    v9_buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_1_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_2_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_3_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_4_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_5_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_6_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_7_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_8_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_9_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v9_buy_condition_10_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)

    # Sell
    v9_sell_condition_0_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.7, 1.1, default=0.99, space="buy", optimize=False, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.7, 1.1, default=0.982, space="buy", optimize=False, load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=False, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=4, space="buy", decimals=1, optimize=False, load=True)

    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space="buy", decimals=1, optimize=False, load=True)

    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space="buy", decimals=1, optimize=False, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space="buy", decimals=2, optimize=False, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space="buy", decimals=2, optimize=False, load=True)

    def custom_stoplossv8(
        self, pair: str, trade: "Trade", current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        # Manage losing trades and open room for better ones.

        if current_profit > 0:
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=50)
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

    def custom_stoploss(
        self, pair: str, trade: "Trade", current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (current_time - timedelta(minutes=280) > trade.open_date_utc):
            return 0.01
        elif current_profit < self.sell_custom_stoploss_1.value:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            candle = dataframe.iloc[-1].squeeze()
            if candle is not None:
                # if (candle["sma_200_dec"]) & (candle["sma_200_dec_1h"]):
                #     return 0.01
                # We are at bottom. Wait...
                if candle["rsi_1h"] < 30:
                    return 0.99
                # Are we still sinking?
                if candle["close"] > candle["ema_200"]:
                    if current_rate * 1.025 < candle["open"]:
                        return 0.01
                if current_rate * 1.015 < candle["open"]:
                    return 0.01

        return 0.99

    def custom_sell(
        self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float, current_profit: float, **kwargs
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
                (current_profit > 0) & (current_profit < self.sell_custom_roi_profit_5.value) & (last_candle["sma_200_dec"])
            ):
                return "roi_target_5"

            elif (
                (current_profit > self.sell_trail_profit_min_1.value)
                & (current_profit < self.sell_trail_profit_max_1.value)
                & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_1.value))
            ):
                return "trail_target_1"
            elif (
                (current_profit > self.sell_trail_profit_min_2.value)
                & (current_profit < self.sell_trail_profit_max_2.value)
                & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_2.value))
            ):
                return "trail_target_2"

        return None

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        # EMA
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        informative_1h["sma_200"] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h["sma_200_dec"] = informative_1h["sma_200"] < informative_1h["sma_200"].shift(20)
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
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift()).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
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

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # START  V6
        if self.v6_buy_condition_0_enable.value:
            conditions.append(
                (  # strategy ClucMay72018
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["close"] < dataframe["ema_50"])
                    & (dataframe["close"] < 0.99 * dataframe["bb_lowerband"])
                    & (  # Guard is on, candle should dig not so hard (0,99)
                        dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4
                    )
                    &  # Try to exclude pumping
                    # (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                        # Don't buy if someone drop the market.
                    (dataframe["volume"] > 0)
                )
            )
        if self.v6_buy_condition_1_enable.value:
            conditions.append(
                (  # strategy ClucMay72018
                    (dataframe["close"] < dataframe["ema_50"])
                    & (dataframe["close"] < 0.975 * dataframe["bb_lowerband"])
                    & (  # Guard is off, candle should dig hard (0,975)
                        dataframe["volume"] < (dataframe["volume"].shift() * 4)
                    )
                    & (dataframe["rsi_1h"] < 15)  # Don't buy if someone drop the market.
                    & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)  # Buy only at dip
                    & (dataframe["volume"] > 0)  # Try to exclude pumping  # Make sure Volume is not 0
                )
            )
        if self.v6_buy_condition_2_enable.value:
            conditions.append(
                (  # strategy MACD Low buy
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.02))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))  # Don't buy if someone drop the market.
                    & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
                    & (dataframe["volume"] > 0)  # Try to exclude pumping  # Make sure Volume is not 0
                )
            )
        if self.v6_buy_condition_3_enable.value:
            conditions.append(
                (  # strategy MACD Low buy
                    (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.03))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))  # Don't buy if someone drop the market.
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
                )
            )
        # END  V6
        if self.v8_buy_condition_0_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["ema_50"] > dataframe["ema_200"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                    & (
                        ((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_1.value
                    )
                    & (
                        ((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_2.value
                    )
                    & dataframe["lower"].shift().gt(0)
                    & dataframe["bbdelta"].gt(dataframe["close"] * self.buy_bb40_bbdelta_close.value)
                    & dataframe["closedelta"].gt(dataframe["close"] * self.buy_bb40_closedelta_close.value)
                    & dataframe["tail"].lt(dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta.value)
                    & dataframe["close"].lt(dataframe["lower"].shift())
                    & dataframe["close"].le(dataframe["close"].shift())
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_buy_condition_1_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                    & (
                        ((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_1.value
                    )
                    & (
                        ((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_2.value
                    )
                    & (dataframe["close"] < dataframe["ema_50"])
                    & (dataframe["close"] < self.buy_bb20_close_bblowerband.value * dataframe["bb_lowerband"])
                    & (dataframe["volume"] < (dataframe["volume_mean_slow"].shift(1) * self.buy_bb20_volume.value))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_buy_condition_2_enable.value:
            conditions.append(
                (
                    (dataframe["close"] < dataframe["sma_5"])
                    & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
                    & (dataframe["ema_50"] > dataframe["ema_200"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                    & (
                        ((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_1.value
                    )
                    & (
                        ((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_2.value
                    )
                    & (
                        ((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_3.value
                    )
                    & (dataframe["rsi"] < dataframe["rsi_1h"] - self.buy_rsi_diff.value)
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_buy_condition_3_enable.value:
            conditions.append(
                (
                    (dataframe["sma_200"] > dataframe["sma_200"].shift(20))
                    & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(16))
                    & (
                        ((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_1.value
                    )
                    & (
                        ((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_2.value
                    )
                    & (
                        ((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_3.value
                    )
                    & (
                        ((dataframe["open"].rolling(24).min() - dataframe["close"]) / dataframe["close"])
                        > self.buy_min_inc.value
                    )
                    & (dataframe["rsi_1h"] > self.buy_rsi_1h.value)
                    & (dataframe["rsi"] < self.buy_rsi.value)
                    & (dataframe["mfi"] < self.buy_mfi.value)
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v8_buy_condition_4_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_100_1h"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                    & (
                        ((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_1.value
                    )
                    & (
                        ((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_2.value
                    )
                    & (
                        ((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"])
                        < self.buy_dip_threshold_3.value
                    )
                    & (dataframe["volume"].rolling(4).mean() * self.buy_volume_1.value > dataframe["volume"])
                    & (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_ema_open_mult_1.value))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))
                    & (dataframe["volume"] > 0)
                )
            )

        # START  VERSION9
        if self.v9_buy_condition_1_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["close"] < dataframe["bb_lowerband"] * self.buy_bb20_close_bblowerband_safe_1.value)
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["open"] - dataframe["close"]
                        < dataframe["bb_upperband"].shift(2) - dataframe["bb_lowerband"].shift(2)
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_2_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] < dataframe["bb_lowerband"] * self.buy_bb20_close_bblowerband_safe_2.value)
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["open"] - dataframe["close"]
                        < dataframe["bb_upperband"].shift(2) - dataframe["bb_lowerband"].shift(2)
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_3_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["close"] < dataframe["bb_lowerband"])
                    & (dataframe["rsi"] < self.buy_rsi_3.value)
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_4_enable.value:
            conditions.append(
                (
                    (dataframe["rsi_1h"] < self.buy_rsi_1h_1.value)
                    & (dataframe["close"] < dataframe["bb_lowerband"])
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_5_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_1.value))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
                )
            )
        if self.v9_buy_condition_6_enable.value:
            conditions.append(
                (
                    (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_2.value))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_7_enable.value:
            conditions.append(
                (
                    (dataframe["rsi_1h"] < self.buy_rsi_1h_2.value)
                    & (dataframe["ema_26"] > dataframe["ema_12"])
                    & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_1.value))
                    & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_8_enable.value:
            conditions.append(
                (
                    (dataframe["rsi_1h"] < self.buy_rsi_1h_3.value)
                    & (dataframe["rsi"] < self.buy_rsi_1.value)
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_9_enable.value:
            conditions.append(
                (
                    (dataframe["rsi_1h"] < self.buy_rsi_1h_4.value)
                    & (dataframe["rsi"] < self.buy_rsi_2.value)
                    & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
                    & (
                        dataframe["volume_mean_slow"]
                        > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v9_buy_condition_10_enable.value:
            conditions.append(
                (
                    (dataframe["close"] < dataframe["sma_5"])
                    & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
                    & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                    & (dataframe["rsi"] < dataframe["rsi_1h"] - 43.276)
                    & (dataframe["volume"] > 0)
                )
            )
        # END  V9

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        if self.v9_sell_condition_0_enable.value:
            conditions.append(
                (
                    (dataframe["close"] > dataframe["bb_middleband"] * 1.01)
                    & (dataframe["volume"] > 0)  # Don't be gready, sell fast  # Make sure Volume is not 0
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
                (qtpylib.crossed_above(dataframe["rsi"], self.v8_sell_rsi_main.value) & (dataframe["volume"] > 0))
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe
