import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter
from functools import reduce

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


class BinClucMad(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.038,         # I feel lucky!
        "10": 0.028,
        "40": 0.015,
        "180": 0.018,        # We're going up?
    }

    stoploss = -0.99  # effectively disabled.

    timeframe = "5m"
    informative_timeframe = "1h"

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001  # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

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
        #############
        # Enable/Disable conditions
        "v9_buy_condition_0_enable": False,
        "v9_buy_condition_1_enable": True,
        "v9_buy_condition_2_enable": True,
        "v9_buy_condition_3_enable": True,
        "v9_buy_condition_4_enable": True,
        "v9_buy_condition_5_enable": True,
        "v9_buy_condition_6_enable": True,
        "v9_buy_condition_7_enable": True,
        "v9_buy_condition_8_enable": True,
        "v9_buy_condition_9_enable": True,
        "v9_buy_condition_10_enable": True,
        "v6_buy_condition_0_enable": True,
        "v6_buy_condition_1_enable": True,
        "v6_buy_condition_2_enable": True,
        "v6_buy_condition_3_enable": True,
        "v6_buy_condition_4_enable": True,
    }
    sell_params = {
        #############
        # Enable/Disable conditions
        "v9_sell_condition_0_enable": False,
        "v8_sell_condition_1_enable": True,
        "v8_sell_condition_2_enable": False,
    }
    ############################################################################

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
    v6_buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_1_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_2_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_3_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    v6_buy_condition_4_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    # Sell
    v9_sell_condition_0_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)
    v8_sell_condition_0_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)
    v8_sell_condition_1_enable = CategoricalParameter([True, False], default=True, space="sell", optimize=False, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.7, 1.1, default=0.99, space="buy", optimize=True, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.7, 1.1, default=0.982, space="buy", optimize=True, load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=True, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=4, space="buy", decimals=1, optimize=True, load=True)

    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space="buy", decimals=1, optimize=True, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space="buy", decimals=1, optimize=True, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space="buy", decimals=1, optimize=True, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space="buy", decimals=1, optimize=True, load=True)

    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space="buy", decimals=1, optimize=True, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space="buy", decimals=1, optimize=True, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space="buy", decimals=1, optimize=True, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space="buy", decimals=2, optimize=True, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space="buy", decimals=2, optimize=True, load=True)

    v8_sell_rsi_main = DecimalParameter(72.0, 90.0, default=80, space="sell", decimals=2, optimize=True, load=True)

    def custom_stoploss(
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
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

        # SSL Channels
        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()

        # EMA
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)

        # SMA
        dataframe["sma_5"] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

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
        # START  V6
        if self.v6_buy_condition_0_enable.value:
            conditions.append(
                (  # strategy ClucMay72018
                    (dataframe["close"] > dataframe["ema_200"])
                    & (dataframe["close"] > dataframe["ema_200_1h"])
                    & (dataframe["close"] < dataframe["ema_50"])
                    & (dataframe["close"] < 0.99 * dataframe["bb_lowerband"])
                    & (
                        (dataframe["volume"] < (dataframe["volume_mean_slow"].shift(1) * 21))
                        | (dataframe["volume_mean_slow"] > (dataframe["volume_mean_slow"].shift(30) * 0.4))
                    )
                    & (dataframe["volume"] > 0)
                )
            )
        if self.v6_buy_condition_1_enable.value:
            conditions.append(
                (  # strategy ClucMay72018
                    (dataframe["close"] < dataframe["ema_50"])
                    & (dataframe["close"] < 0.975 * dataframe["bb_lowerband"])
                    & (
                        (dataframe["volume"] < (dataframe["volume_mean_slow"].shift(1) * 20))
                        | (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
                    )
                    & (dataframe["rsi_1h"] < 15)  # Don't buy if someone drop the market.
                    & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
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
                    & (
                        (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                        | (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
                    )
                    & (dataframe["close"] < (dataframe["bb_lowerband"]))
                    &
                    #
                    (dataframe["volume"] > 0)
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
            conditions.append(((dataframe["rsi"] > self.v8_sell_rsi_main.value) & (dataframe["volume"] > 0)))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe
