from datetime import timedelta
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)


class Saturn5(IStrategy):
    # Strategy created by Shane Jones https://twitter.com/shanejones
    #
    # Assited by a number of contributors https://github.com/shanejones/goddard/graphs/contributors
    #
    # Original repo hosted at https://github.com/shanejones/goddard
    timeframe = "15m"

    # Stoploss
    stoploss = -0.20
    startup_candle_count: int = 480
    trailing_stop = False
    use_custom_stoploss = False
    use_sell_signal = False

    # signal controls
    buy_signal_1 = True
    buy_signal_2 = True
    buy_signal_3 = True

    # ROI table:
    minimal_roi = {
        "0": 0.05,
    }

    # Indicator values:

    # Signal 1
    s1_ema_xs = 3
    s1_ema_sm = 5
    s1_ema_md = 10
    s1_ema_xl = 50
    s1_ema_xxl = 240

    # Signal 2
    s2_ema_input = 50
    s2_ema_offset_input = -1

    s2_bb_sma_length = 49
    s2_bb_std_dev_length = 64
    s2_bb_lower_offset = 3

    s2_fib_sma_len = 50
    s2_fib_atr_len = 14

    s2_fib_lower_value = 4.236

    s3_ema_long = 50
    s3_ema_short = 20
    s3_ma_fast = 10
    s3_ma_slow = 20

    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=0),
            },
            {
                # Stop trading if max-drawdown is reached.
                "method": "MaxDrawdown",
                "lookback_period": to_minutes(hours=12),
                "trade_limit": 20,  # Considering all pairs that have a minimum of 20 trades
                "stop_duration": to_minutes(hours=1),
                "max_allowed_drawdown": 0.2,  # If max-drawdown is > 20% this will activate
            },
            {
                # Stop trading if a certain amount of stoploss occurred within a certain time window.
                "method": "StoplossGuard",
                "lookback_period": to_minutes(hours=3),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(hours=6),
                "only_per_pair": False,  # Looks at all pairs
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=1, minutes=30),
                "trade_limit": 2,  # Considering all pairs that have a minimum of 2 trades
                "stop_duration": to_minutes(hours=15),
                "required_profit": 0.02,  # If profit < 2% this will activate for a pair
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "required_profit": 0.01,  # If profit < 1% this will activate for a pair
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Adding EMA's into the dataframe
        dataframe["s1_ema_xs"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xs)
        dataframe["s1_ema_sm"] = ta.EMA(dataframe, timeperiod=self.s1_ema_sm)
        dataframe["s1_ema_md"] = ta.EMA(dataframe, timeperiod=self.s1_ema_md)
        dataframe["s1_ema_xl"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xl)
        dataframe["s1_ema_xxl"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xxl)

        s2_ema_value = ta.EMA(dataframe, timeperiod=self.s2_ema_input)
        s2_ema_xxl_value = ta.EMA(dataframe, timeperiod=200)
        dataframe["s2_ema"] = s2_ema_value - s2_ema_value * self.s2_ema_offset_input
        dataframe["s2_ema_xxl_off"] = s2_ema_xxl_value - s2_ema_xxl_value * self.s2_fib_lower_value
        dataframe["s2_ema_xxl"] = ta.EMA(dataframe, timeperiod=200)

        s2_bb_sma_value = ta.SMA(dataframe, timeperiod=self.s2_bb_sma_length)
        s2_bb_std_dev_value = ta.STDDEV(dataframe, self.s2_bb_std_dev_length)
        dataframe["s2_bb_std_dev_value"] = s2_bb_std_dev_value
        dataframe["s2_bb_lower_band"] = s2_bb_sma_value - (s2_bb_std_dev_value * self.s2_bb_lower_offset)

        s2_fib_atr_value = ta.ATR(dataframe, timeframe=self.s2_fib_atr_len)
        s2_fib_sma_value = ta.SMA(dataframe, timeperiod=self.s2_fib_sma_len)

        dataframe["s2_fib_lower_band"] = s2_fib_sma_value - s2_fib_atr_value * self.s2_fib_lower_value

        s3_bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe["s3_bb_lowerband"] = s3_bollinger["lower"]

        dataframe["s3_ema_long"] = ta.EMA(dataframe, timeperiod=self.s3_ema_long)
        dataframe["s3_ema_short"] = ta.EMA(dataframe, timeperiod=self.s3_ema_short)
        dataframe["s3_fast_ma"] = ta.EMA(dataframe["volume"] * dataframe["close"], self.s3_ma_fast) / ta.EMA(
            dataframe["volume"], self.s3_ma_fast
        )
        dataframe["s3_slow_ma"] = ta.EMA(dataframe["volume"] * dataframe["close"], self.s3_ma_slow) / ta.EMA(
            dataframe["volume"], self.s3_ma_slow
        )

        # Volume weighted MACD
        dataframe["fastMA"] = ta.EMA(dataframe["volume"] * dataframe["close"], 12) / ta.EMA(dataframe["volume"], 12)
        dataframe["slowMA"] = ta.EMA(dataframe["volume"] * dataframe["close"], 26) / ta.EMA(dataframe["volume"], 26)
        dataframe["vwmacd"] = dataframe["fastMA"] - dataframe["slowMA"]
        dataframe["signal"] = ta.EMA(dataframe["vwmacd"], 9)
        dataframe["hist"] = dataframe["vwmacd"] - dataframe["signal"]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # basic buy methods to keep the strategy simple

        if self.buy_signal_1:
            conditions = [
                dataframe["vwmacd"] < dataframe["signal"],
                dataframe["low"] < dataframe["s1_ema_xxl"],
                dataframe["close"] > dataframe["s1_ema_xxl"],
                qtpylib.crossed_above(dataframe["s1_ema_sm"], dataframe["s1_ema_md"]),
                dataframe["s1_ema_xs"] < dataframe["s1_ema_xl"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")

        if self.buy_signal_2:
            conditions = [
                qtpylib.crossed_above(dataframe["s2_fib_lower_band"], dataframe["s2_bb_lower_band"]),
                dataframe["close"] < dataframe["s2_ema"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_2")

        if self.buy_signal_3:
            conditions = [
                dataframe["low"] < dataframe["s3_bb_lowerband"],
                dataframe["high"] > dataframe["s3_slow_ma"],
                dataframe["high"] < dataframe["s3_ema_long"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_3")

        if not all([self.buy_signal_1, self.buy_signal_2, self.buy_signal_3]):
            dataframe.loc[(), "buy"] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # This is essentailly ignored as we're using strict ROI / Stoploss / TTP sale scenarios
        dataframe.loc[(), "sell"] = 0
        return dataframe
