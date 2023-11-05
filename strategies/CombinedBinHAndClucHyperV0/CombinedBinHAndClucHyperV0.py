# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from abc import ABC, abstractmethod
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds
from datetime import datetime, timedelta
import math

class CombinedBinHAndClucHyperV0(IStrategy):
    timeframe = '1m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # ----------------------------------------------------------------
    # Hyper Params
    # 
    # # Buy 
    buy_a_bbdelta_rate = DecimalParameter(0.004, 0.016, default=0.016, decimals=3)
    buy_a_closedelta_rate = DecimalParameter(0.000, 0.010, default=0.0087, decimals=4)
    buy_a_tail_rate = DecimalParameter(0.12, 0.5, default=0.28, decimals=2)
    buy_a_time_window = IntParameter(40, 100, default=30)
    buy_a_min_sell_rate = DecimalParameter(1.004, 1.1, default=0.004, decimals=3)

    buy_b_close_rate = DecimalParameter(0.4, 1.8, default=0.979, decimals=3)
    buy_b_volume_mean_slow_window = IntParameter(100, 300, default=30)
    buy_b_ema_slow = IntParameter(40, 100, default=50)
    buy_b_time_window = IntParameter(100, 300, default=20)
    buy_b_volume_mean_slow_num = IntParameter(10, 100, default=20)
    # Sell
    sell_bb_middleband_window = IntParameter(50, 200, default=20)
    sell_trailing_stop_positive_offset = DecimalParameter(0.01, 0.03, default=0.008, decimals=3)
    sell_trailing_stop_positive = 0.001

    # ----------------------------------------------------------------
    # Buy hyperspace params:
    buy_params = {
        'buy_a_bbdelta_rate': 0.0160,
        'buy_a_closedelta_rate': 0.0088,
        'buy_a_tail_rate': 0.9,
        'buy_a_time_window': 21,
        'buy_a_min_sell_rate': 1.03,

        'buy_b_close_rate': 0.979,
        'buy_b_time_window': 20,
        'buy_b_ema_slow': 50,
        'buy_b_volume_mean_slow_num': 20,
        'buy_b_volume_mean_slow_window': 30,
    }

    # Sell hyperspace params:
    sell_params = {
        'sell_bb_middleband_window': 91,
        "sell_trailing_stop_positive_offset": 0.008,
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.1
    trailing_stop = False
    trailing_only_offset_is_reached = False
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sell_trailing_stop_positive_offset = self.sell_trailing_stop_positive_offset.value if isinstance(self.sell_trailing_stop_positive_offset, ABC) else self.sell_trailing_stop_positive_offset
        sell_trailing_stop_positive = self.sell_trailing_stop_positive.value if isinstance(self.sell_trailing_stop_positive, ABC) else self.sell_trailing_stop_positive

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle is None:
            return -1

        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc - timedelta(seconds=timeframe_to_seconds(self.timeframe)))
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        if trade_candle.empty:
            return -1

        trade_candle = trade_candle.squeeze()

        slippage_ratio = trade.open_rate / trade_candle['close'] - 1
        slippage_ratio = slippage_ratio if slippage_ratio > 0 else 0
        current_profit_comp = current_profit + slippage_ratio

        if current_profit_comp < sell_trailing_stop_positive_offset:
            return -1
        else:
            return sell_trailing_stop_positive

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        for x in self.buy_a_time_window.range if isinstance(self.buy_a_time_window, ABC) else [self.buy_a_time_window]:
            buy_bollinger = qtpylib.bollinger_bands(dataframe['close'], window=x, stds=2)
            dataframe[f'lower_{x}'] = buy_bollinger['lower']
            dataframe[f'bbdelta_{x}'] = (buy_bollinger['mid'] - dataframe[f'lower_{x}']).abs()
            dataframe[f'closedelta_{x}'] = (dataframe['close'] - dataframe['close'].shift()).abs()
            dataframe[f'tail_{x}'] = (dataframe['close'] - dataframe['low']).abs()

        # strategy ClucMay72018
        for x in self.buy_b_time_window.range if isinstance(self.buy_b_time_window, ABC) else [self.buy_b_time_window]:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=x, stds=2)
            dataframe[f'bb_lowerband_{x}'] = bollinger['lower']

        for x in self.buy_b_ema_slow.range if isinstance(self.buy_b_ema_slow, ABC) else [self.buy_b_ema_slow]:
            dataframe[f'ema_slow_{x}'] = ta.EMA(dataframe, timeperiod=x)

        for x in self.buy_b_volume_mean_slow_window.range if isinstance(self.buy_b_volume_mean_slow_window, ABC) else [self.buy_b_volume_mean_slow_window]:
            dataframe[f'volume_mean_slow_{x}'] = dataframe['volume'].rolling(window=x).mean()

        for x in self.sell_bb_middleband_window.range if isinstance(self.sell_bb_middleband_window, ABC) else [self.sell_bb_middleband_window]:
            sell_bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=x, stds=2)
            dataframe[f'bb_middleband_{x}'] = sell_bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_a_time_window = self.buy_a_time_window.value if isinstance(self.buy_a_time_window, ABC) else self.buy_a_time_window
        buy_a_bbdelta_rate = self.buy_a_bbdelta_rate.value if isinstance(self.buy_a_bbdelta_rate, ABC) else self.buy_a_bbdelta_rate
        buy_a_closedelta_rate = self.buy_a_closedelta_rate.value if isinstance(self.buy_a_closedelta_rate, ABC) else self.buy_a_closedelta_rate
        buy_a_tail_rate = self.buy_a_tail_rate.value if isinstance(self.buy_a_tail_rate, ABC) else self.buy_a_tail_rate
        buy_a_min_sell_rate = self.buy_a_min_sell_rate.value if isinstance(self.buy_a_min_sell_rate, ABC) else self.buy_a_min_sell_rate

        buy_b_ema_slow = self.buy_b_ema_slow.value if isinstance(self.buy_b_ema_slow, ABC) else self.buy_b_ema_slow
        buy_b_close_rate = self.buy_b_close_rate.value if isinstance(self.buy_b_close_rate, ABC) else self.buy_b_close_rate
        buy_b_time_window = self.buy_b_time_window.value if isinstance(self.buy_b_time_window, ABC) else self.buy_b_time_window
        buy_b_volume_mean_slow_window = self.buy_b_volume_mean_slow_window.value if isinstance(self.buy_b_volume_mean_slow_window, ABC) else self.buy_b_volume_mean_slow_window
        buy_b_volume_mean_slow_num = self.buy_b_volume_mean_slow_num.value if isinstance(self.buy_b_volume_mean_slow_num, ABC) else self.buy_b_volume_mean_slow_num

        sell_bb_middleband_window = self.sell_bb_middleband_window.value if isinstance(self.sell_bb_middleband_window, ABC) else self.sell_bb_middleband_window

        dataframe.loc[
            (  # strategy BinHV45
                    dataframe[f'lower_{buy_a_time_window}'].shift().gt(0) &
                    dataframe[f'bbdelta_{buy_a_time_window}'].gt(dataframe['close'] * buy_a_bbdelta_rate) &
                    dataframe[f'closedelta_{buy_a_time_window}'].gt(dataframe['close'] * buy_a_closedelta_rate) &
                    dataframe[f'tail_{buy_a_time_window}'].lt(dataframe[f'bbdelta_{buy_a_time_window}'] * buy_a_tail_rate) &
                    dataframe['close'].lt(dataframe[f'lower_{buy_a_time_window}'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift()) &
                    dataframe[f'bb_middleband_{sell_bb_middleband_window}'].gt(dataframe['close'] *  buy_a_min_sell_rate)
            ) |
            (  # strategy ClucMay72018
                    (dataframe['close'] < dataframe[f'ema_slow_{buy_b_ema_slow}']) &
                    (dataframe['close'] < buy_b_close_rate * dataframe[f'bb_lowerband_{buy_b_time_window}']) &
                    (dataframe['volume'] < (dataframe[f'volume_mean_slow_{buy_b_volume_mean_slow_window}'].shift(1) * buy_b_volume_mean_slow_num))
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sell_bb_middleband_window = self.sell_bb_middleband_window.value if isinstance(self.sell_bb_middleband_window, ABC) else self.sell_bb_middleband_window
        dataframe.loc[(dataframe['close'] > dataframe[f'bb_middleband_{sell_bb_middleband_window}']), 'sell'] = 1
        return dataframe
