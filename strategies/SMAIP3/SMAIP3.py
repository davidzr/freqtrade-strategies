# --- Do not remove these libs ---
# --------------------------------

from datetime import datetime, timedelta

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy

# author @tirail

ma_types = {
    'SMA': ta.SMA,
    'EMA': ta.EMA,
}


class SMAIP3(IStrategy):
    INTERFACE_VERSION = 2

    # hyperopt and paste results here
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 18,
        "buy_trigger": "SMA",
        "low_offset": 0.968,
        "pair_is_bad_1_threshold": 0.130,
        "pair_is_bad_2_threshold": 0.075,
    }

    # #########################################################
    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 55,
        "high_offset": 1.07,
        "sell_trigger": "EMA",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.135,
        "35": 0.061,
        "86": 0.037,
        "167": 0
    }

    # Stoploss:
    stoploss = -0.331

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.098
    trailing_stop_positive_offset = 0.159
    trailing_only_offset_is_reached = True

    base_nb_candles_buy = IntParameter(
        16, 60, default=buy_params['base_nb_candles_buy'], space='buy')
    base_nb_candles_sell = IntParameter(
        16, 60, default=sell_params['base_nb_candles_sell'], space='sell')
    low_offset = DecimalParameter(
        0.8, 0.99, default=buy_params['low_offset'], space='buy')
    high_offset = DecimalParameter(
        0.8, 1.1, default=sell_params['high_offset'], space='sell')
    buy_trigger = CategoricalParameter(
        ma_types.keys(), default=buy_params['buy_trigger'], space='buy')
    sell_trigger = CategoricalParameter(
        ma_types.keys(), default=sell_params['sell_trigger'], space='sell')

    pair_is_bad_1_threshold = DecimalParameter(
        0.00, 0.30, default=0.200, space='buy')
    pair_is_bad_2_threshold = DecimalParameter(
        0.00, 0.25, default=0.072, space='buy')

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    process_only_new_candles = True
    startup_candle_count = 30

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe,
                                                                          int(self.base_nb_candles_buy.value)) * self.low_offset.value
            dataframe['ma_offset_sell'] = ma_types[self.sell_trigger.value](dataframe,
                                                                            int(self.base_nb_candles_sell.value)) * self.high_offset.value

            dataframe['pair_is_bad'] = (
                (((dataframe['open'].shift(12) - dataframe['close']) / dataframe[
                    'close']) >= self.pair_is_bad_1_threshold.value) |
                (((dataframe['open'].shift(6) - dataframe['close']) / dataframe[
                    'close']) >= self.pair_is_bad_2_threshold.value)).astype('int')

            dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe,
                                                                          int(self.base_nb_candles_buy.value)) * self.low_offset.value
            dataframe['pair_is_bad'] = (
                (((dataframe['open'].shift(12) - dataframe['close']) / dataframe[
                    'close']) >= self.pair_is_bad_1_threshold.value) |
                (((dataframe['open'].shift(6) - dataframe['close']) / dataframe[
                    'close']) >= self.pair_is_bad_2_threshold.value)).astype('int')

            dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe.loc[
            (
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['pair_is_bad'] < 1) &
                (dataframe['close'] < dataframe['ma_offset_buy']) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_sell'] = ma_types[self.sell_trigger.value](dataframe,
                                                                            int(self.base_nb_candles_sell.value)) * self.high_offset.value

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ma_offset_sell']) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
