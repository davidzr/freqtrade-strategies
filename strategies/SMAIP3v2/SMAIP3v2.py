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


class SMAIP3v2(IStrategy):
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

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 26,
        "high_offset": 0.985,
        "sell_trigger": "EMA",
    }

    # Stoploss:
    stoploss = -0.23
#    stoploss = -0.15

    # ROI table:
    minimal_roi = {
        "0": 0.026
    }

    base_nb_candles_buy = IntParameter(16, 60, default=buy_params['base_nb_candles_buy'], space='buy')
    base_nb_candles_sell = IntParameter(16, 60, default=sell_params['base_nb_candles_sell'], space='sell')
    low_offset = DecimalParameter(0.8, 0.99, default=buy_params['low_offset'], space='buy')
    high_offset = DecimalParameter(0.8, 1.1, default=sell_params['high_offset'], space='sell')
    buy_trigger = CategoricalParameter(ma_types.keys(), default=buy_params['buy_trigger'], space='buy')
    sell_trigger = CategoricalParameter(ma_types.keys(), default=sell_params['sell_trigger'], space='sell')

    pair_is_bad_1_threshold = DecimalParameter(0.00, 0.30, default=0.200, space='buy')
    pair_is_bad_2_threshold = DecimalParameter(0.00, 0.25, default=0.072, space='buy')

    # Trailing stop:
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.018

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]

        if (last_candle is not None):
            if (sell_reason in ['roi','sell_signal','trailing_stop_loss']):
                if (last_candle['open'] > previous_candle_1['open']) and (last_candle['rsi'] > 50) and (last_candle['rsi'] > previous_candle_1['rsi']):
                    return False
        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # confirm_trade_exit
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=2)

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

        dataframe.loc[
            (
                    (dataframe['ema_50'] > dataframe['ema_200']) &
                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['pair_is_bad'] < 1) &
                    (dataframe['close'] < dataframe['ma_offset_buy']) &
                    (dataframe['volume'] > 0)
#                    & dataframe['btc_up']
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
