from datetime import datetime, timedelta
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

# og        @tirail
# author    @Jooopieeert#0239

ma_types = {
    'SMA': ta.SMA,
    'EMA': ta.EMA,
}
class SMAOG(IStrategy):
    INTERFACE_VERSION = 2
    buy_params = {
        "base_nb_candles_buy": 26,
        "buy_trigger": "SMA",
        "low_offset": 0.968,
        "pair_is_bad_0_threshold": 0.555,
        "pair_is_bad_1_threshold": 0.172,
        "pair_is_bad_2_threshold": 0.198,
    }
    sell_params = {
        "base_nb_candles_sell": 28,
        "high_offset": 0.985,
        "sell_trigger": "EMA",
    }
    base_nb_candles_buy = IntParameter(16, 45, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False, load=True)
    base_nb_candles_sell = IntParameter(16, 45, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False, load=True)
    low_offset = DecimalParameter(0.8, 0.99, default=buy_params['low_offset'], space='buy', optimize=False, load=True)
    high_offset = DecimalParameter(0.8, 1.1, default=sell_params['high_offset'], space='sell', optimize=False, load=True)
    buy_trigger = CategoricalParameter(ma_types.keys(), default=buy_params['buy_trigger'], space='buy', optimize=False, load=True)
    sell_trigger = CategoricalParameter(ma_types.keys(), default=sell_params['sell_trigger'], space='sell', optimize=False, load=True)
    pair_is_bad_0_threshold = DecimalParameter(0.0, 0.600, default=0.220, space='buy', optimize=True, load=True)
    pair_is_bad_1_threshold = DecimalParameter(0.0, 0.350, default=0.090, space='buy', optimize=True, load=True)
    pair_is_bad_2_threshold = DecimalParameter(0.0, 0.200, default=0.060, space='buy', optimize=True, load=True)

    timeframe = '5m'
    stoploss = -0.23
    minimal_roi = {"0": 10,}
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True
    startup_candle_count = 400

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value
            dataframe['ma_offset_sell'] = ma_types[self.sell_trigger.value](dataframe, int(self.base_nb_candles_sell.value)) * self.high_offset.value
            dataframe['pair_is_bad'] = (
                    (((dataframe['open'].rolling(144).min() - dataframe['close']) / dataframe[
                        'close']) >= self.pair_is_bad_0_threshold.value) |
                    (((dataframe['open'].rolling(12).min() - dataframe['close']) / dataframe[
                        'close']) >= self.pair_is_bad_1_threshold.value) |
                    (((dataframe['open'].rolling(2).min() - dataframe['close']) / dataframe[
                        'close']) >= self.pair_is_bad_2_threshold.value)).astype('int')
            dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
            dataframe['rsi_exit'] = ta.RSI(dataframe, timeperiod=2)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value
            dataframe['pair_is_bad'] = (
                    (((dataframe['open'].rolling(144).min() - dataframe['close']) / dataframe[
                        'close']) >= self.pair_is_bad_0_threshold.value) |
                    (((dataframe['open'].rolling(12).min() - dataframe['close']) / dataframe[
                        'close']) >= self.pair_is_bad_1_threshold.value) |
                    (((dataframe['open'].rolling(2).min() - dataframe['close']) / dataframe[
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
            dataframe['ma_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) * self.high_offset.value
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['ma_offset_sell']) &
                    (
                        (dataframe['open'] < dataframe['open'].shift(1)) |
                        (dataframe['rsi_exit'] < 50) |
                        (dataframe['rsi_exit'] < dataframe['rsi_exit'].shift(1))
                    ) &
                    (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
