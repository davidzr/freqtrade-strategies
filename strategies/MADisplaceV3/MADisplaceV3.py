# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import pandas as pd

# inspired by @tirail SMAOffset

buy_params = {
    "ma_lower_length": 15,
    "ma_lower_offset": 0.96,

    "informative_fast_length": 20,
    "informative_slow_length": 25,

    "rsi_fast_length": 4,
    "rsi_fast_threshold": 35,
    "rsi_slow_length": 20,
    "rsi_slow_confirmation": 1
}

sell_params = {
    "ma_middle_1_length": 30,
    "ma_middle_1_offset": 0.995,
    "ma_upper_length": 20,
    "ma_upper_offset": 1.01,
}


class MADisplaceV3(IStrategy):

    ma_lower_length = IntParameter(15, 25, default=buy_params['ma_lower_length'], space='buy')
    ma_lower_offset = DecimalParameter(0.95, 0.97, default=buy_params['ma_lower_offset'], space='buy')

    informative_fast_length = IntParameter(15, 35, default=buy_params['informative_fast_length'], space='disable')
    informative_slow_length = IntParameter(20, 40, default=buy_params['informative_slow_length'], space='disable')

    rsi_fast_length = IntParameter(2, 8, default=buy_params['rsi_fast_length'], space='disable')
    rsi_fast_threshold = IntParameter(5, 35, default=buy_params['rsi_fast_threshold'], space='disable')
    rsi_slow_length = IntParameter(10, 45, default=buy_params['rsi_slow_length'], space='disable')
    rsi_slow_confirmation = IntParameter(1, 5, default=buy_params['rsi_slow_confirmation'], space='disable')

    ma_middle_1_length = IntParameter(15, 35, default=sell_params['ma_middle_1_length'], space='sell')
    ma_middle_1_offset = DecimalParameter(0.93, 1.005, default=sell_params['ma_middle_1_offset'], space='sell')
    ma_upper_length = IntParameter(15, 25, default=sell_params['ma_upper_length'], space='sell')
    ma_upper_offset = DecimalParameter(1.005, 1.025, default=sell_params['ma_upper_offset'], space='sell')

    minimal_roi = {"0": 1}

    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True

    plot_config = {
        'main_plot': {
            'ma_lower': {'color': 'red'},
            'ma_middle_1': {'color': 'green'},
            'ma_upper': {'color': 'pink'},
        },
    }

    use_custom_stoploss = True
    startup_candle_count = 200

    informative_timeframe = '1h'

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        if self.config['runmode'].value == 'hyperopt':
            dataframe = self.informative_dataframe.copy()
        else:
            dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=int(self.informative_fast_length.value))
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=int(self.informative_slow_length.value))

        dataframe['uptrend'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow'])
        ).astype('int')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < -0.04 and current_time - timedelta(minutes=35) > trade.open_date_utc:
            return -0.01

        return -0.99

    def get_main_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=int(self.rsi_fast_length.value))
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=int(self.rsi_slow_length.value))
        dataframe['rsi_slow_descending'] = (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift()).astype('int')

        dataframe['ma_lower'] = ta.SMA(dataframe, timeperiod=int(self.ma_lower_length.value)) * self.ma_lower_offset.value
        dataframe['ma_middle_1'] = ta.SMA(dataframe, timeperiod=int(self.ma_middle_1_length.value)) * self.ma_middle_1_offset.value
        dataframe['ma_upper'] = ta.SMA(dataframe, timeperiod=int(self.ma_upper_length.value)) * self.ma_upper_offset.value

        # drop NAN in hyperopt to fix "'<' not supported between instances of 'str' and 'int' error
        if self.config['runmode'].value == 'hyperopt':
            dataframe = dataframe.dropna()

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value == 'hyperopt':

            self.informative_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        if self.config['runmode'].value != 'hyperopt':

            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # calculate indicators with adjustable params for hyperopt
        # it's calling multiple times and dataframe overrides same columns
        # so check if any calculated column already exist

        if self.config['runmode'].value == 'hyperopt' and 'uptrend' not in dataframe:
            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)
            pd.options.mode.chained_assignment = None

        dataframe.loc[
            (
                (dataframe['rsi_slow_descending'].rolling(self.rsi_slow_confirmation.value).sum() == self.rsi_slow_confirmation.value)
                &
                (dataframe['rsi_fast'] < self.rsi_fast_threshold.value)
                &
                (dataframe['uptrend'] > 0)
                &
                (dataframe['close'] < dataframe['ma_lower'])
                &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value == 'hyperopt' and 'uptrend' not in dataframe:
            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)
            pd.options.mode.chained_assignment = None

        dataframe.loc[
            (
                (
                    (dataframe['uptrend'] == 0)
                    |
                    (dataframe['close'] > dataframe['ma_upper'])
                    |
                    (qtpylib.crossed_below(dataframe['close'], dataframe['ma_middle_1']))

                )
                &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1

        return dataframe

    def merge_informative(self, informative: DataFrame, dataframe: DataFrame) -> DataFrame:

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)

        # don't overwrite the base dataframe's HLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        return dataframe
