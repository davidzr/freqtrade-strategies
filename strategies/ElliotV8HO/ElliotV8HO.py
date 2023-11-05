# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import DecimalParameter, IntParameter


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class ElliotV8HO(IStrategy):
    INTERFACE_VERSION = 2

    # Sell hyperspace params: v1
    # sell_params = {
    #     "base_nb_candles_sell": 24,
    #     "high_offset": 0.991,
    #     "high_offset_2": 0.997
    # }

    # Sell hyperspace params: v5
    # sell_params = {
    #     "base_nb_candles_sell": 30,
    #     "high_offset": 0.973,
    #     "high_offset_2": 1.121,
    # }

    # Sell hyperspace params: v6
    sell_params = {
        "base_nb_candles_sell": 24,
        "high_offset": 1.011,
        "high_offset_2": 0.997,
    }

    # Buy hyperspace params: v1
    buy_params = {
        "base_nb_candles_buy": 19,
        "ewo_high": 5.417,
        "ewo_low": -17.251,
        "low_offset": 0.983,
        "rsi_buy": 61,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.08,
        "40": 0.032,
        "87": 0.016
    }

    # Stoploss:
    stoploss = -0.189

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        15, 60, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        15, 60, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.9, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.2, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -15.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        1.0, 8.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(
        25, 75, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        # 'sell': 'ioc'
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value == 'hyperopt':
            # Calculate all ma_buy values
            for val in self.base_nb_candles_buy.range:
                dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

            # Calculate all ma_sell values
            for val in self.base_nb_candles_sell.range:
                dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        else:
            dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] = ta.EMA(
                dataframe, timeperiod=self.base_nb_candles_buy.value)

            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] = ta.EMA(
                dataframe, timeperiod=self.base_nb_candles_sell.value)

        dataframe['hma_50'] = qtpylib.hull_moving_average(
            dataframe['close'], window=50)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (

                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))


            )
        )

        conditions.append(
            (


                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))

            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            ((dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
