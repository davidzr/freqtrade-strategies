# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, DecimalParameter, IntParameter, CategoricalParameter

# @Rallipanos


def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class NotAnotherSMAOffsetStrategyLite(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        'base_nb_candles_buy': 14,
        'low_offset': 0.975,
    }

    # Sell hyperspace params:
    sell_params = {
        'base_nb_candles_sell': 24,
        'high_offset': 0.991,
    }

    minimal_roi = {'0': 0.025}
    stoploss = -0.1
    # use_custom_stoploss = True

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    order_time_in_force = {'buy': 'gtc', 'sell': 'ioc'}
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit < -0.05 and current_time - timedelta(minutes=720) > trade.open_date_utc:
            return -0.01
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for length in set(list(self.base_nb_candles_buy.range) + list(self.base_nb_candles_sell.range)):
            dataframe[f'ema_{length}'] = ta.EMA(dataframe, timeperiod=length)
        dataframe['ewo'] = ewo(dataframe, self.fast_ewo, self.slow_ewo)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            (dataframe['close'] < (dataframe[f'ema_{self.base_nb_candles_buy.value}'] * self.low_offset.value))
            &
            (dataframe['ewo'] > 0)
            &
            (dataframe['volume'] > 0)
        ), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            (dataframe['close'] > (dataframe[f'ema_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            &
            (dataframe['volume'] > 0)
        ), 'sell'] = 1

        return dataframe
