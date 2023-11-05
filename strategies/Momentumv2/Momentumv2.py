from pandas import DataFrame
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy import (IntParameter, DecimalParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Momentumv2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.22,
        "1260": 0.17,
        "1944": 0.09,
        "7200": 0
    }

    stoploss = -0.08
    use_custom_stoploss = True
    trailing_stop = False
    timeframe = '4h'
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count: int = 100
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Buy Parameters
    buy_ema = IntParameter(10, 100, default=30, space='buy', optimize=True, load=True)

    # Sell Parameters
    sell_rsi = DecimalParameter(70, 99, default=80, space='sell', optimize=True, load=True)

    # Stoploss Parameters
    atr_timeperiod = IntParameter(5, 21, default=7, space='sell')
    atr_multiplier = DecimalParameter(2.5, 3.5, default=2.5, space='sell')

    buy_params = {
        "buy_ema": 80
    }

    sell_params = {
        "sell_rsi": 90,
        "atr_multiplier": 2.6,
        "atr_timeperiod": 12,
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 6
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_ema.value)

        # Average True Index Trailing Stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_timeperiod.value)
        dataframe['atr_trailing'] = dataframe['close'] - \
            (dataframe['atr'] * self.atr_multiplier.value)

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        stoploss_price = last_candle['atr_trailing']

        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1

        return 1

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
        conditions.append(dataframe['close'] > dataframe['ema'])
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']) | (
            qtpylib.crossed_below(dataframe['rsi'], self.sell_rsi.value)))
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe