import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta




class NormalizerStrategy(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.18
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 610

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (current_time - timedelta(minutes=300) > trade.open_date_utc):
            return 0.01
        return 0.99

    def fischer_norm(self, x, lookback):
        res = np.zeros_like(x)
        for i in range(lookback, len(x)):
            x_min = np.min(x[i-lookback: i +1])
            x_max = np.max(x[i-lookback: i +1])
            #res[i] = (2*(x[i] - x_min) / (x_max - x_min)) - 1
            res[i] = (x[i] - x_min) / (x_max - x_min)
        return res
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lookback = [13, 21, 34, 55, 89, 144, 233, 377, 610]
        for look in lookback:
            dataframe[f"norm_{look}"] = self.fischer_norm(dataframe["close"].values, look)
        collist = [col for col in dataframe.columns if col.startswith("norm")]
        dataframe["pct_sum"] = dataframe[collist].sum(axis=1)



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['pct_sum'] < .2) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
            ,
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['pct_sum'] > 8) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
            ,
            'sell'
        ] = 1
        return dataframe