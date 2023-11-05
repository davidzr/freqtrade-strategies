import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta


"""
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2021-05-01 00:00:00 |
| Backtesting to        | 2021-06-09 17:00:00 |
| Max open trades       | 10                  |
|                       |                     |
| Total trades          | 447                 |
| Starting balance      | 1000.000 USDT       |
| Final balance         | 1714.575 USDT       |
| Absolute profit       | 714.575 USDT        |
| Total profit %        | 71.46%              |
| Trades per day        | 11.46               |
| Avg. stake amount     | 180.278 USDT        |
| Total trade volume    | 80584.326 USDT      |
|                       |                     |
| Best Pair             | ALICE/USDT 24.7%    |
| Worst Pair            | HARD/USDT -35.15%   |
| Best trade            | PSG/USDT 17.98%     |
| Worst trade           | XVS/USDT -26.03%    |
| Best day              | 351.588 USDT        |
| Worst day             | -256.636 USDT       |
| Days win/draw/lose    | 25 / 8 / 4          |
| Avg. Duration Winners | 1:36:00             |
| Avg. Duration Loser   | 9:33:00             |
|                       |                     |
| Min balance           | 962.929 USDT        |
| Max balance           | 1714.575 USDT       |
| Drawdown              | 240.78%             |
| Drawdown              | 289.267 USDT        |
| Drawdown high         | 252.196 USDT        |
| Drawdown low          | -37.071 USDT        |
| Drawdown Start        | 2021-05-19 03:00:00 |
| Drawdown End          | 2021-05-19 20:00:00 |
| Market change         | -34.99%             |
===============================================

"""


class NormalizerStrategyHO2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.35,
        "405": 0.248,
        "875": 0.091,
        "1585": 0
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.3
    trailing_stop_positive_offset = 0.379
    trailing_only_offset_is_reached = False

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
