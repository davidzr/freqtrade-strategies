from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
import numpy as np


class BuyAllSellAllStrategy(IStrategy):
    stoploss = -0.25
    timeframe = '5m'

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["buy"] = np.random.randint(0, 2, size=len(dataframe))
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        return dataframe

    def custom_exit(
        self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if (last_candle is not None):
            return True
        return None
