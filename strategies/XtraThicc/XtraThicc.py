# -*- coding: utf-8 -*-
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np
# --------------------------------

"""

XtraThicc v69
"""

class XtraThicc(IStrategy):

    minimal_roi = {
         "0": 100
    }

    # Stoploss:
    stoploss = -0.10

    timeframe = '5m'
    inf_timeframe = '1h'

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    trailing_stop = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    startup_candle_count: int = 72
    process_only_new_candles = False

    def informative_pairs(self):
        # add all whitelisted pairs on informative timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base Thiccness
        dataframe['color'] = np.where(dataframe['close'] > dataframe['open'], 'green', 'red')

        dataframe['how-thicc'] = (dataframe['close'] - dataframe['open']).abs()
        dataframe['avg-thicc'] = dataframe['how-thicc'].abs().rolling(36).mean()
        dataframe['not-thicc'] = dataframe['how-thicc'] < (dataframe['avg-thicc'])
        dataframe['rly-thicc'] = dataframe['how-thicc'] > (dataframe['avg-thicc'])
        dataframe['xtra-thicc'] = np.where(dataframe['rly-thicc'].rolling(8).sum() >= 5,1,0)

        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['3d-low'] = informative['close'].rolling(72).min()
        informative['3d-high'] = informative['close'].rolling(72).max()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['color'] == 'green') &
                (dataframe['color'].shift(1) == 'green') &
                (dataframe['color'].shift(2) == 'red') &
                (dataframe['color'].shift(3) == 'red') &
                (dataframe['xtra-thicc'] == 1) &
                (dataframe['rly-thicc'] == 0) &
                (dataframe['close'] > dataframe[f"3d-low_{self.inf_timeframe}"]) &
                (dataframe['close'] < dataframe[f"3d-high_{self.inf_timeframe}"])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['sell'] = 0

        return dataframe

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
                    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0.01 and last_candle['roc'] < 0.5:
            return 'rode_that_ass'

        return None