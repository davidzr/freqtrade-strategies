import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

class Cluc4(IStrategy):
    minimal_roi = {
        "0": 0.015,
        "20": 0.005,
        "30": 0.001
    }

    stoploss = -0.01
    
    timeframe = '1m'

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe, timeperiod=28)
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['rocr'] = ta.ROCR(informative, timeperiod=168)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(0.65)
            ) &
            ((      dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * 0.006) &
                    dataframe['closedelta'].gt(dataframe['close'] * 0.013) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * 0.968) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (       (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < 0.013 * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 28))
            )),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            #((qtpylib.crossed_above(dataframe['close'],(dataframe['bb_middleband'] * 1.1))) &
            ((qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband'])) &
            (dataframe['volume'] > 0))
            ,
            'sell'
        ] = 1
        return dataframe