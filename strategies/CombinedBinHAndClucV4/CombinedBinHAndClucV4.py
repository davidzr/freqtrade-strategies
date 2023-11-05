# Definition of Bollinger Bands
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class CombinedBinHAndClucV4(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.019
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (  # strategy BinHV45
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * 0.008) &
                dataframe['closedelta'].gt(dataframe['close'] * 0.0175) &
                dataframe['tail'].lt(dataframe['bbdelta'] * 0.25) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (  # strategy ClucMay72018
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ( # Improves the profit slightly.
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['high'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['high'].shift(3) > dataframe['bb_upperband'].shift(3)) &
                (dataframe['high'].shift(4) > dataframe['bb_upperband'].shift(4)) &
                (dataframe['high'].shift(5) > dataframe['bb_upperband'].shift(5)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe
