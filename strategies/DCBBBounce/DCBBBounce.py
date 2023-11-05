
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter



class DCBBBounce(IStrategy):
    """
    Simple strategy based on Contrarian Donchian Channels crossing Bollinger Bands

    How to use it?
    > python3 ./freqtrade/main.py -s DCBBBounce.py
    """

    # Hyperparameters
    # Buy hyperspace params:
    buy_params = {
        "buy_adx": 25.0,
        "buy_adx_enabled": True,
        "buy_ema_enabled": False,
        "buy_period": 52,
        "buy_sar_enabled": True,
        "buy_sma_enabled": False,
    }

    buy_period = IntParameter(10, 120, default=52, space="buy")

    buy_adx = DecimalParameter(1, 99, decimals=0, default=25, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    # set the startup candles count to the longest average used (SMA, EMA etc)
    startup_candle_count = buy_period.value

    # The ROI, Stoploss and Trailing Stop values are typically found using hyperopt
    # if hold enabled, then use the 'common' ROI params
    if sell_hold.value:
        # ROI table:
        minimal_roi = {
            "0": 0.278,
            "39": 0.087,
            "124": 0.038,
            "135": 0
        }

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.172
        trailing_stop_positive_offset = 0.212
        trailing_only_offset_is_reached = False

        # Stoploss:
        stoploss = -0.333
    else:
        # ROI table:
        minimal_roi = {
            "0": 0.261,
            "40": 0.087,
            "95": 0.023,
            "192": 0
        }

        # Stoploss:
        stoploss = -0.33

        # Trailing stop:
        trailing_stop = True
        trailing_stop_positive = 0.168
        trailing_stop_positive_offset = 0.253
        trailing_only_offset_is_reached = False


    # Optimal timeframe for the strategy
    timeframe = '5m'


    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_period.value)

        dataframe["dcbb_diff_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_diff_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])


        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # SAR Parabolic
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=200)
        #print("\nSMA: ", dataframe['sma'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS

        # check that volume is not 0 (can happen in testing, or if there are issues with exchange data)
        # conditions.append(dataframe['volume'] > 0)

        # during back testing, data can be undefined, so check
        conditions.append(dataframe['dc_upper'].notnull())

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma'])

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['ema50'].notnull())
            conditions.append(dataframe['close'] > dataframe['ema50'])

        # ADX with DM+ > DM- indicates uptrend
        if self.buy_adx_enabled.value:
            conditions.append(
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['dm_plus'] >= dataframe['dm_minus'])
            )

        # TRIGGERS
        # closing price above SAR
        #conditions.append(dataframe['sar'] < dataframe['close'])

        # green candle, Lower Bollinger goes below Donchian
        conditions.append(
            (dataframe['dcbb_diff_lower'].notnull()) &
            (dataframe['close'] >= dataframe['open']) &
            (qtpylib.crossed_above(dataframe['dcbb_diff_lower'], 0))
        )

        # build the dataframe using the conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """
        # if hold, then don't set a sell signal
        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:

            conditions = []
            # Upper Bollinger goes above Donchian
            conditions.append(
                (dataframe['dcbb_diff_upper'].notnull()) &
                #(dataframe['close'] <= dataframe['open']) &
                (qtpylib.crossed_below(dataframe['dcbb_diff_upper'], 0))
            )

            # build the dataframe using the conditions
            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
