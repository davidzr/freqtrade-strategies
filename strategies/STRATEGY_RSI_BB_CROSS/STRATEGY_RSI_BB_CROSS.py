# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


_trend_length = 14

class STRATEGY_RSI_BB_CROSS(IStrategy):
    """
    Strategy RSI_BB_CROSS
    author@: Fractate_Dev
    github@: https://github.com/Fractate/freqbot
    How to use it?
    > python3 ./freqtrade/main.py -s RSI_BB_CROSS
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            # "MACD": {
            #     'macd': {'color': 'blue'},
            #     'macdsignal': {'color': 'orange'},
            # },
            "BB": {
                'bb_percent': {'color': 'red'},
                '1': {},
                '0': {},
            },
            # "RSI": {
            #     'rsi': {'color': 'red'},
            #     '70': {},
            #     '30': {},
            # },
            "RSI_Percent": {
                'rsi_percent': {'color': 'red'},
                '1': {},
                '0': {},
            },
            # "bb_above": {
            #     'bb_above_rsi': {}
            # },
            # "bb_below": {
            #     'bb_below_rsi': {}
            # },
            "bb_minus_rsi_percent": {
                'bb_minus_rsi_percent': {},
                '0': {},
            },
            "bb_rsi_count": {
                'bb_above_rsi_count': {},
                'bb_below_rsi_count': {},
            },
        }
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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        # remove later
        for i in range(1):
            print("")

        # Logic
        # Bollinger Band getting less than RSI should occur after a certain number of candles pass
        # time frame of 1 hour

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        dataframe['bb_percent'] = (dataframe['close'] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['70'] = 70
        dataframe['30'] = 30
        dataframe['1'] = 1
        dataframe['0'] = 0

        rsi_limit = 30
        dataframe['rsi_percent'] = (dataframe['rsi'] - rsi_limit) / (100 - rsi_limit * 2)

        dataframe['bb_minus_rsi_percent'] = dataframe['bb_percent'] - dataframe['rsi_percent']

        # Verify RSI positivity or negativity over trend
        dataframe['bb_above_rsi_count'] = True
        dataframe['bb_below_rsi_count'] = True
        for i in range(_trend_length):
            
            dataframe['bb_above_rsi_count'] = (dataframe['bb_minus_rsi_percent'].shift(i) > 0) & dataframe['bb_above_rsi_count']
                # dataframe['bb_above_rsi_count'] += 1
            dataframe['bb_below_rsi_count'] = (dataframe['bb_minus_rsi_percent'].shift(i) < 0) & dataframe['bb_below_rsi_count']
                # dataframe['bb_below_rsi_count'] += 1

        

                
        ## old method
        # bb_above_rsi = False
        # bb_below_rsi = False
        # dataframe['bb_above_rsi_count'] = 0
        # dataframe['bb_below_rsi_count'] = 0

        
        # for i in range(_trend_length):
        #     if np.where(dataframe['bb_percent'].shift(i) < dataframe['rsi_percent'].shift(i)):
        #         dataframe['bb_below_rsi_count'] += 1
        #     if np.where(dataframe['bb_percent'].shift(i) > dataframe['rsi_percent'].shift(i)):
        #         dataframe['bb_above_rsi_count'] += 1


        # if bb_above_rsi == True & bb_below_rsi == False:
        # dataframe['bb_above_rsi'] = 1
        # if bb_above_rsi == False & bb_below_rsi == True:
        #     dataframe['bb_below_rsi'] = 1


        # print(type(dataframe['bb_below_rsi']))

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                # (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                # (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                # (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                # (dataframe['volume'] > 0)  # Make sure Volume is not 0
                qtpylib.crossed_above(dataframe['bb_percent'], dataframe['rsi_percent']) &
                (dataframe['bb_percent'] < 0.5) & 
                (dataframe['rsi_percent'] < 0.5) &
                (dataframe['bb_below_rsi_count'].shift(1))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                # (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                # (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                # (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                # (dataframe['volume'] > 0)  # Make sure Volume is not 0
                qtpylib.crossed_below(dataframe['bb_percent'], dataframe['rsi_percent']) &
                (dataframe['bb_percent'] > 0.5) & 
                (dataframe['rsi_percent'] > 0.5) &
                (dataframe['bb_above_rsi_count'].shift(1))
            ),
            'sell'] = 1
        return dataframe
    