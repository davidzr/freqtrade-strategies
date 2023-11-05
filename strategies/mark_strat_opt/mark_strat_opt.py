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

class mark_strat_opt (IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md
    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy
    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "0": 0.50503, 
    #     "304": 0.13111, 
    #     "669": 0.03614, 
    #     "1039": 0
    #     }
    minimal_roi = {
        "0": 0.06965, 
        "8": 0.01125, 
        "19": 0.00498, 
        "34": 0
        }
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20206
# {   'trailing_only_offset_is_reached': False,
#     'trailing_stop': True,
#     'trailing_stop_positive': 0.0916,
#     'trailing_stop_positive_offset': 0.11514}
    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.07573
    trailing_stop_positive_offset = 0.09805  
    # Optimal ticker interval for the strategy.
    ticker_interval = '1m'
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
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
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
        # Momentum Indicators
        # ------------------------------------
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['sar'] = ta.SAR(dataframe)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        # Bollinger Bands 1 STD
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        # dataframe['bb_middleband1'] = bollinger1['mid']
        # dataframe['bb_upperband1'] = bollinger1['upper']
        # bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband2'] = bollinger2['lower']
        # dataframe['bb_middleband2'] = bollinger2['mid']
        # dataframe['bb_upperband2'] = bollinger2['upper']
        # bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        # dataframe['bb_lowerband3'] = bollinger3['lower']
        # dataframe['bb_middleband3'] = bollinger3['mid']
        # dataframe['bb_upperband3'] = bollinger3['upper']
        # Bollinger Bands 4 STD
        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        # dataframe['bb_middleband4'] = bollinger4['mid']
        # dataframe['bb_upperband4'] = bollinger4['upper']
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
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
                (dataframe['adx'] > 49) &
                (dataframe['fastd'] < 30) &
                (dataframe['mfi'] < 23) &
                (dataframe['rsi'] < 34) &
                (dataframe['close'] < dataframe['bb_lowerband4'])
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
                (dataframe['adx'] < 95) &
                (dataframe['fastd'] > 70) &
                (dataframe['rsi'] > 94) &
                (qtpylib.crossed_above(dataframe['sar'], dataframe['close']))
            ),
            'sell'] = 1
        return dataframe