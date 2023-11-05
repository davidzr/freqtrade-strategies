# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy


class EMABBRSI(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.14142349227153977,
        "25": 0.04585271106137356,
        "41": 0.020732379189664467,
        "67": 0
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.20
    # Optimal ticker interval for the strategy
    ticker_interval = '1h'

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Optional time in force for orders
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Ideas for MA crosse over strategies
        # 5 and 10 crossover
        # buys at 200 EMA re-testing or 100EMA
        # when 7 closes above 25 or sell when 7 crosses below 25
        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)

        dataframe['ema50']=ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200']=ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['rsi'] > 33) &
                # (dataframe['close'] < dataframe['bb_lowerband3'])
                # (dataframe['close'] < dataframe['bb_lowerband3'])
                qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband3'])
               
            )
            |
            (
                (dataframe['close'].shift(1) > dataframe['ema200']) &
                (dataframe['low'] < dataframe['ema200']) &
                (dataframe['close'] > dataframe['ema200']) 
            )
            |
            (
                qtpylib.crossed_above(dataframe['ema50'], dataframe['ema200'])
               
            )
           ,
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
                (
                    (dataframe['close'] > dataframe['bb_lowerband']) &
                     (dataframe['rsi'] > 91)      
                )
                |
                (
                    qtpylib.crossed_below(dataframe['ema50'], dataframe['ema200'])
                
                )
                ,
            'sell'] = 1
        return dataframe
