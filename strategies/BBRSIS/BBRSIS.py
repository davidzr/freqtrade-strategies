import talib.abstract as ta
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair


class BBRSIS(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
	"0": 0.30,
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.99

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

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
    
    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

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

        # Momentum Indicator
        # ------------------------------------
        # RSIs
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma75'] = ta.SMA(dataframe, timeperiod=75)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 3)
        dataframe_medium = resample_to_interval(dataframe, self.get_ticker_indicator() * 6)
        dataframe_long = resample_to_interval(dataframe, self.get_ticker_indicator() * 10)
        
        dataframe_short['rsi'] = ta.RSI(dataframe_short, timeperiod=20)
        dataframe_medium['rsi'] = ta.RSI(dataframe_medium, timeperiod=20)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=20)
        
        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_medium)
        dataframe = resampled_merge(dataframe, dataframe_long)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=20)
        
        dataframe.fillna(method='ffill', inplace = True)
        
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

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
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['sma5'] >= dataframe['sma75']) &
                (dataframe['sma75'] >= dataframe['sma200']) &
                (dataframe['rsi'] < (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 3)] - 5)) &
                (dataframe['volume'] > 0)
            ),
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
                (dataframe['close'] > dataframe['bb_middleband']) &
                (dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*3)] + 5) &
                (dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*6)]) &
                (dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*10)]) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
