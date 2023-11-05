import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair


###########################################################################################################
##                CombinedBinHClucAndMADV5 by ilya                                                       ##
##                                                                                                       ##
##    https://github.com/i1ya/freqtrade-strategies                                                       ##
##    The stratagy most inspired by iterativ (authors of the CombinedBinHAndClucV6)                      ##
##                                                                                                       ##                                                                                                       ##
###########################################################################################################
##     The main point of this strat is:                                                                  ##
##        -  buy at dip                                                                                  ##
##        -  sell quick as fast as you can (release money for the next buy)                              ##
##        -  soft check if market if rising                                                              ##
##        -  hard check is market if fallen                                                              ##
##                                                                                                       ##
###########################################################################################################
##                 GENERAL RECOMMENDATIONS                                                               ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 2 and 4 open trades, with unlimited stake.        ##
##   With my pairlist which can be found in this repo.                                                   ##
##                                                                                                       ##
##   Ensure that you don't override any variables in your config.json. Especially                        ##
##   the timeframe (must be 5m).                                                                         ##
##                                                                                                       ##
##   sell_profit_only:                                                                                   ##
##       True - risk more (gives you higher profit and higher Drawdown)                                  ##
##       False (default) - risk less (gives you less ~10-15% profit and much lower Drawdown)             ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS 2 @iterativ (author of the original strategy)                                 ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH: 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                                     ##
##                                                                                                       ##
###########################################################################################################

# SSL Channels
def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class CombinedBinHClucAndMADV5(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.021,
        "40": 0.005,
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

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

        if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
            return 0.01
        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # SSL Channels
        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h['ssl_down'] = ssl_down_1h
        informative_1h['ssl_up'] = ssl_up_1h

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

         # strategy BinHV45
        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbdelta'] = (bb_40['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # EMA
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[

               # When market at bull mode (guard)
            (  # strategy ClucMay72018
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.99 * dataframe['bb_lowerband']) &                   # Guard is on, candle should dig not so hard (0,99)
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 21)) &
                (dataframe['volume'] > 0)
            )
            |
               # When market at bear mode (without guard)
            (  # strategy ClucMay72018 
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.975 * dataframe['bb_lowerband']) &                  # Guard is off, candle should dig hard (0,975) 
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                 # Don't buy if someone drop the market.
                (dataframe['rsi_1h'] < 15) &                                                # Buy only at dip
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |   
               # When market at bull mode (guard)  
            (  # strategy MACD Low buy 
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                 # Don't buy if someone drop the market.
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
               # When market at bear mode (without guard)    
            (  # strategy MACD Low buy 
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.03)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                 # Don't buy if someone drop the market.
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (
                (dataframe['close'] < dataframe['sma_5']) &
                (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['rsi'] < dataframe['rsi_1h'] - 43.276) &
                (dataframe['volume'] > 0)
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &                  # Don't be gready, sell fast
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe
