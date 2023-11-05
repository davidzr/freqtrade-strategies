import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter
from pandas import DataFrame
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade


###########################################################################################################
##                NostalgiaForInfinityV2 by iterativ                                                     ##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 20 to 60 pairs. Volume pairlist works well.                                         ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                    ##
##   Ensure that you don't override any variables in you config.json. Especially                         ##
##   the timeframe (must be 5m).                                                                         ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS                                                                               ##
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

class NostalgiaForInfinityV2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 10
    }

    stoploss = -1.0

    timeframe = '5m'
    inf_1h = '1h'

    custom_info = {}

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    # Custom stoploss
    use_custom_stoploss = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Buy  params
    buy_params = {
        "buy_bb40_bbdelta_close": 0.017,
        "buy_bb40_closedelta_close": 0.013,
        "buy_bb40_tail_bbdelta": 0.445,

        "buy_bb20_close_bblowerband": 0.992,
        "buy_bb20_volume": 27,

        "buy_rsi_diff": 52.438,
    }

    # Sell params
    sell_params = {
        "sell_rsi_bb": 79.706,

        "sell_rsi_main": 85.023,

        "sell_rsi_2": 87.545,

        "sell_rsi_diff": 0.873,
        "sell_ema_relative": 0.03,
    }

    buy_bb40_bbdelta_close = DecimalParameter(0.005, 0.05, default=0.049, space='buy', optimize=True, load=True)
    buy_bb40_closedelta_close = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=True, load=True)
    buy_bb40_tail_bbdelta = DecimalParameter(0.15, 0.45, default=0.287, space='buy', optimize=True, load=True)
    buy_bb20_close_bblowerband = DecimalParameter(0.8, 1.1, default=0.991, space='buy', optimize=True, load=True)
    buy_bb20_volume = IntParameter(18, 34, default=20, space='buy', optimize=True, load=True)
    buy_rsi_diff = DecimalParameter(36.0, 54.0, default=37.0, space='buy', optimize=True, load=True)

    sell_rsi_bb = DecimalParameter(60.0, 80.0, default=70, space='sell', optimize=True, load=True)
    sell_rsi_main = DecimalParameter(72.0, 90.0, default=78.0, space='sell', optimize=True, load=True)
    sell_rsi_2 = DecimalParameter(72.0, 90.0, default=60.0, space='sell', optimize=True, load=True)
    sell_ema_relative = DecimalParameter(0.005, 0.1, default=0.03, space='sell', optimize=False, load=True)
    sell_rsi_diff = DecimalParameter(0.0, 5.0, default=5.0, space='sell', optimize=True, load=True)


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
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
        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbdelta'] = (bb_40['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # EMA
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

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
            (
                (dataframe['close'] < dataframe['sma_9']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bb40_bbdelta_close.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_bb40_closedelta_close.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_bb40_tail_bbdelta.value) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0)
            )
            |
            (
                (dataframe['close'] < dataframe['sma_9']) &
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < self.buy_bb20_close_bblowerband.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * self.buy_bb20_volume.value))
            )
            |
            (
                (dataframe['close'] < dataframe['sma_5']) &
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_rsi_diff.value) &
                (dataframe['volume'] > 0)
            )
            ,
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi_bb.value) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['volume'] > 0)
            )
            |
            (
                (dataframe['rsi'] > self.sell_rsi_main.value) &
                (dataframe['volume'] > 0)
            )
            |
            (
                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] > self.sell_rsi_2.value) &
                (dataframe['volume'] > 0)
            )
            |
            (
                (dataframe['close'] < dataframe['ema_200']) &
                (((dataframe['ema_200'] - dataframe['close']) / dataframe['close']) < self.sell_ema_relative.value) &

                (dataframe['rsi'] > dataframe['rsi_1h'] + self.sell_rsi_diff.value) &
                (dataframe['volume'] > 0)
            )
            ,
            'sell'
        ] = 1
        return dataframe
