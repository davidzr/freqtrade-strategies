import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter


###########################################################################################################
##                CombinedBinHClucAndMADV9 by ilya                                                       ##
##                                                                                                       ##
##    https://github.com/i1ya/freqtrade-strategies                                                       ##
##    The stratagy most inspired by iterativ (authors of the CombinedBinHAndClucV6)                      ##
##                                                                                                       ##                                                                                                       ##
###########################################################################################################
##     The main point of this strat is:                                                                  ##
##        -  make drawdown as low as possible                                                            ##
##        -  buy at dip                                                                                  ##
##        -  sell quick as fast as you can (release money for the next buy)                              ##
##        -  soft check if market if rising                                                              ##
##        -  hard check is market if fallen                                                              ##
##        -  10 buy signals                                                                              ##
##        -  new stoploss function, preventing from big fall                                             ##
##                                                                                                       ##
###########################################################################################################
##      How to use:                                                                                      ##
##        - no need to HyperOpt                                                                          ##
##        - before use it check the docs -                                                               ##
##          https://www.freqtrade.io/en/stable/configuration/#market-order-pricing                       ##
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
def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']


class CombinedBinHClucAndMADV9(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.028,  # I feel lucky!
        "10": 0.018,
        "40": 0.005,
    }

    stoploss = -0.99  # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001  # it doesn't meant anything, just to guarantee there is a minimal profit.
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
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    buy_params = {
        #############
        # Enable/Disable conditions
        "buy_condition_0_enable": False,
        "buy_condition_1_enable": True,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": True,
        "buy_condition_4_enable": True,
        "buy_condition_5_enable": True,
        "buy_condition_6_enable": True,
        "buy_condition_7_enable": True,
        "buy_condition_8_enable": True,
        "buy_condition_9_enable": True,
        "buy_condition_10_enable": True,
    }

    ############################################################################

    # Buy

    buy_condition_0_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.7, 1.1, default=0.99, space='buy', optimize=False, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.7, 1.1, default=0.982, space='buy', optimize=False,
                                                         load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=4, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space='buy', decimals=1, optimize=False, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space='buy', decimals=2, optimize=False, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space='buy', decimals=2, optimize=False, load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.

        if (current_profit > 0):
            return 0.99
        else:
            trade_time_50 = current_time - timedelta(minutes=50)

            # Trade open more then 60 minutes. For this strategy it's means -> loss
            # Let's try to minimize the loss

            if (trade_time_50 > trade.open_date_utc):

                try:
                    number_of_candle_shift = int((trade_time_50 - trade.open_date_utc).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    # Are we still sinking?
                    if current_rate * 1.015 < candle['open']:
                        return 0.01

                except IndexError as error:

                    # Whoops, set stoploss at 5%
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

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # EMA
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
            (
                    self.buy_condition_1_enable.value &

                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['close'] > dataframe['ema_200_1h']) &

                    (dataframe['close'] < dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &

                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe[
                        'bb_lowerband'].shift(2)) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_2_enable.value &

                    (dataframe['close'] > dataframe['ema_200']) &

                    (dataframe['close'] < dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_2.value) &

                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe[
                        'bb_lowerband'].shift(2)) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_3_enable.value &

                    (dataframe['close'] > dataframe['ema_200_1h']) &

                    (dataframe['close'] < dataframe['bb_lowerband']) &
                    (dataframe['rsi'] < self.buy_rsi_3.value) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_4_enable.value &

                    (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &

                    (dataframe['close'] < dataframe['bb_lowerband']) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_5_enable.value &

                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['close'] > dataframe['ema_200_1h']) &

                    (dataframe['ema_26'] > dataframe['ema_12']) &
                    ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
                    ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                    (dataframe['close'] < (dataframe['bb_lowerband'])) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )
            |
            (
                    self.buy_condition_6_enable.value &

                    (dataframe['ema_26'] > dataframe['ema_12']) &
                    ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
                    ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                    (dataframe['close'] < (dataframe['bb_lowerband'])) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_7_enable.value &

                    (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &

                    (dataframe['ema_26'] > dataframe['ema_12']) &
                    ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
                    ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_8_enable.value &

                    (dataframe['rsi_1h'] < self.buy_rsi_1h_3.value) &
                    (dataframe['rsi'] < self.buy_rsi_1.value) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_9_enable.value &

                    (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
                    (dataframe['rsi'] < self.buy_rsi_2.value) &

                    (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                    (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(
                        30) * self.buy_volume_pump_1.value) &
                    (dataframe['volume'] > 0)
            )
            |
            (
                    self.buy_condition_10_enable.value &

                    (dataframe['close'] < dataframe['sma_5']) &
                    (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
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
                    (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &  # Don't be gready, sell fast
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe