import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta
from functools import reduce


###########################################################################################################
##                CombinedBinHAndClucV7 by iterativ                                                      ##
##                                                                                                       ##
##    Freqtrade https://github.com/freqtrade/freqtrade                                                   ##
##    The authors of the original CombinedBinHAndCluc https://github.com/freqtrade/freqtrade-strategies  ##
##    V7 by iterativ.                                                                                    ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 20 to 60 pairs. Volume pairlist works well.                                         ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                    ##
##   Ensure that you don't override any variables in you config.json. Especially                         ##
##   the timeframe (must be 5m) & sell_profit_only (must be true).                                       ##
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


class CombinedBinHAndClucV7(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.0181
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h' # informative tf

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # Custom stoploss
    use_custom_stoploss = True

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

    # Buy Hyperopt params

    buy_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.14, space='buy', decimals=2, optimize=False, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.02, 0.4, default=0.34, space='buy', decimals=2, optimize=False, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.25, 0.44, default=0.38, space='buy', decimals=2, optimize=False, load=True)

    buy_bb40_bbdelta_close = DecimalParameter(0.005, 0.04, default=0.031, space='buy', optimize=True, load=True)
    buy_bb40_closedelta_close = DecimalParameter(0.01, 0.03, default=0.021, space='buy', optimize=True, load=True)
    buy_bb40_tail_bbdelta = DecimalParameter(0.2, 0.4, default=0.264, space='buy', optimize=True, load=True)

    buy_bb20_close_bblowerband = DecimalParameter(0.8, 1.1, default=0.992, space='buy', optimize=True, load=True)
    buy_bb20_volume = IntParameter(18, 36, default=29, space='buy', optimize=True, load=True)

    buy_rsi_diff = DecimalParameter(34.0, 60.0, default=50.48, space='buy', decimals=2, optimize=True, load=True)

    buy_min_inc = DecimalParameter(0.005, 0.05, default=0.01, space='buy', decimals=2, optimize=True, load=True)
    buy_rsi_1h = DecimalParameter(40.0, 70.0, default=67.0, space='buy', decimals=2, optimize=True, load=True)
    buy_rsi = DecimalParameter(30.0, 40.0, default=38.5, space='buy', decimals=2, optimize=True, load=True)
    buy_mfi = DecimalParameter(36.0, 65.0, default=36.0, space='buy', decimals=2, optimize=True, load=True)

    # Sell Hyperopt params

    sell_roi_profit_1 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=2, optimize=False, load=True)
    sell_roi_rsi_1 = DecimalParameter(30.0, 38.0, default=34, space='sell', decimals=2, optimize=False, load=True)
    sell_roi_profit_2 = DecimalParameter(0.02, 0.05, default=0.03, space='sell', decimals=2, optimize=False, load=True)
    sell_roi_rsi_2 = DecimalParameter(34.0, 44.0, default=38, space='sell', decimals=2, optimize=False, load=True)
    sell_roi_profit_3 = DecimalParameter(0.0, 0.0, default=0.0, space='sell', decimals=2, optimize=False, load=True)
    sell_roi_rsi_3 = DecimalParameter(48.0, 56.0, default=50, space='sell', decimals=2, optimize=False, load=True)

    sell_rsi_main = DecimalParameter(72.0, 90.0, default=77, space='sell', decimals=2, optimize=True, load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (current_time - timedelta(minutes=280) > trade.open_date_utc):
            return 0.01
        return 0.99

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # Prevent sell, if there is more potential, in order to maximize profit
        if (last_candle is not None):
            current_profit = trade.calc_profit_ratio(rate)
            if (sell_reason == 'roi'):
                if (current_profit > self.sell_roi_profit_1.value):
                    if (last_candle['rsi'] > self.sell_roi_rsi_1.value):
                        return False
                elif (current_profit > self.sell_roi_profit_2.value):
                    if (last_candle['rsi'] > self.sell_roi_rsi_2.value):
                        return False
                elif (current_profit > self.sell_roi_profit_3.value):
                    if (last_candle['rsi'] > self.sell_roi_rsi_3.value):
                        return False
        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
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

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['sma_200'] = ta.EMA(dataframe, timeperiod=200)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

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
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &

                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bb40_bbdelta_close.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_bb40_closedelta_close.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_bb40_tail_bbdelta.value) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &

                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < self.buy_bb20_close_bblowerband.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * self.buy_bb20_volume.value))
            )
        )

        conditions.append(
            (
                (dataframe['close'] < dataframe['sma_5']) &
                (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &

                (dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_rsi_diff.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['sma_200'] > dataframe['sma_200'].shift(20)) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(16)) &

                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
                (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &

                (((dataframe['open'].rolling(24).min() - dataframe['close']) / dataframe['close']) > self.buy_min_inc.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h.value) &
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['mfi'] < self.buy_mfi.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['rsi'] > self.sell_rsi_main.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe
