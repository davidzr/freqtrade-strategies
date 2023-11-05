# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame


# The main idea is to buy only when overall uptrend in higher informative
# but in local dip cause BinCluc some king of pullback strategy.


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


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


class CombinedBinHAndClucV2(IStrategy):
    minimal_roi = {
        '120': 0.01,
        '60': 0.02,
        '30': 0.05,
        '0': 0.1
    }

    informative_timeframe = '1h'
    timeframe = '5m'

    stoploss = -0.05

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 2,
            "stop_duration_candles": 12,
            "only_per_pair": True
        }
    ]

    startup_candle_count = 200

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    @staticmethod
    def get_informative_indicators(dataframe: DataFrame, metadata: dict):

        ssl_down, ssl_up = SSLChannels(dataframe, 25)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_high'] = (ssl_up > ssl_down).astype('int') * 3
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=30)
        stoch = ta.STOCHRSI(dataframe, 30, 35, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=3)

        dataframe['go_long'] = (
            (dataframe['ssl_high'] > 0)
            &
            (dataframe['mfi'].shift().rolling(3).mean() > dataframe['mfi'])
            &
            (dataframe['srsi_fk'].shift().rolling(3).mean() > dataframe['srsi_fk'])
        ).astype('int') * 4

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        informative = self.get_informative_indicators(informative.copy(), metadata)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)
        # don't overwrite the base dataframe's HLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        typical_price = qtpylib.typical_price(dataframe)

        # strategy BinHV45
        bollinger_b = qtpylib.bollinger_bands(typical_price, window=40, stds=2)
        mid = bollinger_b['mid']
        lower = bollinger_b['lower']
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # strategy ClucMay72018
        bollinger_c = qtpylib.bollinger_bands(typical_price, window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger_c['lower']
        dataframe['bb_middleband'] = bollinger_c['mid']
        dataframe['bb_upperband'] = bollinger_c['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['go_long'] > 0)
            &
            (
                (  # strategy BinHV45
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * 0.008) &
                    dataframe['closedelta'].gt(dataframe['close'] * 0.0175) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * 0.25) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
                ) |
                (  # strategy ClucMay72018
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20))
                )
             ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband']))
            ),
            'sell'
        ] = 1
        return dataframe