from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicators import RMI


# The main idea of this strategy is to buy in dips and sell after recovery.


def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')


class TheRealPullbackV2(IStrategy):

    minimal_roi = {
        "0": 100
    }

    stoploss = -0.035

    timeframe = '5m'

    process_only_new_candles = True
    ignore_roi_if_buy_signal = True
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe['bb_bottom_cross'] = qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband']).astype('int')

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)

        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['cci'] = ta.CCI(dataframe, 30)

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        dataframe['cmf'] = chaikin_mf(dataframe)

        dataframe['rmi'] = RMI(dataframe, length=8, mom=4)

        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        dataframe['fastEMA'] = ta.EMA(dataframe['volume'], timeperiod=12)
        dataframe['slowEMA'] = ta.EMA(dataframe['volume'], timeperiod=26)
        dataframe['pvo'] = ((dataframe['fastEMA'] - dataframe['slowEMA']) / dataframe['slowEMA']) * 100

        dataframe['is_dip'] = (
            (dataframe['rmi'] < 20)
            &
            (dataframe['cci'] <= -150)
            &
            (dataframe['srsi_fk'] < 20)
            # Maybe comment mfi and cmf to make more trades
            &
            (dataframe['mfi'] < 25)
            &
            (dataframe['cmf'] <= -0.1)
        ).astype('int')

        dataframe['is_break'] = (
            (dataframe['bb_width'] > 0.025)
            &
            (dataframe['bb_bottom_cross'].rolling(10).sum() > 1)
            &
            (dataframe['close'] < 0.99 * dataframe['bb_lowerband'])
        ).astype('int')

        dataframe['buy_signal'] = (
            (dataframe['is_dip'] > 0)
            &
            (dataframe['is_break'] > 0)

        ).astype('int')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['buy_signal'] > 0),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bb_middleband']))
                |
                (qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband']))

            ), 'sell'] = 1

        return dataframe