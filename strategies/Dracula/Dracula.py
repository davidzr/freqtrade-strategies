
from freqtrade.strategy import DecimalParameter, IntParameter

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

from pandas import DataFrame, Series
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as taa
import ta
from functools import reduce
import numpy as np


###########################################################################################################
##                Dracula by 6h057                                                                       ##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 1  open trade, with unlimited stake.              ##
##   A pairlist with  80 pairs. Volume pairlist works well.                                              ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                    ##
##   Ensure that you don't override any variables in you config.json. Especially                         ##
##   the timeframe (should be 5m).                                                                       ##
##                                                                                                       ##
###########################################################################################################
###########################################################################################################
##               DONATIONS                                                                               ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = taa.EMA(df, timeperiod=ema_length)
    ema2 = taa.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def chaikin_money_flow(dataframe, n=20, fillna=False):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

class SupResFinder():
    def isSupport(self, df, i):
        support = df['bb_bbl_i'][i] == 1 and (
            df['bb_bbl_i'][i+1] == 0 or df['close'][i+1] > df['open'][i+1]) and df['close'][i] < df['open'][i]

        return support

    def isResistance(self, df, i):
        resistance = df['bb_bbh_i'][i] == 1 and (
            df['bb_bbh_i'][i+1] == 0 or df['close'][i+1] < df['open'][i+1]) and df['close'][i] > df['open'][i]

        return resistance

    def getSupport(self, df):
        levels = [df['close'][0]]

        for i in range(1, df.shape[0]-2):
            if self.isSupport(df, i):
                o = df['open'][i]
                c = df['close'][i]
                l = c if c < o else o
                levels.append(l)
            else:
                levels.append(levels[-1])
        levels.append(levels[-1])
        levels.append(levels[-1])
        return levels

    def getResistance(self, df):
        levels = [df['open'][0]]

        for i in range(1, df.shape[0]-2):
            if self.isResistance(df, i):
                o = df['open'][i]
                c = df['close'][i]
                l = c if c > o else o
                levels.append(l)
            else:
                levels.append(levels[-1])
        levels.append(levels[-1])
        levels.append(levels[-1])
        return levels


class Dracula(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_bbt": 0.035,
        "ewo_high": 5.638,
        "ewo_low": -19.993,
        "low_offset": 0.978,
        "rsi_buy": 61,
    }

    # Sell hyperspace params:
    sell_params = {
        "high_offset": 1.006
    }
    # ROI table:
    minimal_roi = {
        "0": 10
    }

    info_timeframe = "5m"
    # Stoploss:
    stoploss = -0.2
    min_lost = -0.005

    buy_bbt = DecimalParameter(
        0, 100, decimals=4, default=0.023, space='buy')
    # Buy hypers
    timeframe = '1m'
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    custom_info = {}
    supResFinder = SupResFinder()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['bb_bbh'] = ta.volatility.bollinger_hband(close=dataframe["close"], window=20)
        dataframe['bb_bbl'] = ta.volatility.bollinger_lband(close=dataframe["close"], window=20)

        dataframe['bb_bbh_i'] = dataframe['high'] >= dataframe['bb_bbh']
        dataframe['bb_bbl_i'] = ta.volatility.bollinger_lband_indicator(
            close=dataframe["low"], window=20)
        dataframe['bb_bbt'] = (dataframe['bb_bbh'] - dataframe['bb_bbl']) / dataframe['bb_bbh']

        dataframe['ema'] = taa.EMA(dataframe, timeperiod=150)
        dataframe['resistance'] = self.supResFinder.getResistance(dataframe)
        dataframe['support'] = self.supResFinder.getSupport(dataframe)

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # RSI
        dataframe['rsi'] = taa.RSI(dataframe, timeperiod=14)
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prev = dataframe.shift(1)
        prev1 = dataframe.shift(2)
        lost_protect = (dataframe['ema'] > (dataframe['close'] * 1.07)).rolling(10).sum() == 0

        item_buy_logic = []
        item_buy_logic.append(dataframe['volume'] > 0)
        item_buy_logic.append(dataframe['cmf'] > 0)
        item_buy_logic.append(prev['bb_bbl_i'] == 1)
        item_buy_logic.append(prev['close'] >= prev1['support'])
        item_buy_logic.append(prev['ema'] < prev['close'])
        item_buy_logic.append((dataframe['open'] < dataframe['close']))
        item_buy_logic.append(prev['open'] > prev['close'])
        item_buy_logic.append((dataframe['bb_bbt'] > self.buy_bbt.value))
        item_buy_logic.append(lost_protect)
        dataframe.loc[
            reduce(lambda x, y: x & y, item_buy_logic),
            ['buy', 'buy_tag']] = (1, f'buy_1')

        item_buy_logic = []
        item_buy_logic.append(dataframe['volume'] > 0)
        item_buy_logic.append(dataframe['cmf'] > 0)
        item_buy_logic.append(dataframe['bb_bbl_i'] == 1)
        item_buy_logic.append(dataframe['open'] >= prev1['support'])
        item_buy_logic.append(prev['ema'] < prev['close'])
        item_buy_logic.append((dataframe['open'] < dataframe['close']))
        item_buy_logic.append((dataframe['bb_bbt'] > self.buy_bbt.value))
        item_buy_logic.append(lost_protect)
        dataframe.loc[
            reduce(lambda x, y: x & y, item_buy_logic),
            ['buy', 'buy_tag']] = (1, f'buy_2')

        return dataframe

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        prev_candle = dataframe.iloc[-2].squeeze()
        prev1_candle = dataframe.iloc[-3].squeeze()
        if prev_candle['bb_bbh_i'] == 1 \
                and last_candle['close'] < last_candle['open'] \
                and prev_candle['close'] > prev_candle['open'] \
                and prev_candle['close'] < prev1_candle['resistance'] \
                and last_candle['volume'] > 0:
            return 'sell_signal_1_'+trade.buy_tag
        elif last_candle['bb_bbh_i'] == 1 \
                and last_candle['close'] < last_candle['open'] \
                and last_candle['open'] < prev1_candle['resistance'] \
                and last_candle['volume'] > 0:
            return 'sell_signal_2_'+trade.buy_tag
        elif last_candle['close'] < last_candle['open'] \
            and last_candle['ema'] > (last_candle['close'] * 1.07) \
                and last_candle['volume'] > 0:
            return 'stop_loss_'+trade.buy_tag
        elif (last_candle['close'] < last_candle['open']) and (last_candle['close'] <= (last_candle['bb_bbl'] * 1.002)) and current_profit >= 0:
            return 'take_profit_'+trade.buy_tag
        elif 'sma' in trade.buy_tag and current_profit >= 0.01:
            return 'sma'
        elif 'sma' in trade.buy_tag  and (last_candle['close'] > (last_candle['ema_49'] * self.high_offset.value)):
            return 'stop_loss_sma'
        return None

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0

        return dataframe
