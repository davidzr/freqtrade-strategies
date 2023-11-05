# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds
import pandas_ta as pta

# Buy hyperspace params:
buy_params = {
     "base_nb_candles_buy": 11,
     "ewo_high": 2.337,
     "ewo_low": -15.87,
     "low_offset": 0.979,
     "rsi_buy": 55,
 }

# # Sell hyperspace params:
# sell_params = {
#     "base_nb_candles_sell": 26,
#     "high_offset": 0.994,
# }

# Buy hyperspace params:
#buy_params = {
#    "base_nb_candles_buy": 11,
#    "ewo_high": 2.337,
#    "ewo_low": -15.87,
#    "low_offset": 0.979,
#    "rsi_buy": 55,
#}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 17,
    "high_offset": 0.997,
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

    # Momentum Indicators
    # ------------------------------------

    # ADX
    dataframe['adx'] = ta.ADX(dataframe)

    # Plus Directional Indicator / Movement
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)

    # Minus Directional Indicator / Movement
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)

    # Aroon, Aroon Oscillator
    aroon = ta.AROON(dataframe)
    dataframe['aroonup'] = aroon['aroonup']
    dataframe['aroondown'] = aroon['aroondown']
    dataframe['aroonosc'] = ta.AROONOSC(dataframe)

    # Awesome Oscillator
    dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

    # Keltner Channel
    keltner = qtpylib.keltner_channel(dataframe)
    dataframe["kc_upperband"] = keltner["upper"]
    dataframe["kc_lowerband"] = keltner["lower"]
    dataframe["kc_middleband"] = keltner["mid"]
    dataframe["kc_percent"] = (
        (dataframe["close"] - dataframe["kc_lowerband"]) /
        (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
    )
    dataframe["kc_width"] = (
        (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
    )

    # Ultimate Oscillator
    dataframe['uo'] = ta.ULTOSC(dataframe)

    # Commodity Channel Index: values [Oversold:-100, Overbought:100]
    dataframe['cci'] = ta.CCI(dataframe)

    # RSI
    dataframe['rsi'] = ta.RSI(dataframe)

    # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (dataframe['rsi'] - 50)
    dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
    dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

    # # Stochastic Slow
    stoch = ta.STOCH(dataframe)
    dataframe['slowd'] = stoch['slowd']
    dataframe['slowk'] = stoch['slowk']

    # Stochastic Fast
    stoch_fast = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch_fast['fastd']
    dataframe['fastk'] = stoch_fast['fastk']

    # # Stochastic RSI
    # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
    # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
    stoch_rsi = ta.STOCHRSI(dataframe)
    dataframe['fastd_rsi'] = stoch_rsi['fastd']
    dataframe['fastk_rsi'] = stoch_rsi['fastk']

    # MACD
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']

    # MFI
    dataframe['mfi'] = ta.MFI(dataframe)

    # # ROC
    dataframe['roc'] = ta.ROC(dataframe)

    # Overlap Studies
    # ------------------------------------

    # Bollinger Bands
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe['bb_lowerband'] = bollinger['lower']
    dataframe['bb_middleband'] = bollinger['mid']
    dataframe['bb_upperband'] = bollinger['upper']
    dataframe["bb_percent"] = (
        (dataframe["close"] - dataframe["bb_lowerband"]) /
        (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
    )
    dataframe["bb_width"] = (
        (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
    )

    # Parabolic SAR
    dataframe['sar'] = ta.SAR(dataframe)

    # TEMA - Triple Exponential Moving Average
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

    # Cycle Indicator
    # ------------------------------------
    # Hilbert Transform Indicator - SineWave
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']

    # Pattern Recognition - Bullish candlestick patterns
    # ------------------------------------
    # Hammer: values [0, 100]
    dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
    # Inverted Hammer: values [0, 100]
    dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
    # Dragonfly Doji: values [0, 100]
    dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
    # Piercing Line: values [0, 100]
    dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)  # values [0, 100]
    # Morningstar: values [0, 100]
    dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # values [0, 100]
    # Three White Soldiers: values [0, 100]
    dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)  # values [0, 100]

    # Pattern Recognition - Bearish candlestick patterns
    # ------------------------------------
    # Hanging Man: values [0, 100]
    dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
    # Shooting Star: values [0, 100]
    dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
    # Gravestone Doji: values [0, 100]
    dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
    # Dark Cloud Cover: values [0, 100]
    dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
    # Evening Doji Star: values [0, 100]
    dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
    # Evening Star: values [0, 100]
    dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

    # Pattern Recognition - Bullish/Bearish candlestick patterns
    # ------------------------------------
    # Three Line Strike: values [0, -100, 100]
    dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
    # Spinning Top: values [0, -100, 100]
    dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)  # values [0, -100, 100]
    # Engulfing: values [0, -100, 100]
    dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # values [0, -100, 100]
    # Harami: values [0, -100, 100]
    dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)  # values [0, -100, 100]
    # Three Outside Up/Down: values [0, -100, 100]
    dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)  # values [0, -100, 100]
    # Three Inside Up/Down: values [0, -100, 100]
    dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]

    # # Chart type
    # # ------------------------------------
    # # Heikin Ashi Strategy
    heikinashi = qtpylib.heikinashi(dataframe)
    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']

    return dataframe


class ElliotV531(IStrategy):
    INTERFACE_VERSION = 2

    # # ROI table:
    # minimal_roi = {
    #     "0": 0.215,
    #     "40": 0.132,
    #     "87": 0.086,
    #     "201": 0.03
    # }

    # # Stoploss:
    # stoploss = -0.189
    # # ROI table:
    # minimal_roi = {
    #     "0": 0.178,
    #     "11": 0.085,
    #     "31": 0.029,
    #     "126": 0
    # }

    # # Stoploss:
    # stoploss = -0.339

    # # Trailing stop:
    # trailing_stop = True
    # trailing_stop_positive = 0.177
    # trailing_stop_positive_offset = 0.243
    # trailing_only_offset_is_reached = False
    # # ROI table:
    # minimal_roi = {
    #     "0": 0.207,
    #     "17": 0.049,
    #     "28": 0.026,
    #     "109": 0
    # }
    # ROI table:
    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.132,
        "87": 0.086,
        "201": 0.03
    }

    # Stoploss:
    stoploss = -0.189

    # # Trailing stop:
    # trailing_stop = True
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.1
    # trailing_only_offset_is_reached = False

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.007
    trailing_stop_positive_offset = 0.0157
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = False
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '30m'

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'blue'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = True

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.05
    }

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0):
            return 0.99
        else:
            trade_time_50 = current_time - timedelta(minutes=241)

            # Trade open more then 241 minutes. For this strategy it's means -> loss
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
       
        return 1

#        stoploss = self.stoploss
#        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
#        last_candle = dataframe.iloc[-1].squeeze()
#        if last_candle is None:
#            return stoploss
#
#        trade_date = timeframe_to_prev_date(
#            self.timeframe, trade.open_date_utc - timedelta(seconds=timeframe_to_seconds(self.timeframe)))
#        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
#        if trade_candle.empty:
#            return stoploss
#
#        trade_candle = trade_candle.squeeze()
#
#        dur_minutes = (current_time - trade.open_date_utc).seconds // 60
#
#        slippage_ratio = trade.open_rate / trade_candle['close'] - 1
#        slippage_ratio = slippage_ratio if slippage_ratio > 0 else 0
#        current_profit_comp = current_profit + slippage_ratio

#        if current_profit_comp >= self.trailing_stop_positive_offset:
#            return self.trailing_stop_positive
#
#        for x in self.minimal_roi:
#            dur = int(x)
#            roi = self.minimal_roi[x]
#            if dur_minutes >= dur and current_profit_comp >= roi:
#                return 0.001

#        return stoploss

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['hma_50'] = pta.hma(dataframe['close'], 50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
