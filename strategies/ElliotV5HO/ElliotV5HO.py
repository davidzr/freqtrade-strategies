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

# Buy hyperspace params:
# buy_params = {
#     "base_nb_candles_buy": 9,
#     "ewo_high": 3.439,
#     "ewo_low": -9.168,
#     "low_offset": 0.977,
#     "rsi_buy": 70,
# }

# # Sell hyperspace params:
# sell_params = {
#     "base_nb_candles_sell": 26,
#     "high_offset": 0.994,
# }

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 11,
    "ewo_high": 2.337,
    "ewo_low": -15.87,
    "low_offset": 0.979,
    "rsi_buy": 55,
}

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


class ElliotV5HO(IStrategy):
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
    minimal_roi = {
        "0": 0.051,
        "20": 0.032,
        "45": 0.016,
        "70": 0
    }

    # Stoploss:
    stoploss = -0.302

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
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
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
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 79

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()

        # Positive market, big trailing, we are catching a big fish.
        if current_profit > 0.30:
            return -0.15

        if current_profit > 0.2:
            return stoploss_from_open(+0.10, current_profit)

        # 5% profit guaranteed.
        if current_profit > 0.1:
            return stoploss_from_open(+0.05, current_profit)

        return 1

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
