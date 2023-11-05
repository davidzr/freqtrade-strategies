# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter, stoploss_from_open
from datetime import datetime, timedelta
from functools import reduce

# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

# Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

class BBRSITV(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "ewo_high": 4.86,
        "for_ma_length": 22,
        "for_sigma": 1.74,
    }

    # Sell hyperspace params:
    sell_params = {
        "for_ma_length_sell": 65,
        "for_sigma_sell": 1.895,
        "rsi_high": 72,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.25  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True
    startup_candle_count = 30

    protections = [
        #   {
        #       "method": "StoplossGuard",
        #       "lookback_period_candles": 12,
        #       "trade_limit": 1,
        #       "stop_duration_candles": 6,
        #       "only_per_pair": True
        #   },
        #   {
        #       "method": "StoplossGuard",
        #       "lookback_period_candles": 12,
        #       "trade_limit": 2,
        #       "stop_duration_candles": 6,
        #       "only_per_pair": False
        #   },
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]

    ewo_high = DecimalParameter(0, 7.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    for_sigma = DecimalParameter(0, 10.0, default=buy_params['for_sigma'], space='buy', optimize=True)
    for_sigma_sell = DecimalParameter(0, 10.0, default=sell_params['for_sigma_sell'], space='sell', optimize=True)
    rsi_high = IntParameter(60, 100, default=sell_params['rsi_high'], space='sell', optimize=True)
    for_ma_length = IntParameter(5, 80, default=buy_params['for_ma_length'], space='buy', optimize=True)
    for_ma_length_sell = IntParameter(5, 80, default=sell_params['for_ma_length_sell'], space='sell', optimize=True)

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # //@version=3
        # study(" RSI + BB (EMA) + Dispersion (2.0)", overlay=false)
        #
        # // Инициализация параметров
        # src = input(title="Source", type=source, defval=close) // Устанавливаем тип цены для расчетов
        src = 'close'
        # for_rsi = input(title="RSI_period", type=integer, defval=14) // Период для RSI
        for_rsi = 14
        # for_ma = input(title="Basis_BB", type=integer, defval=20) // Период для MA внутри BB
        # for_ma = 20
        # for_mult = input(title="Stdev", type=integer, defval=2, minval=1, maxval=5) // Число стандартных отклонений для BB
        for_mult = 2
        # for_sigma = input(title="Dispersion", type=float, defval=0.1, minval=0.01, maxval=1) // Дисперсия вокруг MA
        for_sigma = 0.1
        #
        # // Условия работы скрипта
        # current_rsi = rsi(src, for_rsi) // Текущее положение индикатора RSI
        dataframe['rsi'] = ta.RSI(dataframe[src], for_rsi)
        dataframe['rsi_4'] = ta.RSI(dataframe[src], 4)
        if self.config['runmode'].value == 'hyperopt':
            for for_ma in range(5, 81):
                # basis = ema(current_rsi, for_ma)
                dataframe[f'basis_{for_ma}'] = ta.EMA(dataframe['rsi'], for_ma)
                # dev = for_mult * stdev(current_rsi, for_ma)
                dataframe[f'dev_{for_ma}'] = ta.STDDEV(dataframe['rsi'], for_ma)
                # upper = basis + dev
                #dataframe[f'upper_{for_ma}'] = (dataframe[f'basis_{for_ma}'] + (dataframe[f'dev_{for_ma}'] * for_mult))
                # lower = basis - dev
                #dataframe[f'lower_{for_ma}'] = dataframe[f'basis_{for_ma}'] - (dataframe[f'dev_{for_ma}'] * for_mult)
                # disp_up = basis + ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                # dataframe[f'disp_up_{for_ma}'] = dataframe[f'basis_{for_ma}'] + ((dataframe[f'upper_{for_ma}'] - dataframe[f'lower_{for_ma}']) * for_sigma)
                # disp_down = basis - ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                # dataframe[f'disp_down_{for_ma}'] = dataframe[f'basis_{for_ma}'] - ((dataframe[f'upper_{for_ma}'] - dataframe[f'lower_{for_ma}']) * for_sigma)
                # color_rsi = current_rsi >= disp_up ? lime : current_rsi <= disp_down ? red : #ffea00 // Текущий цвет RSI, в зависимости от его местоположения внутри BB
        else:
            dataframe[f'basis_{self.for_ma_length.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'basis_{self.for_ma_length_sell.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length_sell.value)
            # dev = for_mult * stdev(current_rsi, for_ma)
            dataframe[f'dev_{self.for_ma_length.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'dev_{self.for_ma_length_sell.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length_sell.value)

        #
        # // Дополнительные линии и заливка для областей для RSI
        # h1 = hline(70, color=#d4d4d4, linestyle=dotted, linewidth=1)
        h1 = 70
        # h2 = hline(30, color=#d4d4d4, linestyle=dotted, linewidth=1)
        h2 = 30
        # fill (h1, h2, transp=95)
        #
        # // Алерты и условия срабатывания
        # rsi_Green = crossover(current_rsi, disp_up)
        # rsi_Red = crossunder(current_rsi, disp_down)

        # alertcondition(condition=rsi_Green,
        #      title="RSI cross Above Dispersion Area",
        #      message="The RSI line closing crossed above the Dispersion area.")
        #
        # alertcondition(condition=rsi_Red,
        #      title="RSI cross Under Dispersion Area",
        #      message="The RSI line closing crossed below the Dispersion area")
        #
        # // Результаты и покраска
        # plot(basis, color=black)
        # plot(upper, color=#00fff0, linewidth=2)
        # plot(lower, color=#00fff0, linewidth=2)
        # s1 = plot(disp_up, color=white)
        # s2 = plot(disp_down, color=white)
        # fill(s1, s2, color=white, transp=80)
        # plot(current_rsi, color=color_rsi, linewidth=2)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # upper = basis + dev
                # lower = basis - dev
                # disp_up = basis + ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                # disp_up = basis + ((basis + dev * for_mult) - (basis - dev * for_mult)) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                # disp_up = basis + (basis + dev * for_mult - basis + dev * for_mult)) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                # disp_up = basis + (2 * dev * for_sigma * for_mult) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value))) &
                (dataframe['EWO'] >  self.ewo_high.value) &
                (dataframe['volume'] > 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['rsi'] > self.rsi_high.value) |
                    # upper = basis + dev
                    # lower = basis - dev
                    # disp_down = basis - ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                    # disp_down = basis - ((2* dev * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                    (dataframe['rsi'] > dataframe[f'basis_{self.for_ma_length_sell.value}'] + ((dataframe[f'dev_{self.for_ma_length_sell.value}'] * self.for_sigma_sell.value)))
                ) &
                (dataframe['volume'] > 0)

            ),
            'sell'] = 1
        return dataframe

class BBRSITV4(BBRSITV):
    minimal_roi = {
        "0": 0.07
    }
    ignore_roi_if_buy_signal = True
    startup_candle_count = 400

    stoploss = -0.3  # value loaded from strategy

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value)))
                &
                (
                    (
                        (dataframe['EWO'] > self.ewo_high.value)
                        &
                        (dataframe['EWO'] < 10)
                    )
                    |
                    (
                        (dataframe['EWO'] >= 10)
                        &
                        (dataframe['rsi'] < 40)
                    )
                )
                &
                (dataframe['rsi_4'] < 25)
                &
                (dataframe['volume'] > 0)
                # &
                # (dataframe["roc_bbwidth_max"] < 70)
            ),
            'buy'] = 1

        return dataframe

class BBRSITV1(BBRSITV):
    """
    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
============================================================================= STRATEGY SUMMARY =============================================================================
|              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |              Drawdown |
|-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+-----------------------|
|         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT  79.84% |
| SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT  61.08% |
|               BBRSITV |    309 |           1.10 |         340.17 |          3869.800 |         128.99 |        2:53:00 |   223     0    86  72.2 |  261.984 USDT  25.84% |
============================================================================================================================================================================
    """
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "ewo_high": 4.964,
        "for_ma_length": 12,
        "for_sigma": 2.313,
    }

    # Sell hyperspace params:
    sell_params = {
        "for_ma_length_sell": 78,
        "for_sigma_sell": 1.67,
        "rsi_high": 60,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.25  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

class BBRSITV2(BBRSITV):
    """
    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
============================================================================= STRATEGY SUMMARY =============================================================================
|              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |              Drawdown |
|-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+-----------------------|
|         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT  79.84% |
| SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT  61.08% |
|               BBRSITV |    486 |           1.11 |         537.58 |          7689.862 |         256.33 |        5:01:00 |   287     0   199  59.1 | 1279.461 USDT  75.45% |
============================================================================================================================================================================
    """
    # Buy hyperspace params:
    buy_params = {
        "ewo_high": 4.85,
        "for_ma_length": 11,
        "for_sigma": 2.066,
    }

    # Sell hyperspace params:
    sell_params = {
        "for_ma_length_sell": 61,
        "for_sigma_sell": 1.612,
        "rsi_high": 87,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.25  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy


class BBRSITV3(BBRSITV):
    """

    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
    ============================================================================== STRATEGY SUMMARY =============================================================================
    |              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |               Drawdown |
    |-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+------------------------|
    |         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT   79.84% |
    | SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT   61.08% |
    |               BBRSITV |    627 |           1.14 |         715.85 |         12998.605 |         433.29 |        5:35:00 |   374     0   253  59.6 | 2294.408 USDT  100.60% |
    ============================================================================================================================================================================="""
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "ewo_high": 4.86,
        "for_ma_length": 22,
        "for_sigma": 1.74,
    }

    # Sell hyperspace params:
    sell_params = {
        "for_ma_length_sell": 65,
        "for_sigma_sell": 1.895,
        "rsi_high": 72,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.25  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.078
    trailing_stop_positive_offset = 0.095
    trailing_only_offset_is_reached = False
    
class BBRSITV5(BBRSITV):
    minimal_roi = {
        "0": 0.04
    }
    ignore_roi_if_buy_signal = True
    startup_candle_count = 400
    use_custom_stoploss = True

    stoploss = -0.3  # value loaded from strategy
    sell_params = {
        ##
        "pHSL": -0.178,
        "pPF_1": 0.01,
        "pPF_2": 0.048,
        "pSL_1": 0.009,
        "pSL_2": 0.043,
    }
    
    is_optimize_trailing = True
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value)))
                &
                (
                    (
                        (dataframe['EWO'] > self.ewo_high.value)
                        &
                        (dataframe['EWO'] < 10)
                    )
                    |
                    (
                        (dataframe['EWO'] >= 10)
                        &
                        (dataframe['rsi'] < 40)
                    )
                )
                &
                (dataframe['rsi_4'] < 25)
                &
                (dataframe['volume'] > 0)
                # &
                # (dataframe["roc_bbwidth_max"] < 70)
            ),
            'buy'] = 1

        return dataframe
