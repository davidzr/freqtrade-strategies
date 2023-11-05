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


#Divergence variables
rangeUpper = 60
rangeLower = 5

# Buy hyperspace params:
buy_params = {
      "antipump_threshold": 0.257,
      "base_nb_candles_buy": 14,
      "ewo_high": 2.327,
      "ewo_high_2": -2.327,
      "ewo_low": -20.988,
      "low_offset": 0.975,
      "low_offset_2": 0.955,
      "rsi_buy": 40,

      #SMAOffsetProtectOptV1
      "base_nb_candles_buy2": 16,
      "ewo_high2": 5.638,
      "ewo_low2": -19.993,
      "low_offset2": 0.978,
      "rsi_buy2": 61,
}

# Sell hyperspace params:
sell_params = {
      "base_nb_candles_sell": 24,
      "high_offset": 0.991,
      "high_offset_2": 0.997,
      #SMAOffsetProtectOptV1
      "base_nb_candles_sell2": 49,
      "high_offset2": 1.006, "cstp_bail_how": "roc",
      #####
      "cstp_bail_roc": -0.032,
      "cstp_bail_time": 1108,
      "cstp_bb_trailing_input": "bb_lowerband_neutral_inf",
      "cstp_threshold": -0.036,
      "cstp_trailing_max_stoploss": 0.054,
      "cstp_trailing_only_offset_is_reached": 0.06,
      "cstp_trailing_stop_profit_devider": 2,
      "droi_pullback": True,
      "droi_pullback_amount": 0.03,
      "droi_pullback_respect_table": False,
      "droi_trend_type": "any",
}




class custom(IStrategy):
    INTERFACE_VERSION = 2

    # Modified ROI - 20210620
    # ROI table:
    minimal_roi = {
        "0": 0.028,
        "10": 0.018,
        "30": 0.010,
        "40": 0.005
    }


    custom_trade_info = {}

    # Stoploss:
    stoploss = -0.11

    antipump_threshold = DecimalParameter(0, 0.4, default=0.25, space='buy', optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    #SMAOffsetProtectOptV1
    low_offset2 = DecimalParameter(0.9, 0.99, default=buy_params['low_offset2'], space='buy', optimize=True)
    base_nb_candles_buy2 = IntParameter(5, 80, default=buy_params['base_nb_candles_buy2'], space='buy', optimize=True)
    base_nb_candles_sell2 = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    high_offset2 = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.00, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    #SMAOffsetProtectOptV1
    ewo_low2 = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low2'], space='buy', optimize=True)
    ewo_high2 = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high2'], space='buy', optimize=True)
    rsi_buy2 = IntParameter(30, 70, default=buy_params['rsi_buy2'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = False
    #trailing_stop_positive = 0.005
    #trailing_stop_positive_offset = 0.03
    #trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = False
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count: int = 100

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }
    # Strategy Specific Variable Storage
    custom_trade_info = {}


    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        len = 14
        src = dataframe['close']
        lbL = 10 #5
        dataframe['osc'] = ta.RSI(src, len)
        dataframe['osc'] = dataframe['osc'].fillna(0)

        # plFound = na(pivotlow(osc, lbL, lbR)) ? false : true
        dataframe['min'] = dataframe['osc'].rolling(lbL).min()
        dataframe['prevMin'] = np.where(dataframe['min'] > dataframe['min'].shift(), dataframe['min'].shift(), dataframe['min'])
        dataframe.loc[
            (dataframe['osc'] == dataframe['prevMin'])
        , 'plFound'] = 1
        dataframe['plFound'] = dataframe['plFound'].fillna(0)

        # phFound = na(pivothigh(osc, lbL, lbR)) ? false : true
        dataframe['max'] = dataframe['osc'].rolling(lbL).max()
        dataframe['prevMax'] = np.where(dataframe['max'] < dataframe['max'].shift(), dataframe['max'].shift(), dataframe['max'])
        dataframe.loc[
            (dataframe['osc'] == dataframe['prevMax'])
        , 'phFound'] = 1
        dataframe['phFound'] = dataframe['phFound'].fillna(0)


        #------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        # oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscHL'] = 1

        # Price: Lower Low
        # priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe, 'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] < dataframe['valuewhen_plFound_low'])
            , 'priceLL'] = 1
        #bullCond = plotBull and priceLL and oscHL and plFound
        dataframe.loc[
            (
                (dataframe['priceLL'] == 1) &
                (dataframe['oscHL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'bullCond'] = 1

        # //------------------------------------------------------------------------------
        # // Hidden Bullish
        # // Osc: Lower Low
        #
        # oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscLL'] = 1
        #
        # // Price: Higher Low
        #
        # priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe,'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] > dataframe['valuewhen_plFound_low'])
            , 'priceHL'] = 1
        # hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
        dataframe.loc[
            (
                (dataframe['priceHL'] == 1) &
                (dataframe['oscLL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'hiddenBullCond'] = 1

        # //------------------------------------------------------------------------------
        # // Regular Bearish
        # // Osc: Lower High
        #
        # oscLH = osc[lbR] < valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscLH'] = 1
        #
        # // Price: Higher High
        #
        # priceHH = high[lbR] > valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] > dataframe['valuewhen_phFound_high'])
            , 'priceHH'] = 1
        #
        # bearCond = plotBear and priceHH and oscLH and phFound
        dataframe.loc[
            (
                (dataframe['priceHH'] == 1) &
                (dataframe['oscLH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'bearCond'] = 1

        # //------------------------------------------------------------------------------
        # // Hidden Bearish
        # // Osc: Higher High
        #
        # oscHH = osc[lbR] > valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscHH'] = 1
        #
        # // Price: Lower High
        #
        # priceLH = high[lbR] < valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] < dataframe['valuewhen_phFound_high'])
            , 'priceLH'] = 1
        #
        # hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound
        dataframe.loc[
            (
                (dataframe['priceLH'] == 1) &
                (dataframe['oscHH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'hiddenBearCond'] = 1

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy2.range:
            dataframe[f'ma_buy2_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell2.range:
            dataframe[f'ma_sell2_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['hma_9'] = qtpylib.hull_moving_average(dataframe['close'], window=9)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        #pump stregth
        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['rmi'] = RMI(dataframe, length=24, mom=5)

        dataframe['roc'] = dataframe['close'].pct_change(12).rolling(12).max() * 100

        # Base pair informative timeframe indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        # Get the "average day range" between the 1d high and 1d low to set up guards
        informative['1d-high'] = informative['close'].rolling(24).max()
        informative['1d-low'] = informative['close'].rolling(24).min()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dont_buy_conditions = []

        dont_buy_conditions.append(
            (dataframe['pump_strength'] > self.antipump_threshold.value) &
            (dataframe['bearCond'] < 1) &
            (dataframe['hiddenBearCond'] < 1)
        )
        #""

        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] <35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            )
        )

        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] <35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
                (dataframe['rsi']<25)
            )
        )

        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] < 35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            )
        )
        #""

        #SMAOffsetProtectOptV1
        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['close'] < (dataframe[f'ma_buy2_{self.base_nb_candles_buy2.value}'] * self.low_offset2.value)) &
                (dataframe['EWO'] > self.ewo_high2.value) &
                (dataframe['rsi'] < self.rsi_buy2.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['close'] < (dataframe[f'ma_buy2_{self.base_nb_candles_buy2.value}'] * self.low_offset2.value)) &
                (dataframe['EWO'] < self.ewo_low2.value) &
                (dataframe['volume'] > 0)
            )
        )


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'sell'] = 0

        """
        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )
        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell2_{self.base_nb_candles_sell2.value}'] * self.high_offset2.value)) &
                (dataframe['volume'] > 0)
            )
        )
        """

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe

    #""
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # only neg
        #if current_profit < -0.1:
        #    if last_candle['hiddenBearCond'] == 1:
        #        return 'Hidden_Bear_div_loss'

        #if current_profit < -0.1:
        #    if last_candle['bearCond'] == 1:
        #        return 'Bear on neg'

        #only pos
        if current_profit > 0:
            if last_candle['hiddenBearCond'] == 1:
                return 'Hidden_Bear_div_profit'

        if current_profit > 0:
            if last_candle['bearCond'] == 1:
                return 'Bear_div_profit'

        # both
        #if current_profit > 0 or current_profit < -0.01:
        #    if last_candle['hiddenBearCond'] == 1:
        #        return 'Hidden_Bear'

        #if current_profit > 0 or current_profit < -0.01:
        #    if last_candle['bearCond'] == 1:
        #        return 'Bear'


        return None
    #""




class SMAoffset_antipump_div(custom):

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dont_buy_conditions = []

        dont_buy_conditions.append(
            (dataframe['pump_strength'] > self.antipump_threshold.value) &
            (dataframe['bearCond'] < 1) &
            (dataframe['hiddenBearCond'] < 1)
        )
        #""
        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] <35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            )
        )
        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] <35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
                (dataframe['rsi']<25)
            )
        )

        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] < 35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            )
        )
        #""

        #SMAOffsetProtectOptV1
        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['close'] < (dataframe[f'ma_buy2_{self.base_nb_candles_buy2.value}'] * self.low_offset2.value)) &
                (dataframe['EWO'] > self.ewo_high2.value) &
                (dataframe['rsi'] < self.rsi_buy2.value) &
                (dataframe['volume'] > 0)
            )
        )

        #""
        conditions.append(
            (
                ( (dataframe['bullCond'] > 0) | (dataframe['hiddenBullCond'] > 0) )&
                (dataframe['close'] < (dataframe[f'ma_buy2_{self.base_nb_candles_buy2.value}'] * self.low_offset2.value)) &
                (dataframe['EWO'] < self.ewo_low2.value) &
                (dataframe['volume'] > 0)
            )
        )
        #""

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe



    """
    Custom Stoploss
    """
    def custom_stoploss_test(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if not self.config['runmode'].value in ('backtest', 'hyperopt'):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if(len(dataframe) >= 1):
                last_candle = dataframe.iloc[-1]
                if((last_candle['sell_copy'] == 1) & (last_candle['buy_copy'] == 0)):
                    sl_new = 0.001

        return sl_new


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            sroc = dataframe['sroc'].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            sroc = self.custom_trade_info[trade.pair]['sroc'].loc[current_time]['sroc']

        if current_profit < self.cstp_threshold.value:
            if self.cstp_bail_how.value == 'roc' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if (sroc/100) <= self.cstp_bail_roc.value:
                    return 0.001
            if self.cstp_bail_how.value == 'time' or self.cstp_bail_how.value == 'any':
                # Dynamic bailout based on time
                if trade_dur > self.cstp_bail_time.value:
                    return 0.001

        return 1

def valuewhen(dataframe, condition, source, occurrence):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    copy = copy.sort_values(by=[condition, 'colFromIndex'], ascending=False).reset_index(drop=True)
    copy['valuewhen'] = np.where(copy[condition] > 0, copy[source].shift(-occurrence), 100)
    copy['valuewhen'] = copy['valuewhen'].fillna(100)
    copy['barrsince'] = copy['colFromIndex'] - copy['colFromIndex'].shift(-occurrence)
    copy.loc[
        (
            (rangeLower <= copy['barrsince']) &
            (copy['barrsince']  <= rangeUpper)
        )
    , "in_range"] = 1
    copy['in_range'] = copy['in_range'].fillna(0)
    copy = copy.sort_values(by=['colFromIndex'], ascending=True).reset_index(drop=True)
    return copy['valuewhen'], copy['in_range']

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def RMI(dataframe, *, length=20, mom=5):
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc
