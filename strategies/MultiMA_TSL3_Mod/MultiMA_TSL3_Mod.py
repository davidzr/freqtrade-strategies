import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, RealParameter,BooleanParameter, timeframe_to_minutes)
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema, VIDYA

###########################################################################################################
##    MultiMA_TSL, modded by stash86, based on SMAOffsetProtectOptV1 (modded by Perkmeister)             ##
##    Based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
##    Thanks to                                                                                          ##
##    - Perkmeister, for their snippets for the sell signals and decaying EMA sell                       ##
##    - ChangeToTower, for the PMax idea                                                                 ##
##    - JimmyNixx, for their snippet to limit close value from the peak (that I modify into 5m tf check) ##
##    - froggleston, for the Heikinashi check snippet from Cryptofrog                                    ##
##    - Uzirox, for their pump detection code                                                            ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################

# I hope you do enough testing before proceeding, either backtesting and/or dry run.
# Any profits and losses are all your responsibility

class MultiMA_TSL3_Mod(IStrategy):
    INTERFACE_VERSION = 2

    DATESTAMP = 0
    SELLMA = 1
    SELL_TRIGGER=2

    # Buy hyperspace params:
    buy_params = {
        "buy_rsi_fast_max": 98,
        "buy_rsi_fast_min": 36,
        "buy_rsi_max": 79,
        "buy_rsi_min": 24,
        "ewo_high": 0.546,
        "ewo_high2": 8.497,
        "ewo_low": -14.239,
        "ewo_low2": -15.614,
        "fast_ewo": 12,
        "pmax_pct_max": 83.754,
        "pmax_pct_min": 20.09,
        "slow_ewo": 150,
        "volume_pct_max": 8.721,
        "volume_pct_min": 0.247,
        "buy_condition_ema_enable": True,  # value loaded from strategy
        "close_pct_max": 0.06785,
        "close_pct_min": 0.01121,

    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_ema_sell": 65,
        "base_nb_candles_ema_sell2": 49,
        "high_offset_sell_ema": 1.074,
    }

    # Protection hyperspace params:
    protection_params = {
        "cooldown_lookback": 39,
        "low_profit_lookback": 29,
        "low_profit_min_req": -0.03,
        "low_profit_stop_duration": 52,
    }



    # ROI table:
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.15
    use_custom_stoploss = True

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.018

    # Buy hyperspace params:
    """optimize_buy_ema = False # Not used
    base_nb_candles_buy_ema = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_ema)
    low_offset_ema = DecimalParameter(0.9, 1.1, default=0.958, space='buy', optimize=optimize_buy_ema)
    base_nb_candles_buy_ema2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_ema)
    low_offset_ema2 = DecimalParameter(0.9, 1.1, default=0.958, space='buy', optimize=optimize_buy_ema)

    optimize_buy_trima = False # Not used
    base_nb_candles_buy_trima = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_trima)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_trima)
    base_nb_candles_buy_trima2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_trima)
    low_offset_trima2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_trima)
    
    optimize_buy_zema = False # Not used
    base_nb_candles_buy_zema = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_zema)
    low_offset_zema = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_zema)
    base_nb_candles_buy_zema2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_zema)
    low_offset_zema2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_zema)

    optimize_buy_hma = False # Not used
    base_nb_candles_buy_hma = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_hma)
    low_offset_hma = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_hma)
    base_nb_candles_buy_hma2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_hma)
    low_offset_hma2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_hma)"""

    buy_condition_enable_optimize = False # Not used
    buy_condition_ema_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    """buy_condition_trima_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    buy_condition_zema_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    buy_condition_hma_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)"""

    ewo_check_optimize = True
    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, space='buy', optimize=ewo_check_optimize)
    ewo_high = DecimalParameter(0.0, 12.0, default=6.0, space='buy', optimize=ewo_check_optimize)
    ewo_low2 = DecimalParameter(-20.0, -8.0, default=-20.0, space='buy', optimize=ewo_check_optimize)
    ewo_high2 = DecimalParameter(2.0, 12.0, default=6.0, space='buy', optimize=ewo_check_optimize)
    fast_ewo = IntParameter(10, 50, default=50, space='buy', optimize=True)
    slow_ewo = IntParameter(100, 200, default=200, space='buy', optimize=True)
    
    pct_optimize = True
    pmax_pct_min = DecimalParameter(1.00, 100.00, default=1, space='buy', optimize=pct_optimize)
    pmax_pct_max = DecimalParameter(1.00, 100.00, default=1, space='buy', optimize=pct_optimize)
    volume_pct_min = DecimalParameter(0.01, 20, default=0.01, space='buy', optimize=pct_optimize)
    volume_pct_max = DecimalParameter(0.01, 20, default=0.01, space='buy', optimize=pct_optimize)

    high_precision_pct_optimize = False # Optimise this setting individually
    close_pct_min = RealParameter(0.0001, 0.1, default=0.01, space='buy', optimize=high_precision_pct_optimize)
    close_pct_max = RealParameter(0.0001, 0.1, default=0.01, space='buy', optimize=high_precision_pct_optimize)
    
    buy_rsi_optimize = True
    buy_rsi_min = IntParameter(0, 100, default=1, space='buy', optimize=buy_rsi_optimize)
    buy_rsi_max = IntParameter(0, 100, default=100, space='buy', optimize=buy_rsi_optimize)
    buy_rsi_fast_min = IntParameter(0, 100, default=1, space='buy', optimize=buy_rsi_optimize)
    buy_rsi_fast_max = IntParameter(0, 100, default=100, space='buy', optimize=buy_rsi_optimize)
    
    # Sell hyperspace params:

    optimize_sell_ema = True
    base_nb_candles_ema_sell = IntParameter(5, 80, default=20, space='sell', optimize=True)
    high_offset_sell_ema = DecimalParameter(0.99, 1.1, default=1.012, space='sell', optimize=True)
    base_nb_candles_ema_sell2 = IntParameter(5, 80, default=20, space='sell', optimize=True)

    # Protection hyperspace params:

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)

    low_profit_optimize = True
    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2, optimize=low_profit_optimize)

    
    
    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.low_profit_lookback.value,
            "trade_limit": 1,
            "stop_duration": int(self.low_profit_stop_duration.value),
            "required_profit": self.low_profit_min_req.value
        })

        return prot

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # storage dict for custom info
    custom_info = { }

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
       
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if(len(dataframe) < 1):
            return False
        last_candle = dataframe.iloc[-1]

        if(self.custom_info[pair][self.DATESTAMP] != last_candle['date']):
            # new candle, update EMA and check sell

            # smoothing coefficients
            sell_ema = self.custom_info[pair][self.SELLMA]
            if(sell_ema == 0):
                sell_ema = last_candle['ema_sell'] 
            emaLength = 32
            alpha = 2 /(1 + emaLength) 

            # update sell_ema
            sell_ema = (alpha * last_candle['close']) + ((1 - alpha) * sell_ema)
            self.custom_info[pair][self.SELLMA] = sell_ema
            self.custom_info[pair][self.DATESTAMP] = last_candle['date']

            if((last_candle['close'] > (sell_ema * self.high_offset_sell_ema.value)) & (last_candle['buy_copy'] == 0)):
                if self.config['runmode'].value in ('live', 'dry_run'):
                    self.custom_info[pair][self.SELL_TRIGGER] = 1
                    return False

                buy_tag = 'empty'

                if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
                    buy_tag = trade.buy_tag
                else:
                    trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
                    buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
                    if not buy_signal.empty:
                        buy_signal_candle = buy_signal.iloc[-1]
                        buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'

                return f'New Sell Signal ({buy_tag})'
        
        return False

    #credit to Perkmeister for this custom stoploss to help the strategy ride a green candle when the sell signal triggered
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if(self.custom_info[pair][self.SELL_TRIGGER] == 1):
            if self.config['runmode'].value in ('live', 'dry_run'):
                sl_new = 0.001

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.01

        return sl_new

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if(len(dataframe) < 1):
            return False
        last_candle = dataframe.iloc[-1].squeeze()
        if ((rate > last_candle['close'])) : 
            return False

        self.custom_info[pair][self.DATESTAMP] = last_candle['date']
        self.custom_info[pair][self.SELLMA] = last_candle['ema_sell']

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        self.custom_info[pair][self.SELL_TRIGGER] = 0
        return True

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # EWO
        #dataframe['ema_delta'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema2.value)) - ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value # EWO delta? Not used anyway
        
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]

        dataframe['ha_up'] = (heikinashi['close'] > heikinashi['open']).astype('int')
        dataframe['ha_down'] = (heikinashi['open'] > heikinashi['close']).astype('int')
        

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        dataframe = HA(dataframe, 4)

        if self.config['runmode'].value in ('live', 'dry_run'):
            # Exchange downtime protection
            dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)
        else:
            dataframe['live_data_ok'] = True

        # Check if the entry already exists
        if not metadata["pair"] in self.custom_info:
            # Create empty entry for this pair {datestamp, sellma, sell_trigger}
            self.custom_info[metadata["pair"]] = ['', 0, 0]

        dataframe['24hr_high'] = (dataframe['high'].rolling(window=288, min_periods= 288).max())
        dataframe['smooth_high'] =ta.EMA(dataframe['24hr_high'], timeperiod=2)
        dataframe['high_rising'] = (dataframe['smooth_high'] > dataframe['smooth_high'].shift()).astype('int')
        dataframe['high_falling'] = (dataframe['smooth_high'] < dataframe['smooth_high'].shift()).astype('int')
        
        dataframe['24hr_low'] = (dataframe['low'].rolling(window=288, min_periods= 288).min())
        dataframe['smooth_low'] =ta.EMA(dataframe['24hr_low'], timeperiod=2)
        dataframe['low_rising'] = (dataframe['smooth_low'] > dataframe['smooth_low'].shift()).astype('int')
        dataframe['low_falling'] = (dataframe['smooth_low'] < dataframe['smooth_low'].shift()).astype('int')
        
        dataframe['24hr_delta'] = (dataframe['24hr_high'] - dataframe['24hr_low'])
        dataframe['smooth_delta'] =ta.EMA(dataframe['24hr_delta'], timeperiod=2)
        dataframe['delta_rising'] = (dataframe['smooth_delta'] > dataframe['smooth_delta'].shift()).astype('int')

        dataframe['pmax_high_delta'] = (dataframe['24hr_high'] - dataframe['pmax_thresh'])
        dataframe['smooth_pmax_high'] =ta.EMA(dataframe['pmax_high_delta'], timeperiod=2)
        dataframe['pmax_low_delta'] = (dataframe['pmax_thresh'] - dataframe['24hr_low'])
        dataframe['smooth_pmax_low'] =ta.EMA(dataframe['pmax_low_delta'], timeperiod=2)

        dataframe['pmax_pct'] = (dataframe['pmax_thresh'] - dataframe['24hr_low']) / (dataframe['24hr_high'] - dataframe['24hr_low']) * 100
        dataframe['pmax_pct_rising'] = (dataframe['pmax_pct'] > dataframe['pmax_pct'].shift()).astype('int')

        dataframe['smooth_volume'] =ta.EMA(dataframe['volume'], timeperiod=2)
        dataframe['smooth_volume_slow'] =ta.EMA(dataframe['volume'], timeperiod=12)
        dataframe['volume_pct'] =(dataframe['volume']).pct_change()
        dataframe['smooth_volume_pct'] =ta.EMA(dataframe['volume_pct'], timeperiod=2)
        dataframe['volume_pct_rising'] = (dataframe['volume_pct'] > dataframe['volume_pct'].shift()).astype('int')
        dataframe['smooth_volume_pct_rising'] =ta.EMA(dataframe['volume_pct_rising'], timeperiod=2)

        dataframe['close_pct'] =(dataframe['close']).pct_change()

    
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        """dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value
        dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema2.value)) *self.low_offset_ema2.value"""
        dataframe['ema_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_ema_sell.value))   
        
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        if (self.buy_condition_ema_enable.value):

            buy_offset_ema = (
                ( 
                    #(dataframe['pm'] <= dataframe['pmax_thresh'])
                    #&
                    #(dataframe['ha_up'].rolling(self.ha_rolling_up.value).sum() == self.ha_rolling_up.value)
                    #&
                    #(qtpylib.crossed_above(dataframe['HA_Close'].shift(self.ha_rolling_up.value -1 ), dataframe['HA_Open'].shift(self.ha_rolling_up.value + 1)))
                    #&
                    #(dataframe['ha_down'].shift(self.ha_rolling_up.value).rolling(self.ha_rolling_down.value).sum() == self.ha_rolling_down.value)
                    #&
                    (qtpylib.crossed_below(dataframe['sar'], dataframe['pmax_thresh']))
                    &
                    (dataframe['pmax_thresh'] > dataframe['pm'])
                    &
                    (dataframe['pmax_thresh'] > dataframe['sar'])
                    #&
                    #(dataframe['high_rising'] == 1)
                )
            )
            dataframe.loc[buy_offset_ema, 'buy_tag'] += 'ema '
            conditions.append(buy_offset_ema)

        """if (self.buy_condition_zema_enable.value):
            dataframe['zema_offset_buy'] = zema(dataframe, int(self.base_nb_candles_buy_zema.value)) *self.low_offset_zema.value
            dataframe['zema_offset_buy2'] = zema(dataframe, int(self.base_nb_candles_buy_zema2.value)) *self.low_offset_zema2.value
            buy_offset_zema = (
                (
                    (dataframe['close'] < dataframe['zema_offset_buy'])
                    &
                    (dataframe['pm'] <= dataframe['pmax_thresh'])
                )
                |
                (
                    (dataframe['close'] < dataframe['zema_offset_buy2'])
                    &
                    (dataframe['pm'] > dataframe['pmax_thresh'])
                )
            )
            dataframe.loc[buy_offset_zema, 'buy_tag'] += 'zema '
            conditions.append(buy_offset_zema)

        if (self.buy_condition_hma_enable.value):
            dataframe['hma_offset_buy'] = qtpylib.hull_moving_average(dataframe['close'], window=int(self.base_nb_candles_buy_hma.value)) *self.low_offset_hma.value
            dataframe['hma_offset_buy2'] = qtpylib.hull_moving_average(dataframe['close'], window=int(self.base_nb_candles_buy_hma2.value)) *self.low_offset_hma2.value
            buy_offset_hma = (
                (
                    (
                        (dataframe['close'] < dataframe['hma_offset_buy'])
                        &
                        (dataframe['pm'] <= dataframe['pmax_thresh'])
                        &
                        (dataframe['rsi'] < 35)
    
                    )
                    |
                    (
                        (dataframe['close'] < dataframe['hma_offset_buy2'])
                        &
                        (dataframe['pm'] > dataframe['pmax_thresh'])
                        &
                        (dataframe['rsi'] < 30)
                    )
                )
                &
                (dataframe['rsi_fast'] < 30)
                
            )
            dataframe.loc[buy_offset_hma, 'buy_tag'] += 'hma '
            conditions.append(buy_offset_hma)"""

        add_check = (
            (dataframe['live_data_ok'])
            &
            (dataframe['pmax_pct'] > self.pmax_pct_min.value)
            &
            (dataframe['volume_pct'] > self.volume_pct_min.value)   
            &
            (dataframe['close_pct'] > self.close_pct_min.value)
            &
            (dataframe['rsi'] > self.buy_rsi_min.value)
            &
            (dataframe['rsi_fast'] > self.buy_rsi_fast_min.value)
            &
            (dataframe['pmax_pct'] < self.pmax_pct_max.value)
            &
            (dataframe['volume_pct'] < self.volume_pct_max.value)
            &
            (dataframe['close_pct'] < self.close_pct_max.value)
            &
            (dataframe['rsi'] < self.buy_rsi_max.value)
            &
            (dataframe['rsi_fast'] < self.buy_rsi_fast_max.value)
            &
            (dataframe['ewo'] > self.ewo_high.value)
            &
            
            #(dataframe['open'] < dataframe['ema_offset_buy'])
            #&
            #(dataframe['buy_low_rolling'].shift().rolling(self.buy_smooth_ha_rolling.value).sum() == self.buy_low_rolling.value)
            #&
            #(dataframe['delta_rising'].rolling(5).sum() == self.buy_smooth_ha_rolling.value)
            #&
            #(dataframe['close'] > (dataframe['ema_sell'] * self.high_offset_sell_ema.value))
            #&
            #(dataframe['close'].rolling(288).max() < (dataframe['close'] * 1.10 ))
            #&
            #(dataframe['Smooth_HA_O'].shift(1) < dataframe['Smooth_HA_H'].shift(1))
            #&
            #(dataframe['rsi_fast'] > self.buy_rsi_fast.value)
            #&
            #(dataframe['rsi_84'] > 60)
            #&
            #(dataframe['rsi_112'] > 60)
            #&
            #(dataframe['ewo'] > self.ewo_high.value)
            #&
            #(
            #    (
            #        (dataframe['close'] > dataframe['pmax_thresh'])
            #        &
            #        (dataframe['pm'] > dataframe['pmax_thresh'])
            #        &
            #        (
            #            (dataframe['ewo'] < self.ewo_low.value)
            #            |
            #            (
            #                (dataframe['ewo'] > self.ewo_high.value)
            #                &
            #                (dataframe['rsi'] < self.rsi_buy.value)
            #            )
            #        )
            #    )
            #    |
            #    (
            #        (dataframe['close'] > dataframe['pmax_thresh'])
            #        &
            #        (dataframe['pm'] > dataframe['pmax_thresh'])
            #        &
            #        (
            #            (dataframe['ewo'] < self.ewo_low2.value)
            #            |
            #            (
            #                (dataframe['ewo'] > self.ewo_high2.value)
            #                &
            #                (dataframe['rsi'] < self.rsi_buy2.value)
            #            )
            #        )
            #    )
            #)
            #&
            (dataframe['volume'] > 0)
        )
        
        if conditions:
            dataframe.loc[
                (add_check & reduce(lambda x, y: x | y, conditions)),
                ['buy_copy','buy']
            ]=(1,1)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0

        return dataframe

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']
