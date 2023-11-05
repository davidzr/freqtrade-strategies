import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import (IStrategy, timeframe_to_prev_date, merge_informative_pair, stoploss_from_open, 
                                IntParameter, DecimalParameter, CategoricalParameter, RealParameter)
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Dict, List
from skopt.space import Dimension


###########################################################################################################
##                CombinedBinHAndClucV5 by iterativ                                                      ##
##                                                                                                       ##
##    Fretrade https://github.com/freqtrade/freqtrade                                                    ##
##    The authors of the original CombinedBinHAndCluc https://github.com/freqtrade/freqtrade-strategies  ##
##    V5 by iterativ.                                                                                    ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 20 to 40 pairs. Volume pairlist works well.                                         ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
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


class CombinedBinHAndClucV5Hyperoptable(IStrategy):
    minimal_roi = {
        "0": 0.018
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    buy_bin_bbdelta_close =  RealParameter(0.004, 0.15, default=0.008, space='buy', optimize=True, load=True)
    buy_bin_closedelta_close = RealParameter(0.01, 0.03, default=0.0175, space='buy', optimize=True, load=True)
    buy_bin_tail_bbdelta = RealParameter(0.1, 0.5, default=0.25, space='buy', optimize=True, load=True)
    buy_cluc_close_bblowerband = RealParameter(0.5, 1.5, default=0.985, space='buy', optimize=True, load=True)
    buy_cluc_volume = IntParameter(15, 30, default=20, space='buy', optimize=True, load=True)
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
                        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (current_time - timedelta(minutes=300) > trade.open_date_utc):
            return 0.01
        return 0.99
    # for hyperopt
                                                                                               
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (  # strategy BinHV45
                (dataframe['lower'].shift() > 0 ) &
                (dataframe['bbdelta'] > (dataframe['close'] * self.buy_bin_closedelta_close.value)) &
                (dataframe['closedelta'] > (dataframe['close'] * self.buy_bin_closedelta_close.value)) &
                (dataframe['tail'] < (dataframe['bbdelta'] * self.buy_bin_tail_bbdelta.value)) &
                (dataframe['close'] < (dataframe['lower'].shift())) &
                (dataframe['close'] <= (dataframe['close'].shift())) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (  # strategy ClucMay72018
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < self.buy_cluc_close_bblowerband.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * self.buy_cluc_volume.value)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ( # Improves the profit slightly.
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe
        
    # nested hyperopt class    
    class HyperOpt:

        # defining as dummy, so that no error is thrown about missing
        # sell indicator space when hyperopting for all spaces
        @staticmethod
        def sell_indicator_space() -> List[Dimension]:
            return []
