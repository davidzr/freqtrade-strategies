# --- Do not remove these libs --------------------------------------------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import logging
import pandas as pd
import numpy as np
from pandas import DataFrame, Series, DatetimeIndex, merge
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from functools import reduce


###########################################################################################################
##                CombinedBinHAndClucV6 by iterativ                                                      ##
##                hyperoptable additions and refactoring by nightshift2k                                 ##
##                                                                                                       ##
##                source & discussion -> freqtrade discord:                                              ##
##                https://discord.gg/ZdsMQ3tE                                                            ##
##                                                                                                       ##
##    Freqtrade https://github.com/freqtrade/freqtrade                                                   ##
##    The authors of the original CombinedBinHAndCluc https://github.com/freqtrade/freqtrade-strategies  ##
##    V6 by iterativ.                                                                                    ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 20 to 40 pairs. Volume pairlist works well.                                         ##
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


# -------------------------------------------------------------------------------------------------
# --- logger for parameter merging output, only remove if you remove it further down too! ---------
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------------------------------------

class CombinedBinHAndClucV6H(IStrategy):

    minimal_roi = {
        "0": 0.0181
    }

    max_open_trades = 5

    # Using custom stoploss
    stoploss = -0.99
    use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    timeframe = '5m'
    informative_timeframe = '1h'

    # derived from EMA 200
    startup_candle_count: int = 200

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False


    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = True

    # hyperspace default buy params 
    buy_params = {
        'buy_bin_bbdelta_close': 0.031,
        'buy_bin_closedelta_close': 0.018,
        'buy_bin_tail_bbdelta': 0.233,
        'buy_bin_guard': True,

        'buy_cluc_close_bblowerband': 0.993,
        'buy_cluc_volume': 21,
        'buy_cluc_guard': True,

        'buy_long_rsi_diff': 43.276, 

        'buy_bin_enable': True,
        'buy_cluc_enable': True,
        'buy_long_enable': True,

        'buy_minimum_conditions': 1
    }

    # hyperspace default sell params
    sell_params = {
        'sell_roi_override_rsi_threshold': 50, # to disable holding with high RSI set to 100
        'cstp_loss_threshold': 0,
        'cstp_bail_time': 5
    }

    # if you want to see which buy conditions were met
    # or if there is an trade exit override due to high RSI set to True
    # logger will output the buy and trade exit conditions
    cust_log_verbose = False


    ###########################################################################################################
    ## IMPORTANT NOTICE REGARDING HYPEROPT:                                                                  ##
    ##   Use the optimize dict below to enable optimization (set to True) of one or more                     ##
    ##   hyperspace parameters.                                                                              ##
    ##                                                                                                       ##
    ##   Please be aware: Enabling ALL at once will very likely produce very low results!                    ## 
    ##   So please enable max. 1-2 parameters at once, you'll need to play around                            ##
    ##   what makes more or less sense. At the end it always depends on a lot more                           ##
    ##   parameters (stake, max trades, hyperopt-loss, etc.)                                                 ##
    ###########################################################################################################
  
    # use this switches to enable the optimization of hyperspace params
    cust_optimize = {
        'buy_bin_bbdelta_close': False,
        'buy_bin_closedelta_close': False,
        'buy_bin_tail_bbdelta': False,
        'buy_bin_guard': False,
        'buy_cluc_close_bblowerband': False,
        'buy_cluc_volume': False,
        'buy_cluc_guard': False,
        'buy_long_rsi_diff': False,
        'buy_bin_enable': False,
        'buy_cluc_enable': False,
        'buy_long_enable': False,
        'buy_minimum_conditions': False,
        'sell_roi_override_rsi_threshold': False,
        'cstp_bail_time': False,
        'cstp_loss_threshold': False
    }

    # bin buy parameters
    buy_bin_bbdelta_close =  DecimalParameter(0.0, 0.05, default=0.031, space='buy', optimize=cust_optimize['buy_bin_bbdelta_close'], load=True)
    buy_bin_closedelta_close = DecimalParameter(0.0, 0.03, default=0.018, decimals=4, space='buy', optimize=cust_optimize['buy_bin_closedelta_close'], load=True)
    buy_bin_tail_bbdelta = DecimalParameter(0.0, 1.0, default=0.233, decimals=3, space='buy', optimize=cust_optimize['buy_bin_tail_bbdelta'], load=True)
    buy_bin_guard = CategoricalParameter([True, False], default=True, space='buy', optimize=cust_optimize['buy_bin_guard'], load=True)

    # cluc buy parameters
    buy_cluc_close_bblowerband = DecimalParameter(0.0, 1.5, default=0.993, decimals=3, space='buy', optimize=cust_optimize['buy_cluc_close_bblowerband'], load=True)
    buy_cluc_volume = IntParameter(10, 40, default=21, space='buy', optimize=cust_optimize['buy_cluc_volume'], load=True)
    buy_cluc_guard = CategoricalParameter([True, False], default=True, space='buy', optimize=cust_optimize['buy_cluc_guard'], load=True)

    # log buy parameters
    buy_long_rsi_diff = DecimalParameter(40, 45, default=43.276, decimals=3, space='buy', optimize=cust_optimize['buy_long_rsi_diff'], load=True)

    # enable bin, cluc or long 
    buy_bin_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=cust_optimize['buy_bin_enable'], load=True)
    buy_cluc_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=cust_optimize['buy_cluc_enable'], load=True)
    buy_long_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=cust_optimize['buy_long_enable'], load=True)
    
    # minimum conditions to match in buy
    buy_minimum_conditions = IntParameter(1, 2, default=1, space='buy', optimize=cust_optimize['buy_minimum_conditions'], load=True)

    # if RSI above threshold, override ROI
    sell_roi_override_rsi_threshold = IntParameter(40, 70, default=50, space='sell', optimize=cust_optimize['sell_roi_override_rsi_threshold'], load=True)

    # custom stoploss, if trade open > x hours and loss > threshold then sell    
    cstp_bail_time = IntParameter(1, 36, default=5, space='sell', optimize=cust_optimize['cstp_bail_time'])
    cstp_loss_threshold = DecimalParameter(-0.25, 0, default=0, decimals=2, space='sell', optimize=cust_optimize['cstp_loss_threshold'])


    """
    Informative Pairs
    """
    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs


    """
    Informative Timeframe Indicators
    """
    def get_informative_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        # EMA
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # SSL Channels
        ssldown, sslup = SSLChannels_ATR(dataframe, 20)
        dataframe['ssl-up'] = sslup
        dataframe['ssl-down'] = ssldown
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')

        return dataframe


    """
    Main Timeframe Indicators
    """
    def get_main_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
        
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # EMA
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # drop NAN in hyperopt to fix "'<' not supported between instances of 'str' and 'int' error
        if self.config['runmode'].value == 'hyperopt':
            dataframe = dataframe.dropna()

        return dataframe


    """
    Populate Informative and Main Timeframe Indicators
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
   
        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in a timeframe of 5m or less."

        assert self.dp, "DataProvider is required for multiple timeframes."

        informative = self.get_informative_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
        dataframe = self.get_main_indicators(dataframe, metadata)

        return dataframe


    """
    Buy Signal
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
      
        # reset additional dataframe rows
        dataframe.loc[:, 'buy_cond_bin'] = False
        dataframe.loc[:, 'buy_cond_cluc'] = False
        dataframe.loc[:, 'buy_cond_long'] = False
        dataframe.loc[:, 'conditions_count'] = 0
        

        # strategy BinHV45 + guards by @iterativ
        dataframe.loc[
            (
                # @iterativ adition (guard)
                (
                    (
                        (dataframe['close'] > dataframe['ema_200_1h']) &
                        (dataframe['ema_50'] > dataframe['ema_200']) &
                        (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                        (self.buy_bin_guard.value == True)
                    ) |
                    (self.buy_bin_guard.value == False)
                ) &

                # strategy BinHV45
                dataframe['lower'].shift().gt(0) &

                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bin_bbdelta_close.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_bin_closedelta_close.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_bin_tail_bbdelta.value) &

                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (self.buy_bin_enable.value == True)
            ),
            'buy_cond_bin'
        ] = 1

        # strategy ClucMay72018 + guards by @iterativ
        dataframe.loc[
            (   
                # @iterativ addition (guard)
                (
                    (
                        (dataframe['close'] > dataframe['ema_200']) &
                        (dataframe['close'] > dataframe['ema_200_1h']) &
                        (self.buy_cluc_guard.value == True)
                    ) |
                    (self.buy_cluc_guard.value == False)
                ) &

                # strategy ClucMay72018
                (dataframe['close'] < dataframe['ema_50']) &
                (dataframe['close'] < self.buy_cluc_close_bblowerband.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * self.buy_cluc_volume.value)) &
                (self.buy_cluc_enable.value == True)
            ),
            'buy_cond_cluc'
        ] = 1

        # long strategy by @iterativ
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['sma_5']) &
                (dataframe['ssl-dir_1h'] == 'up') &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_long_rsi_diff.value) &
                (self.buy_long_enable.value == True)
            ),
            'buy_cond_long'
        ] = 1

        # count the amount of conditions met
        dataframe.loc[:, 'conditions_count'] = dataframe['buy_cond_bin'].astype(int) + dataframe['buy_cond_cluc'].astype(int) + dataframe['buy_cond_long'].astype(int)

        # append the minimum amount of conditions to be met
        conditions.append(dataframe['conditions_count'] >= self.buy_minimum_conditions.value)
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'
            ] = 1

        # verbose logging enable only for verbose information or troubleshooting
        if self.cust_log_verbose == True:
            for index, row in dataframe.iterrows():
                if row['buy'] == 1:               
                    buy_cond_details = f"count={int(row['conditions_count'])}/bin={int(row['buy_cond_bin'])}/cluc={int(row['buy_cond_cluc'])}/long={int(row['buy_cond_long'])}"
                    logger.info(f"{metadata['pair']} - candle: {row['date']} - buy condition - details: {buy_cond_details}")

        return dataframe


    """
    Sell Signal
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'
            ] = 1

        return dataframe


    """
    Custom Stop Loss
    """
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        
        if (current_profit < self.cstp_loss_threshold.value) & (current_time - timedelta(hours=int(self.cstp_bail_time.value)) > trade.open_date_utc):
            return 0.01

        return self.stoploss


    """
    Trade Exit Confirmation
    """
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if self.cust_log_verbose == True:
            logger.info(f"{pair} - candle: {last_candle['date']} - exit trade {sell_reason} with profit {trade.calc_profit_ratio(rate)}")

        # failsafe for user triggered forced sells > always have highest prio!
        if sell_reason == 'force_sell':
            return True

        # Prevent ROI trigger, if there is more potential, in order to maximize profit
        if last_candle is not None and ((sell_reason == 'roi') ):
            rsi = 0
            if 'rsi' in last_candle.index:
                rsi = last_candle['rsi']
           
            if ( (rsi >= self.sell_roi_override_rsi_threshold.value) ):
                if self.cust_log_verbose == True:
                    logger.info(f"{pair} - candle: {last_candle['date']} - not exiting trade with current profit {trade.calc_profit_ratio(rate)}, rsi = {rsi} which is > than {self.sell_roi_override_rsi_threshold.value}")
                return False
            else:
                return True

        return True


# --- custom indicators ---------------------------------------------------------------------------

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']
