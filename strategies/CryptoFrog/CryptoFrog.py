from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from cachetools import TTLCache

## I hope you know what these are already
from pandas import DataFrame
import numpy as np

## Indicator libs
import talib.abstract as ta
from finta import TA as fta

## FT stuffs
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open, IntParameter, DecimalParameter, CategoricalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from skopt.space import Dimension

class CryptoFrog(IStrategy):

    # ROI table - this strat REALLY benefits from roi and trailing hyperopt:
    minimal_roi = {
        "0": 0.213,
        "39": 0.103,
        "96": 0.037,
        "166": 0
    }

    # Stoploss:
    stoploss = -0.085

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.047
    trailing_only_offset_is_reached = False
    
    use_custom_stoploss = True
    custom_stop = {
        # Linear Decay Parameters
        'decay-time': 166,       # minutes to reach end, I find it works well to match this to the final ROI value - default 1080
        'decay-delay': 0,         # minutes to wait before decay starts
        'decay-start': -0.085, # -0.32118, # -0.07163,     # starting value: should be the same or smaller than initial stoploss - default -0.30
        'decay-end': -0.02,       # ending value - default -0.03
        # Profit and TA  
        'cur-min-diff': 0.03,     # diff between current and minimum profit to move stoploss up to min profit point
        'cur-threshold': -0.02,   # how far negative should current profit be before we consider moving it up based on cur/min or roc
        'roc-bail': -0.03,        # value for roc to use for dynamic bailout
        'rmi-trend': 50,          # rmi-slow value to pause stoploss decay
        'bail-how': 'immediate',  # set the stoploss to the atr offset below current price, or immediate
        # Positive Trailing
        'pos-trail': True,        # enable trailing once positive  
        'pos-threshold': 0.005,   # trail after how far positive
        'pos-trail-dist': 0.015   # how far behind to place the trail
    }

    # Dynamic ROI
    droi_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default='any', space='sell', optimize=True)
    droi_pullback = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    droi_pullback_amount = DecimalParameter(0.005, 0.02, default=0.005, space='sell')
    droi_pullback_respect_table = CategoricalParameter([True, False], default=False, space='sell', optimize=True)    
    
    # Custom Stoploss
    cstp_threshold = DecimalParameter(-0.05, 0, default=-0.03, space='sell')
    cstp_bail_how = CategoricalParameter(['roc', 'time', 'any'], default='roc', space='sell', optimize=True)
    cstp_bail_roc = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell')
    cstp_bail_time = IntParameter(720, 1440, default=720, space='sell')    
    
    stoploss = custom_stop['decay-start']    

    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes
        
    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    use_dynamic_roi = True    
    
    timeframe = '5m'
    informative_timeframe = '1h'

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    plot_config = {
        'main_plot': {
            'Smooth_HA_H': {'color': 'orange'},
            'Smooth_HA_L': {'color': 'yellow'},
        },
        'subplots': {
            "StochRSI": {
                'srsi_k': {'color': 'blue'},
                'srsi_d': {'color': 'red'},
            },
            "MFI": {
                'mfi': {'color': 'green'},
            },
            "BBEXP": {
                'bbw_expansion': {'color': 'orange'},
            },
            "FAST": {
                'fastd': {'color': 'red'},
                'fastk': {'color': 'blue'},
            },
            "SQZMI": {
                'sqzmi': {'color': 'lightgreen'},
            },
            "VFI": {
                'vfi': {'color': 'lightblue'},
            },
            "DMI": {
                'dmi_plus': {'color': 'orange'},
                'dmi_minus': {'color': 'yellow'},
            },
            "EMACO": {
                'emac_1h': {'color': 'red'},
                'emao_1h': {'color': 'blue'},
            },
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        #pairs.append("BTC/USDT")
        #pairs.append("ETH/USDT")
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    ## smoothed Heiken Ashi
    def HA(self, dataframe, smoothing=None):
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
    
    def hansen_HA(self, informative_df, period=6):
        dataframe = informative_df.copy()
        
        dataframe['hhclose']=(dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['hhopen']= ((dataframe['open'].shift(2) + dataframe['close'].shift(2))/ 2) #it is not the same as real heikin ashi since I found that this is better.
        dataframe['hhhigh']=dataframe[['open','close','high']].max(axis=1)
        dataframe['hhlow']=dataframe[['open','close','low']].min(axis=1)

        dataframe['emac'] = ta.SMA(dataframe['hhclose'], timeperiod=period) #to smooth out the data and thus less noise.
        dataframe['emao'] = ta.SMA(dataframe['hhopen'], timeperiod=period)
        
        return {'emac': dataframe['emac'], 'emao': dataframe['emao']}
    
    ## detect BB width expansion to indicate possible volatility
    def bbw_expansion(self, bbw_rolling, mult=1.1):
        bbw = list(bbw_rolling)

        m = 0.0
        for i in range(len(bbw)-1):
            if bbw[i] > m:
                m = bbw[i]

        if (bbw[-1] > (m * mult)):
            return 1
        return 0

    ## do_indicator style a la Obelisk strategies
    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Stoch fast - mainly due to 5m timeframes
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']        
        
        #StochRSI for double checking things
        period = 14
        smoothD = 3
        SmoothK = 3
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        stochrsi  = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # Bollinger Bands because obviously
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        # SAR Parabol - probably don't need this
        dataframe['sar'] = ta.SAR(dataframe)
        
        ## confirm wideboi variance signal with bbw expansion
        dataframe["bb_width"] = ((dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"])
        dataframe['bbw_expansion'] = dataframe['bb_width'].rolling(window=4).apply(self.bbw_expansion)

        # confirm entry and exit on smoothed HA
        dataframe = self.HA(dataframe, 4)

        # thanks to Hansen_Khornelius for this idea that I apply to the 1hr informative
        # https://github.com/hansen1015/freqtrade_strategy
        hansencalc = self.hansen_HA(dataframe, 6)
        dataframe['emac'] = hansencalc['emac']
        dataframe['emao'] = hansencalc['emao']
        
        # money flow index (MFI) for in/outflow of money, like RSI adjusted for vol
        dataframe['mfi'] = fta.MFI(dataframe)
        
        ## sqzmi to detect quiet periods
        dataframe['sqzmi'] = fta.SQZMI(dataframe) #, MA=hansencalc['emac'])
        
        # Volume Flow Indicator (MFI) for volume based on the direction of price movement
        dataframe['vfi'] = fta.VFI(dataframe, period=14)
        
        dmi = fta.DMI(dataframe, period=14)
        dataframe['dmi_plus'] = dmi['DI+']
        dataframe['dmi_minus'] = dmi['DI-']
        dataframe['adx'] = fta.ADX(dataframe, period=14)
        
        ## for stoploss - all from Solipsis4
        ## simple ATR and ROC for stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)        
        dataframe['rmi'] = RMI(dataframe, length=24, mom=5)
        ssldown, sslup = SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')        
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3,1,0) 
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['close'].shift(),1,0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3,1,0)        
        
        return dataframe

    ## stolen from Obelisk's Ichi strat code and backtest blog post, and Solipsis4
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
        
        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 30), "Backtest this strategy in 5m or 1m timeframe."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)
            
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
            
            skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume', 'emac', 'emao']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)

        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['sroc'] = dataframe[['date', 'sroc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['ssl-dir'] = dataframe[['date', 'ssl-dir']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['candle-up-trend'] = dataframe[['date', 'candle-up-trend']].copy().set_index('date')            
            
        return dataframe

    ## cryptofrog signals
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    ## close ALWAYS needs to be lower than the heiken low at 5m
                    (dataframe['close'] < dataframe['Smooth_HA_L'])
                    &
                    ## Hansen's HA EMA at informative timeframe
                    (dataframe['emac_1h'] < dataframe['emao_1h'])
                )
                &
                (
                    (
                        ## potential uptick incoming so buy
                        (dataframe['bbw_expansion'] == 1) & (dataframe['sqzmi'] == False)
                        &
                        (
                            (dataframe['mfi'] < 20)
                            |
                            (dataframe['dmi_minus'] > 30)
                        )
                    )
                    |
                    (
                        # this tries to find extra buys in undersold regions
                        (dataframe['close'] < dataframe['sar'])
                        &
                        ((dataframe['srsi_d'] >= dataframe['srsi_k']) & (dataframe['srsi_d'] < 30))
                        &
                        ((dataframe['fastd'] > dataframe['fastk']) & (dataframe['fastd'] < 23))
                        &
                        (dataframe['mfi'] < 30)
                    )
                    |
                    (
                        # find smaller temporary dips in sideways
                        (
                            ((dataframe['dmi_minus'] > 30) & qtpylib.crossed_above(dataframe['dmi_minus'], dataframe['dmi_plus']))
                            &
                            (dataframe['close'] < dataframe['bb_lowerband'])
                        )
                        |
                        (
                            ## if nothing else is making a buy signal
                            ## just throw in any old SQZMI shit based fastd
                            ## this needs work!
                            (dataframe['sqzmi'] == True)
                            &
                            ((dataframe['fastd'] > dataframe['fastk']) & (dataframe['fastd'] < 20))
                        )
                    )
                    ## volume sanity checks
                    &
                    (dataframe['vfi'] < 0.0)                    
                    &
                    (dataframe['volume'] > 0)                    
                )
            ),
            'buy'] = 1

        return dataframe
    
    ## more going on here
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    ## close ALWAYS needs to be higher than the heiken high at 5m
                    (dataframe['close'] > dataframe['Smooth_HA_H'])
                    &
                    ## Hansen's HA EMA at informative timeframe
                    (dataframe['emac_1h'] > dataframe['emao_1h'])
                )
                &
                (
                    ## try to find oversold regions with a corresponding BB expansion
                    (
                        (dataframe['bbw_expansion'] == 1)
                        &
                        (
                            (dataframe['mfi'] > 80)
                            |
                            (dataframe['dmi_plus'] > 30)
                        )
                    )
                    ## volume sanity checks
                    &
                    (dataframe['vfi'] > 0.0)
                    &
                    (dataframe['volume'] > 0)                    
                )
            ),
            'sell'] = 1
        return dataframe

    """
    Everything from here completely stolen from the godly work of @werkkrew
    
    Custom Stoploss 
    """ 
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

    """
    Freqtrade ROI Overload for dynamic ROI functionality
    """
    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        minimal_roi = self.minimal_roi
        _, table_roi = self.min_roi_reached_entry(trade_dur)

        # see if we have the data we need to do this, otherwise fall back to the standard table
        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]
                candle_trend = dataframe['candle-up-trend'].iat[-1]
                ssl_dir = dataframe['ssl-dir'].iat[-1]
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            else:
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']
                candle_trend = self.custom_trade_info[trade.pair]['candle-up-trend'].loc[current_time]['candle-up-trend']
                ssl_dir = self.custom_trade_info[trade.pair]['ssl-dir'].loc[current_time]['ssl-dir']

            min_roi = table_roi
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            pullback_value = (max_profit - self.droi_pullback_amount.value)
            in_trend = False

            if self.droi_trend_type.value == 'rmi' or self.droi_trend_type.value == 'any':
                if rmi_trend == 1:
                    in_trend = True
            if self.droi_trend_type.value == 'ssl' or self.droi_trend_type.value == 'any':
                if ssl_dir == 'up':
                    in_trend = True
            if self.droi_trend_type.value == 'candle' or self.droi_trend_type.value == 'any':
                if candle_trend == 1:
                    in_trend = True

            # Force the ROI value high if in trend
            if (in_trend == True):
                min_roi = 100
                # If pullback is enabled, allow to sell if a pullback from peak has happened regardless of trend
                if self.droi_pullback.value == True and (current_profit < pullback_value):
                    if self.droi_pullback_respect_table.value == True:
                        min_roi = table_roi
                    else:
                        min_roi = current_profit / 2

        else:
            min_roi = table_roi

        return trade_dur, min_roi

    # Change here to allow loading of the dynamic_roi settings
    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:  
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.use_dynamic_roi:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi    
    
    # Get the current price from the exchange (or local cache)
    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            # Check if cache has been invalidated
            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate    
    
    """
    Stripped down version from Schism, meant only to update the price data a bit
    more frequently than the default instead of getting all sorts of trade information
    """
    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = False

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):
            
            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

        return trade_data

    # nested hyperopt class
    class HyperOpt:

        # defining as dummy, so that no error is thrown about missing
        # sell indicator space when hyperopting for all spaces
        @staticmethod
        def indicator_space() -> List[Dimension]:
            return []

## goddamnit

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

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

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc
