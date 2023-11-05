# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime, date, timedelta
from technical.indicators import ichimoku, chaikin_money_flow
from freqtrade.exchange import timeframe_to_prev_date
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class epretrace(IStrategy):
    """

    author@: ??

    idea:
        this strategy is based on the link here:

        https://github.com/freqtrade/freqtrade-strategies/issues/95
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        #"14400": 0.001,    # non loosing after 10 days
        #"0": 0.
        "0": 1000
    }

    # Stoploss -disable
    stoploss = -0.999
    #stoploss = -0.05
    use_custom_stoploss = True
    # Trailing stoploss
    #trailing_stop = True
    #trailing_only_offset_is_reached = True
    #trailing_stop_positive = 0.015
    #trailing_stop_positive_offset = 0.02
    # Optimal timeframe for the strategy
    timeframe = '5m'
	
    #buy params
    ep_retracement_window = IntParameter(1, 100, default=50, space='buy')
    #ep_window = IntParameter(1, 100, default=50, space='buy')
    ep_retracement = DecimalParameter(
        0, 1, decimals=2, default=0.95, space='buy')
    ep_retracement2 = DecimalParameter(
        0, 1, decimals=2, default=0.75, space='buy')
    #epma1 = IntParameter(2, 210, default=50, space='buy')
    #epma2 = IntParameter(2, 210, default=200, space='buy')
    #epma1 = 50
    #epma2 = 200
    epcat1 = CategoricalParameter(['open', 'high', 'low', 'close',
                                          #  'ma_fast', 'ma_slow', {...}
                                          ], default='close', space='buy')
    #sell params
    ep_target = DecimalParameter(
        0, 1, decimals=2, default=0.31, space='sell')
    ep_stop = DecimalParameter(
        0, 1, decimals=2, default=0.21, space='sell')
    #ep_retracement_window = 35
    #ep_retracement = 0.90
    #ep_window = 3
    #ep_target = 0.3
    #ep_stop = 0.2
    # buy params
    """epwindow = IntParameter(2, 100, default=7, space='buy')
    #buy_fast_ma_timeframe = IntParameter(2, 100, default=14, space='buy')
    #buy_slow_ma_timeframe = IntParameter(2, 100, default=28, space='buy')
    eptarg = DecimalParameter(
        0, 4, decimals=4, default=2.25446, space='sell')
    epstop = DecimalParameter(
        0, 4, decimals=4, default=0.29497, space='sell')
    epcat1 = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                          #  'ma_fast', 'ma_slow', {...}
                                          ], default='close', space='buy')
    epcat2 = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',
                                          #  'ma_fast', 'ma_slow', {...}
                                          ], default='close', space='buy')
    epma1 = IntParameter(2, 100, default=25, space='buy')
    epma2 = IntParameter(2, 100, default=50, space='buy')
    epma3 = IntParameter(2, 100, default=100, space='buy')
    epma4 = IntParameter(2, 100, default=100, space='buy')
    epma5 = IntParameter(2, 100, default=100, space='buy')
    epma6 = IntParameter(2, 100, default=100, space='sell')
    # sell params
    #sell_mojo_ma_timeframe = IntParameter(2, 100, default=7, space='sell')
    #sell_fast_ma_timeframe = IntParameter(2, 100, default=14, space='sell')
    #sell_slow_ma_timeframe = IntParameter(2, 100, default=28, space='sell')
    #sell_div_max = DecimalParameter(
    #    0, 2, decimals=4, default=1.54593, space='sell')
    #sell_div_min = DecimalParameter(
    #    0, 2, decimals=4, default=2.81436, space='sell')
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Indicators
        #macd = ta.MACD(dataframe)
        #dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)
        #dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        #dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        #dataframe['epma1'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma1)
        #dataframe['epma2'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma2)
        #dataframe['epm3'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma3.value)
        #dataframe['epm4'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma4.value)
        #dataframe['epm5'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma5.value)
        #dataframe['epm6'] = ta.EMA(dataframe,
        #                                 timeperiod=self.epma6.value)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        #result = chaikin_money_flow(testdata_1m_btc, 14)
        #dataframe['cmf'] = chaikin_money_flow(dataframe)
        #dataframe['cci'] = ta.CCI(dataframe)
        #dataframe['eplow'] = dataframe['open'].rolling(2).min()
        #dataframe['eplow'] = dataframe['eplow'].fillna(1000000)
        #dataframe['target'] = dataframe['close'] + (dataframe["close"] - dataframe["eplow"]) * 1.5
        #data_frame[self.value] = pd.rolling_min(data_frame[self.data], self.period)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #Buy if last 5 candles show a strong downtrend (linear regression angle) and close is inferior to the 25 candle linear regression line - 1 * ATR (over 25 candles)
        dataframe.loc[
            (
                #(dataframe['epm1'].rolling(self.epwindow.value).min() > dataframe['epm2'].rolling(self.epwindow.value).max()) &
                #(dataframe['epm2'].rolling(self.epwindow.value).min() > dataframe['epm3'].rolling(self.epwindow.value).max()) &
                #(dataframe[self.epcat1.value].rolling(self.epwindow.value).min() > dataframe['epm4'].rolling(self.epwindow.value).max()) &
                #(dataframe['close'].rolling(6).min() > dataframe['ema50'].rolling(6).max()) &
                #(dataframe['epma1'] > dataframe['epma1'].shift(1).rolling(self.ep_window.value).max()) &
                #(dataframe['epma1'] > dataframe['epma2']) &
                (dataframe[self.epcat1.value] < (dataframe[self.epcat1.value].shift(1).rolling(self.ep_retracement_window.value).max())* self.ep_retracement.value) &
                (dataframe[self.epcat1.value] > (dataframe[self.epcat1.value].shift(1).rolling(self.ep_retracement_window.value).max())* self.ep_retracement2.value) &
                #qtpylib.crossed_above(dataframe[self.epcat2.value], dataframe['epm5']) &
                #qtpylib.crossed_above(dataframe['ema25'], dataframe['ema50']) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sell if RSI is greater than 31 and close is superior to the 25 candle linear regression line
        dataframe.loc[
            (
                (dataframe['close'] > 1000000) &
                #qtpylib.crossed_below(dataframe['close'], dataframe['ema25']) &
                #(dataframe['senkou_a'] > dataframe['senkou_b']) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
                
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        
        if not trade_candle.empty:
            #base_line = trade_candle['eplow'].iloc[0]
            #base_line = trade_candle['ema50'].iloc[0]
            #base_line = base_line + 0.5 
            #open_price = trade_candle['open'].iloc[0]
            #set_stoploss = (open_price / base_line) - 1.01
            #set_stoploss = trade.open_rate - (trade.open_rate) * self.epstop.value
            set_stoploss = trade.open_rate - (trade.open_rate) * self.ep_stop.value
            if current_rate - set_stoploss <= 0.001:
                #print ("current_rate:",current_rate)
                #print ("base_line:",base_line)
                #print ("sub:",current_rate - base_line)
                return -0.000001
              
            #print ("current_rate:",current_rate)
            #print ("base_line:",base_line)
            
        return 1

    def custom_sell(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
                
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        
        if not trade_candle.empty:
            #epstoploss = trade_candle['eplow'].iloc[0]
            #epstoploss = trade_candle['epm6'].iloc[0]
            #open_price = trade_candle['open'].iloc[0]
            #eptarget = trade.open_rate + (trade.open_rate - epstoploss) * 1.5
            #eptarget = trade.open_rate + (trade.open_rate) * self.eptarg.value
            eptarget = trade.open_rate + (trade.open_rate) * self.ep_target.value
            #print ("Target:",eptarget)
            #print ("epstoploss:",epstoploss)
            #print ("Price de abertura:",trade.open_rate)
            if current_rate >= eptarget: #Let prices stabilize before setting
                #print ("Atingiu a meta:",current_rate)
                #print ("Price de abertura:",trade.open_rate)
                #print ("epstoploss:",epstoploss)
                return 'sell_ep2mas'

            
        return 0 
