# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class EMA520015_V17(IStrategy):


    minimal_roi = {
        "0": 0.15
    }
    
   # Buy and sell at market price
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }


    stoploss = -0.1
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True
    
    # Optimal timeframe for the strategy
    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        macd = ta.MACD(dataframe,fastperiod=300, slowperiod=650, signalperiod=10)
        dataframe['macd'] = macd['macd']
        dataframe['macdhist'] = macd['macdhist']
        
        
        #Exp Moving Average (200 periods)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema350'] = ta.EMA(dataframe, timeperiod=350)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)



        return dataframe
        
        
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            
                           (dataframe['close'].shift(1) < dataframe['ema20'])
                           & (dataframe['close'] > dataframe['ema20'])


            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                           (dataframe['close'].shift(1) > dataframe['ema20'])
                           & (dataframe['close'] < dataframe['ema20'])
                        
                          
            ),
            'sell'] = 1
        return dataframe
        
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if (current_profit > 0.02) and (last_candle['ema20'] < last_candle['ema200']):
  
            return 'sell2'
