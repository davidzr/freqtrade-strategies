# --- Do not remove these libs ---
from typing import DefaultDict
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import  DecimalParameter, IntParameter


# @Rallipanos mod. Uzirox


def zlema2(dataframe, fast):
    df = dataframe.copy()
    zema1=ta.EMA(df['close'], fast)
    zema2=ta.EMA(zema1, fast)
    d1=zema1-zema2
    df['zlema2']=zema1+d1
    return df['zlema2']


order_types = {
    'buy': 'limit',
    'sell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
    }    

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class NotAnotherSMAOffsetStrategy_uzi3(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.032,
        "87": 0.016,
       # "201": 0
    }


    # Stoploss:
    stoploss = -0.1


    # Protection
    fast_ewo   = 50
    slow_ewo   = 200

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.005
    ignore_roi_if_buy_signal = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 400

    slippage_protection = {
            'retries': 3,
            'max_slippage': -0.02
        }

    buy_signals = {}

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



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if(((dataframe['close'].iloc[-1] - dataframe['close'].iloc[-2])/dataframe['close'].iloc[-2])*100) < -2:
            # si scende pesante         
            
            self.stoploss = -0.3 #stoploss

            # buy params
            self.trailing_stop_positive_offset = 0.03
            self.base_nb_candles_buy  = IntParameter(5, 80, default=14, space='buy', optimize=False)
            self.low_offset           = DecimalParameter(0.9, 0.99, default=0.975, space='buy', optimize=False)
            self.low_offset_2         = DecimalParameter(0.9, 0.99, default=0.955, space='buy', optimize=False)  
            self.ewo_low              = DecimalParameter(-20.0, -8.0,default=-20.988, space='buy', optimize=False)
            self.ewo_high             = DecimalParameter(2.0, 12.0, default=2.327, space='buy', optimize=False)
            self.ewo_high_2           = DecimalParameter(-6.0, 12.0, default=-2.327, space='buy', optimize=False)
            self.rsi_buy              = IntParameter(30, 70, default=69, space='buy', optimize=False)

            # sell params
            self.base_nb_candles_sell = IntParameter(5, 80, default=16, space='sell', optimize=True)
            self.high_offset          = DecimalParameter(0.95, 1.1, default=0.991, space='sell', optimize=True)
            self.high_offset_2        = DecimalParameter(0.99, 1.5, default=0.997, space='sell', optimize=True)     


        else:
            # normale - si sale

            # buy params
            self.base_nb_candles_buy  = IntParameter(5, 80, default=14, space='buy', optimize=False)
            self.low_offset           = DecimalParameter(0.9, 0.99, default=0.986, space='buy', optimize=False)
            self.low_offset_2         = DecimalParameter(0.9, 0.99, default=0.944, space='buy', optimize=False)  
            self.ewo_low              = DecimalParameter(-20.0, -8.0,default=-16.917, space='buy', optimize=False)
            self.ewo_high             = DecimalParameter(2.0, 12.0, default=4.179, space='buy', optimize=False)
            self.ewo_high_2           = DecimalParameter(-6.0, 12.0, default=-2.609, space='buy', optimize=False)
            self.rsi_buy              = IntParameter(30, 70, default=58, space='buy', optimize=False)

            # sell params
            self.base_nb_candles_sell = IntParameter(5, 80, default=16, space='sell', optimize=True)
            self.high_offset          = DecimalParameter(0.95, 1.1, default=1.054, space='sell', optimize=True)
            self.high_offset_2        = DecimalParameter(0.99, 1.5, default=1.018, space='sell', optimize=True)     

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        

        # *MAs
        dataframe['hma_50']  = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod = 100)      
        dataframe['ema_10']  = zlema2(dataframe, 10)
        dataframe['sma_9']   = ta.SMA(dataframe, timeperiod = 9)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

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
        (
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        ),
        ['buy', 'buy_tag']] = (1, 'ewo1')


        dataframe.loc[
        (
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
                (dataframe['rsi']<25)
        ),
        ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
        (
                (dataframe['rsi_fast'] < 35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        ),
        ['buy', 'buy_tag']] = (1, 'ewolow')


        # buy in bull market
        dataframe.loc[
        (  
                (dataframe['ema_10'].rolling(10).mean() > dataframe['ema_100'].rolling(10).mean()) &
                (dataframe['lower'].shift().gt(0)) &
                (dataframe['bbdelta'].gt(dataframe['close'] * 0.031)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.018)) &
                (dataframe['tail'].lt(dataframe['bbdelta'] * 0.233)) &
                (dataframe['close'].lt(dataframe['lower'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['volume'] > 0) 
        )
        |
        (
                (dataframe['ema_10'].rolling(10).mean() > dataframe['ema_100'].rolling(10).mean()) &
                (dataframe['close']  > dataframe['ema_100']) &
                (dataframe['close']  < dataframe['ema_slow']) &
                (dataframe['close']  < 0.993 * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 21)) &
                (dataframe['volume'] > 0)
        ),
        ['buy', 'buy_tag']] = (1, 'bb_bull')    

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50) &
                (dataframe['volume'] > 0) & 
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            )
            |
            (                                
                (dataframe['sma_9'] > (dataframe['sma_9'].shift(1) + dataframe['sma_9'].shift(1)*0.005)) &
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )
            
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
    
    plot_config = {
        'main_plot':{
            'ema_100':{},
            'ema_10':{},
            'sma_9':{}
        }     
    }  