# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

class InverseV2(IStrategy):
    
    INTERFACE_VERSION = 2
    
    # Buy hyperspace params:
    buy_params = {
        "buy_fisher_cci_1": -0.42,
        "buy_fisher_cci_2": 0.41, 
        "buy_fisher_length": 31,  
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_fisher_cci_1": 0.42,  
        "sell_fisher_cci_2": -0.34, 
    }

    # ROI table:  
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.078
    trailing_stop_positive_offset = 0.174
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '1h'
    info_timeframe = '4h'
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        'main_plot': {
        },
        'subplots': {
            "fisher": {
                'fisher_stoch': {'color': 'blue'},
                'fisher_cci': {'color': 'red'},
                'fisher_rsi': {'color': 'black'},
                'fisher_mfi': {'color': 'purple'},
            },
        }
    }
    
    # Hyperoptable parameters
    buy_fisher_length = IntParameter(low=13, high=55, default=34, space="buy", optimize=True, load=True)
    buy_fisher_cci_1 = DecimalParameter(low=-0.6, high=-0.3, decimals=2, default=-0.5, space='buy', optimize=True, load=True)
    buy_fisher_cci_2 = DecimalParameter(low=0.3, high=0.6, decimals=2, default=0.5, space='buy', optimize=True, load=True)
    
    sell_fisher_cci_1 = DecimalParameter(low=0.3, high=0.6, decimals=2, default=0.5, space='sell', optimize=True, load=True)
    sell_fisher_cci_2 = DecimalParameter(low=-0.6, high=-0.3, decimals=2, default=-0.5, space='sell', optimize=True, load=True)
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]

        if (last_candle is not None):
            # if (sell_reason in ['roi','sell_signal','trailing_stop_loss']):
            if (sell_reason in ['sell_signal']):
                if last_candle['di_up'] and (last_candle['adx'] > previous_candle_1['adx']):
                    return False
        return True
    
    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.info_timeframe) for pair in pairs]
        
        informative_pairs.append(('BTC/USDT', self.info_timeframe))
        
        return informative_pairs

    def informative_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        
        # Get the informative pair
        informative_p = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe)

        # EMA
        informative_p['ema_50'] = ta.EMA(informative_p, timeperiod=50)
        informative_p['ema_100'] = ta.EMA(informative_p, timeperiod=100)
        informative_p['ema_200'] = ta.EMA(informative_p, timeperiod=200)
        
        # SSL Channels
        ssl_down, ssl_up = self.SSLChannels(informative_p, 20)
        informative_p['ssl_down'] = ssl_down
        informative_p['ssl_up'] = ssl_up

        return informative_p
    
    def informative_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        
        informative_btc = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe=self.info_timeframe)
        
        informative_btc['btc_cci'] = ta.CCI(informative_btc)
        
        return informative_btc
    
    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_fisher_length.value)
        # # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # wmarsi = ta.WMA(rsi, timeperiod = 9)
        # dataframe['fisher_rsi'] = (numpy.exp(2 * wmarsi) - 1) / (numpy.exp(2 * wmarsi) + 1)
        
        # # MFI - Money Flow Index
        # dataframe['mfi'] = ta.MFI(dataframe, timeperiod=self.buy_fisher_length.value)
        # # Inverse Fisher transform on MFI
        # mfi = 0.1 * (dataframe['mfi'] - 50)
        # wmamfi = ta.WMA(mfi, timeperiod = 9)
        # dataframe['fisher_mfi'] = (numpy.exp(2 * wmamfi) - 1) / (numpy.exp(2 * wmamfi) + 1)
        
        # # Stochastic
        # stoch_fast = ta.STOCHF(dataframe, fastk_period=self.buy_fisher_length.value)
        # dataframe['fastk'] = stoch_fast['fastk']
        # # Inverse Fisher transform on Stochastic
        # stoch = 0.1 * (dataframe['fastk'] - 50)
        # wmastoch = ta.WMA(stoch, timeperiod = 9)
        # dataframe['fisher_stoch'] = (numpy.exp(2 * wmastoch) - 1) / (numpy.exp(2 * wmastoch) + 1)
        
        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        for cci_length in self.buy_fisher_length.range:
            dataframe[f'cci'] = ta.CCI(dataframe, timeperiod=cci_length)
            # Inverse Fisher transform on CCI
            cci = 0.1 * (dataframe[f'cci'] / 4)
            wmacci = ta.WMA(cci, timeperiod = 9)
            dataframe[f'fisher_cci_{cci_length}'] = (numpy.exp(2 * wmacci) - 1) / (numpy.exp(2 * wmacci) + 1)
        
        # dataframe['fisher_average'] = (
        #     (dataframe['fisher_rsi'] + 
        #      dataframe['fisher_cci'] + 
        #      dataframe['fisher_mfi'] + 
        #      dataframe['fisher_stoch']
        #     ) / 4).astype(float)
        
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # confirm_trade_exit
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=3)
        dataframe['di_up'] = ta.PLUS_DI(dataframe, timeperiod=3) > ta.MINUS_DI(dataframe, timeperiod=3)
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        '''
        --> Informative timeframe
        ___________________________________________________________________________________________
        '''
        if self.info_timeframe != 'none':
            informative_p = self.informative_indicators(dataframe, metadata)
            dataframe = merge_informative_pair(dataframe, informative_p, self.timeframe, self.info_timeframe, ffill=True)
            drop_columns = [(s + "_" + self.info_timeframe) for s in ['date']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
            
            
        '''
        --> Informative btc timeframe
        ___________________________________________________________________________________________
        '''
        if self.info_timeframe != 'none':
            informative_btc = self.informative_btc_indicators(dataframe, metadata)
            dataframe = merge_informative_pair(dataframe, informative_btc, self.timeframe, self.info_timeframe, ffill=True)
            drop_columns = [(s + "_" + self.info_timeframe) for s in ['date']]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> The indicators for the normal timeframe
        ___________________________________________________________________________________________
        '''
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_above(dataframe[f'fisher_cci_{self.buy_fisher_length.value}'], self.buy_fisher_cci_1.value))
                    |
                    (
                        (qtpylib.crossed_below(dataframe[f'fisher_cci_{self.buy_fisher_length.value}'], self.buy_fisher_cci_2.value).rolling(8).max() == 1) &
                        (qtpylib.crossed_above(dataframe[f'fisher_cci_{self.buy_fisher_length.value}'], self.buy_fisher_cci_2.value))
                    )
                ) &
                (dataframe[f'ssl_up_{self.info_timeframe}'] > dataframe[f'ssl_down_{self.info_timeframe}']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe[f'ema_50_{self.info_timeframe}'] > dataframe[f'ema_100_{self.info_timeframe}']) &
                (dataframe[f'ema_50_{self.info_timeframe}'] > dataframe[f'ema_200_{self.info_timeframe}']) &
                
                (dataframe[f'btc_cci_{self.info_timeframe}'] < 0) &
                
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_below(dataframe[f'fisher_cci_{self.buy_fisher_length.value}'], self.sell_fisher_cci_1.value)) 
                    | (qtpylib.crossed_below(dataframe[f'fisher_cci_{self.buy_fisher_length.value}'], self.sell_fisher_cci_2.value))    
                ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

    # SSL Channels
    def SSLChannels(self, dataframe, length = 7):
        df = dataframe.copy()
        df['ATR'] = ta.ATR(df, timeperiod=14)
        df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
        df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
        df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
        df['hlv'] = df['hlv'].ffill()
        df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
        df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
        return df['sslDown'], df['sslUp']
    