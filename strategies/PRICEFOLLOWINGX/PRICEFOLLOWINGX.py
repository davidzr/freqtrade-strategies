# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce

# This class is a sample. Feel free to customize it.
class PRICEFOLLOWINGX(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    @property
    def protections(self):
            return [
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": 48,
                    "trade_limit": 5,
                    "stop_duration_candles": 5,
                    "max_allowed_drawdown": 0.75
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24,
                    "trade_limit": 3,
                    "stop_duration_candles": 5,
                    "only_per_pair": True
                },
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 30,
                    "trade_limit": 2,
                    "stop_duration_candles": 6,
                    "required_profit": 0.005
                },
            ]
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "120":0.015,
        "60": 0.025,
        "30": 0.03,
        "0": 0.015
       }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.5

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    # Hyperoptable parameters
    rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True, load=True)
    #buy_rsi = DecimalParameter(0, 50, decimals = 2, default = 40, space="buy", optimize=True, load=True)
    ema_pct = DecimalParameter(0.001, 0.100, decimals = 3, default = 0.040, space="buy", optimize=True, load=True)
    buy_frsi = DecimalParameter(-0.71, 0.50, decimals = 2, default = -0.40, space="buy", optimize=True, load=True)
    frsi_pct = DecimalParameter(0.01, 0.20, decimals = 2, default = 0.10, space="buy", optimize=True, load=True)
    #sellspace
    ema_sell_pct = DecimalParameter(0.001, 0.020, decimals = 3, default = 0.003, space="sell", optimize=True, load=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True, load=True)
    sell_frsi = DecimalParameter(-0.30, 0.70, decimals=2, default=0.2, space="sell", load=True)


    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'ema7':{},
            'ema7low':{},
            'ema10high':{},
            'ha_open':{},
            'ha_close':{},
        },
        'subplots': {
            #"MACD": {
            #    'macd': {'color': 'blue'},
            #    'macdsignal': {'color': 'orange'},
            #},
            "RSI": {
                'frsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            
        """
        return [("ETH/BUSD", "1h"),
                ("LINK/BUSD", "1h"),
                ("RVN/BUSD", "1h"),
                ("MATIC/BUSD", "30m")
                        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

       
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, window=14)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['frsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

       
        # Stochastic Fast
        #stoch_fast = ta.STOCHF(dataframe)
        #dataframe['fastd'] = stoch_fast['fastd']
        #dataframe['fastk'] = stoch_fast['fastk']

       
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Parabolic SAR
        #dataframe['sar'] = ta.SAR(dataframe)
        #Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=19, stds=2.2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        #hilbert = ta.HT_SINE(dataframe)
        #dataframe['htsine'] = hilbert['sine']
        #dataframe['htleadsine'] = hilbert['leadsine']
        
        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']


        # # EMA - Exponential Moving Average
        dataframe['ema7'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['emalow'] = ta.EMA(dataframe, timeperiod=12, price='low')
        dataframe['emahigh'] = ta.EMA(dataframe, timeperiod=14, price='high')
        #dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True)

        last_emalow = dataframe['emalow'].tail()
        last_tema = dataframe['tema'].tail()
        haclose = dataframe['ha_close'].tail(3)
        haclose3rdlast, haclose2ndlast, hacloselast = haclose
        haopen = dataframe['ha_open']
        Conditions = []
        #GUARDS
        if self.rsi_enabled.value:
           Conditions.append(qtpylib.crossed_below(dataframe['frsi'], self.buy_frsi.value))
           Conditions.append(dataframe['tema'] < dataframe['bb_lowerband'])
           Conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['emalow']))
           #Conditions.append(dataframe['best_bid'] < dataframe['bb_lowerband'])
            
        else:
           Conditions.append(dataframe['tema'] > dataframe['bb_middleband'])
           Conditions.append(qtpylib.crossed_above(dataframe['tema'], dataframe['ema7']))
           #Conditions.append(dataframe['best_bid'] < dataframe['bb_lowerband'])
           #Conditions.append(((abs(last_emalow - last_tema)) / last_tema) > self.ema_sell_pct.value)
        
        if Conditions:
             dataframe.loc[
                 reduce(lambda x, y: x & y, Conditions),
                 'buy'] = 1

        return dataframe

        #if dataframe['buy'] != 1:

          #    dataframe.loc[
          #       (qtpylib.crossed_above(dataframe['tema'], dataframe['ha_high']))&
          #       (60 < dataframe['rsi'] < 90 )&
          #       (dataframe['macd'] > dataframe['macdsignal']),
          #       'buy'] = 1

        
        #return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

            haopen = dataframe['ha_open']
            haclose = dataframe['ha_close']
            last_tema = dataframe['tema'].tail()
            last_emahigh = dataframe['emahigh'].tail()
            conditions = []
            # GUARDS AND TRENDS
            if self.sell_rsi_enabled.value:
                 conditions.append(qtpylib.crossed_below(dataframe['frsi'], self.sell_frsi.value))
                 conditions.append(dataframe['tema'] < dataframe['bb_middleband'])
                 #conditions.append(haclose2ndlast > hacloselast)
                 conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['ema7']))
                 #conditions.append(dataframe['best_bid'] < dataframe['ha_close'].shift(1))
            else:
                 conditions.append(dataframe['tema'] < dataframe['bb_middleband'])
                 conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['ema7']))
                 #conditions.append(dataframe['best_bid'] < dataframe['ha_close'].shift(1))
                 #conditions.append(((abs(last_emahigh - last_tema)) / last_tema) > self.ema_sell_pct.value)

            if conditions:
                 dataframe.loc[
                      reduce(lambda x, y: x & y, conditions),
                      'sell'] = 1

            return dataframe
