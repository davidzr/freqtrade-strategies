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
class PRICEFOLLOWING2(IStrategy):
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
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.025,
        "30": 0.03,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    # Hyperoptable parameters
    rsi_value = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    rsi_enabled = BooleanParameter(default=False, space='buy', optimize=True, load=True)
    ema_pct = DecimalParameter(0.0001, 0.1, decimals = 4, default = 0.004, space="buy", optimize=True)

    ema_sell_pct = DecimalParameter(0.0001, 0.1, decimals = 4, default = 0.003, space="sell", optimize=True, load=True)
    sell_rsi_value = IntParameter(low=25, high=100, default=70, space='sell', optimize=True, load=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True, load=True)
    
            
    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 15

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
            'ha_open':{},
            'ha_close':{},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
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
        return [("ETH/USDT", "15m"),
                ("BTC/USDT", "15m"),
                ("RVN/USDT", "15m")
                        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

       
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        #rsi = 0.1 * (dataframe['rsi'] - 50)
        #dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

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

        # # EMA - Exponential Moving Average
        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema24'] = ta.EMA(dataframe, timeperiod=24)
        #dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        #dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        
        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']
        
        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        
        
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True)

        last_ema7 = dataframe['ema7'].tail()
        last_tema = dataframe['tema'].tail()
        haclose = dataframe['ha_close'].tail(4)
        haclose4thlast, haclose3rdlast, haclose2ndlast, hacloselast = haclose
        haopen = dataframe['ha_open']
        Conditions = []
        #GUARDS
        if self.rsi_enabled.value:
                Conditions.append(dataframe['rsi'] < self.rsi_value.value)
                Conditions.append(qtpylib.crossed_below(dataframe['ema7'], dataframe['tema']))
        else:
            Conditions.append(qtpylib.crossed_below(dataframe['ema7'], dataframe['tema']))
            #Conditions.append(haclose3rdlast > haclose2ndlast > hacloselast)
            #Conditions.append(dataframe['tema'] < dataframe['tema'].shift(1))
            #Conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['ema7']))
            Conditions.append(((last_tema - last_ema7) / last_tema) < self.ema_pct.value)
       
        if Conditions:
             dataframe.loc[
                 reduce(lambda x, y: x & y, Conditions),
                 'buy'] = 1
       
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            haopen = dataframe['ha_open']
            haclose = dataframe['ha_close']
            last_ema7 = dataframe['ema7'].tail()
            last_tema = dataframe['tema'].tail()
            haclose = dataframe['ha_close'].tail(2)
            haclose2ndlast, hacloselast = haclose
            conditions = []
            # GUARDS AND TRENDS
            if self.sell_rsi_enabled.value:
                conditions.append(dataframe['rsi'] < self.sell_rsi_value.value)
                conditions.append(qtpylib.crossed_above(dataframe['ema7'], dataframe['tema']))
            else:
                #conditions.append(haclose2ndlast > hacloselast)
                conditions.append(qtpylib.crossed_above(dataframe['ema7'], dataframe['tema']))
                conditions.append(dataframe['best_bid'] < haclose)
                conditions.append(((last_tema - last_ema7) / last_ema7) < self.ema_sell_pct.value)

            if conditions:
                 dataframe.loc[
                      reduce(lambda x, y: x & y, conditions),
                      'sell'] = 1

            return dataframe
