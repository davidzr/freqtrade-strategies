# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (merge_informative_pair, 
                                BooleanParameter, 
                                CategoricalParameter, 
                                DecimalParameter,
                                IStrategy, 
                                IntParameter)

# --------------------------------
# Add your lib to import here
# import talib.abstract as ta
import ta
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class TrixV15Strategy(IStrategy):
    """
    Sources : 
    Cripto Robot : https://www.youtube.com/watch?v=uE04UROWkjs&list=PLpJ7cz_wOtsrqEQpveLc2xKLjOBgy4NfA&index=4
    Github : https://github.com/CryptoRobotFr/TrueStrategy/blob/main/TrixStrategy/Trix_Complete_backtest.ipynb
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
      "0": 0.553,
      "423": 0.144,
      "751": 0.059,
      "1342": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.31

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = True
    # trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.9  # Disabled / not configured
    
    # Buy hyperspace params:
    buy_params = {
      "buy_stoch_rsi_enabled": True,
      "buy_ema_multiplier": 0.85,
      "buy_ema_src": "open",
      "buy_ema_timeperiod": 10,
      "buy_ema_timeperiod_enabled": True,
      "buy_stoch_rsi": 0.901,
      "buy_trix_signal_timeperiod": 19,
      "buy_trix_signal_type": "trigger",
      "buy_trix_src": "low",
      "buy_trix_timeperiod": 8
    }

    sell_params = {
      "sell_stoch_rsi_enabled": True,
      "sell_stoch_rsi": 0.183,
      "sell_trix_signal_timeperiod": 19,
      "sell_trix_signal_type": "trailing",
      "sell_trix_src": "high",
      "sell_trix_timeperiod": 10
    }

    # HYPEROPTABLE PARAMETERS
    # buy
    buy_stoch_rsi_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    buy_stoch_rsi = DecimalParameter(0.6, 0.99, decimals=3, default=0.987, space="buy", optimize=True, load=True)
    buy_trix_timeperiod = IntParameter(7, 13, default=12, space="buy", optimize=True, load=True)
    buy_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=True, load=True)
    buy_trix_signal_timeperiod = IntParameter(19, 25, default=22, space="buy", optimize=True, load=True)
    buy_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trigger', space="buy", optimize=True, load=True)
    buy_ema_timeperiod_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    buy_ema_timeperiod = IntParameter(9, 100, default=21, space="buy", optimize=True, load=True)
    buy_ema_multiplier = DecimalParameter(0.8, 1.2, decimals=2, default=1.00, space="buy", optimize=True, load=True)
    buy_ema_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=True, load=True)
    # sell
    sell_stoch_rsi_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sell_stoch_rsi = DecimalParameter(0.01, 0.4, decimals=3, default=0.048, space="sell", optimize=True, load=True)
    sell_trix_timeperiod = IntParameter(7, 11, default=9, space="sell", optimize=True, load=True)
    sell_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="sell", optimize=True, load=True)
    sell_trix_signal_timeperiod = IntParameter(17, 23, default=19, space="sell", optimize=True, load=True)
    sell_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trailing', space="sell", optimize=True, load=True)

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 19

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
            'trix_b_8': {'color': 'blue'},
            'trix_s_10': {'color': 'orange'},
            'ema_b_signal': {'color': 'red'},
        },
         'subplots': {
            "TRIX BUY": {
                'trix_b_pct': {'color': 'blue'},
                'trix_b_signal_19': {'color': 'orange'},
            },
            "TRIX SELL": {
                'trix_s_pct': {'color': 'blue'},
                'trix_s_signal_19': {'color': 'orange'},
            },
            "STOCH RSI": {
                'stoch_rsi': {'color': 'blue'},
            },
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # # get access to all pairs available in whitelist.
        # pairs = self.dp.current_whitelist()
        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        # 
        # return informative_pairs

    # def informative_1d_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        #informative_1d = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        # SMA
        #informative_1d['1d_sma_200'] = ta.SMA(informative_1d, timeperiod=200)

        # return informative_1d
        return []

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
        # The indicators for the 1h informative timeframe
        # informative_1d = self.informative_1d_indicators(dataframe, metadata)
        # dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, self.informative_timeframe, ffill=True)

        # Momentum Indicators
        # ------------------------------------

        # # Stochastic RSI
        if self.buy_stoch_rsi_enabled.value or self.sell_stoch_rsi_enabled.value:
            dataframe['stoch_rsi'] = ta.momentum.stochrsi(close=dataframe['close'], window=14, smooth1=3, smooth2=3)

        # Overlap Studies
        # ------------------------------------

        # -- EMA --
        for val in self.buy_ema_timeperiod.range:
            dataframe[f'ema_b_{val}'] = ta.trend.ema_indicator(close=dataframe[self.buy_ema_src.value], window=val)
        dataframe['ema_b_signal'] = dataframe[f'ema_b_{self.buy_ema_timeperiod.value}'] * self.buy_ema_multiplier.value

        # -- Trix Indicator --
        for val in self.buy_trix_timeperiod.range:
            dataframe[f'trix_b_{val}'] = ta.trend.ema_indicator(ta.trend.ema_indicator(ta.trend.ema_indicator(close=dataframe[self.buy_trix_src.value], window=val), window=val), window=val)
        dataframe['trix_b_pct'] = dataframe[f'trix_b_{self.buy_trix_timeperiod.value}'].pct_change() * 100
        for val in self.buy_trix_signal_timeperiod.range:
            dataframe[f'trix_b_signal_{val}'] = ta.trend.sma_indicator(dataframe['trix_b_pct'], window=val)

        for val in self.sell_trix_timeperiod.range:
            dataframe[f'trix_s_{val}'] = ta.trend.ema_indicator(ta.trend.ema_indicator(ta.trend.ema_indicator(close=dataframe[self.sell_trix_src.value], window=val), window=val), window=val)
        dataframe['trix_s_pct'] = dataframe[f'trix_s_{self.sell_trix_timeperiod.value}'].pct_change() * 100
        for val in self.sell_trix_signal_timeperiod.range:
            dataframe[f'trix_s_signal_{val}'] = ta.trend.sma_indicator(dataframe['trix_s_pct'], window=val)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []

        # Guards and trends
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['trix_s_pct'] > dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])
        if self.buy_stoch_rsi_enabled.value:
            conditions.append(dataframe['stoch_rsi'] < self.buy_stoch_rsi.value)
        if self.buy_ema_timeperiod_enabled.value:
            conditions.append(dataframe['close'] > dataframe['ema_b_signal'])
        if self.buy_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_b_pct'] > dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}'])

        # Triggers
        if self.buy_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_above(dataframe['trix_b_pct'], dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        conditions = []

        # Guards and trends
        conditions.append(dataframe['volume'] > 0)
        if self.sell_stoch_rsi_enabled.value:
            conditions.append(dataframe['stoch_rsi'] > self.sell_stoch_rsi.value)
        if self.sell_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_s_pct'] < dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])
        
        # Triggers
        if self.sell_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_below(dataframe['trix_s_pct'], dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1
        return dataframe