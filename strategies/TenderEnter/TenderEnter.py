# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import math
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
# from freqtrade.exchange import timeframe_to_minutes
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
# from freqtrade.strategy.strategy_helper import merge_informative_pair

class TenderEnter(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    custom_stops = {}
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "180":  0.05, # 5% after 240 min
    #     "200":  0.03,
    #     "240":  0.00,
    #     "0":  0.08 # 8% imidietly
    # }
    minimal_roi = {
        "0": 0.21296,
        "94": 0.13203,
        "190": 0.04443,
        "374": 0
    }
    # minimal_roi = {
    #     "180":  0.2, # 5% after 240 min
    #     "200":  0.1,
    #     "240":  0.00,
    #     "0":  0.3 # 8% imidietly
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # stoploss = -0.4
    stoploss = -0.25933

    # Trailing stoploss
    # trailing_stop = True
    # trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.03
    # trailing_only_offset_is_reached = True
    # trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
    trailing_stop = True
    trailing_stop_positive = 0.25571
    trailing_stop_positive_offset = 0.35142
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    inf_tf = '15m' #timeframe of second line


    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 102
    
    
    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    

    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
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
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        
        # return [(metadata['pair'], "15m")]
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
        # pairs = self.dp.current_whitelist()
        
        # self.custom_stops = {pair: False for pair in pairs}
        # for pair in pairs:
        #     vals = {}
        #     vals[pair]=False
        #     arr.append(vals)
        #     self.stopsByPair = [{pair: False} for pair in pairs]
        # print('TT', metadata['pair'])    
        # Momentum Indicators
        # ------------------------------------

        # ADX
        # dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']

        # MFI
        # dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']
        # dataframe["bb_percent"] = (
        #     (dataframe["close"] - dataframe["bb_lowerband"]) /
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        # )
        # dataframe["bb_width"] = (
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        # )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        # dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe['htsine'] = hilbert['sine']
        # dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        
     

        
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="15m")
        # dataframe["date"] = df["open_date"] + df["Delta"].map(pd.Timedelta.to_pytimedelta)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, "15m", ffill=True)
        
        # informative = self.dp.get_pair_dataframe(metadata['pair'], inf_tf)
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_tf)
        # informative = self.dp.get_pair_dataframe(pair=f"{self.stake_currency}/USDT", timeframe=self.inf_tf)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, True)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_tf, ffill=True)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
    
        dataframe.loc[(
            self.compareFields(dataframe, 'close', 1, 1017) &
            self.compareFields(dataframe, 'close', 2, 1017) &
            self.compareFields(dataframe, 'volume', 1, 65) &
            self.compareFields(dataframe, 'volume', 2, 65) &
            (dataframe['volume'] > 0)),'buy'] = 1
        return dataframe

    def compareFields(self, dt, fieldname, shift, ratio=1.034):
        return dt[fieldname].shift(shift)/dt[fieldname] > ratio/1000

    # def calcAngle(self, p1, p2, delta_x) -> bool:
    #     delta_y = p2 - p1
    #     theta_radians = np.arctan2(delta_y, delta_x)
    #     return theta_radians < -0.900275

    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, **kwargs) -> bool:
    #     print('z', metadata["pair"], self.custom_stops[metadata["pair"]])                    
    #     if self.custom_stops[metadata["pair"]] == False:
    #         self.custom_stops[metadata["pair"]] = True
    #         return True
    #     else:
    #         return False

    # def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
    #                        rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
    #     self.custom_stops[metadata["pair"]] = True
    #     return True

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                # False
                # (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 0
        return dataframe
    
    # def merge_informative_pair(dataframe, informative, minutes, inf_tf, ffill):
    #     print('>>', inf_tf,ffill)
    #     # Shift date by 1 candle
    #     # This is necessary since the data is always the "open date"
    #     # and a 15m candle starting at 12:15 should not know the close of the 1h candle from 12:00 to 13:00
    #     minutes = timeframe_to_minutes(inf_tf)
    #     # Only do this if the timeframes are different:
    #     informative['date_merge'] = informative["date"] + pd.to_timedelta(minutes, 'm')

    #     # Rename columns to be unique
    #     informative.columns = [f"{col}_{inf_tf}" for col in informative.columns]
    #     # Assuming inf_tf = '1d' - then the columns will now be:
    #     # date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

    #     # Combine the 2 dataframes
    #     # all indicators on the informative sample MUST be calculated before this point
    #     dataframe = pd.merge(dataframe, informative, left_on='date', right_on=f'date_merge_{inf_tf}', how='left')
    #     # FFill to have the 1d value available in every row throughout the day.
    #     # Without this, comparisons would only work once per day.
    #     dataframe = dataframe.ffill()
    #     return dataframe    