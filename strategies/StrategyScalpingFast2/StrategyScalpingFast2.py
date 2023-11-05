# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import timeframe_to_minutes
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge
from functools import reduce
import numpy  # noqa
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class StrategyScalpingFast2(IStrategy):
    """
        Based on ReinforcedSmoothScalp
        https://github.com/freqtrade/freqtrade-strategies/blob/master/user_data/strategies/berlinguyinca/ReinforcedSmoothScalp.py
        this strategy is based around the idea of generating a lot of potentatils buys and make tiny profits on each trade

        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses
    """
    # Buy hyperspace params:
    buy_params = {
        "mfi-value": 19,
        "fastd-value": 29,
        "fastk-value": 19,
        "adx-value": 30,
        "mfi-enabled": False,
        "fastd-enabled": False,
        "adx-enabled": False,
        "fastk-enabled": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell-mfi-value": 89,
        "sell-fastd-value": 72,
        "sell-fastk-value": 68,
        "sell-adx-value": 86,
        "sell-cci-value": 157,
        "sell-mfi-enabled": True,
        "sell-fastd-enabled": True,
        "sell-adx-enabled": True,
        "sell-cci-enabled": False,
        "sell-fastk-enabled": False,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.082,
        "18": 0.06,
        "51": 0.012,
        "123": 0
    }
    use_sell_signal = False
    # Stoploss:
    stoploss = -0.326
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    #minimal_roi = {
    #    "0": 0.02
    #}
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # should not be below 3% loss

    #stoploss = -0.1
    # Optimal timeframe for the strategy
    # the shorter the better
    timeframe = '1m'

    # resample factor to establish our general trend. Basically don't buy if a trend is not given
    resample_factor = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tf_res = timeframe_to_minutes(self.timeframe) * self.resample_factor
        df_res = resample_to_interval(dataframe, tf_res)
        df_res['sma'] = ta.SMA(df_res, 50, price='close')
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)
        dataframe['resample_sma'] = dataframe[f'resample_{tf_res}_sma']

        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe["volume"] > 0)
        conditions.append(dataframe['open'] < dataframe['ema_low'])
        conditions.append(dataframe['resample_sma'] < dataframe['close'])

        if self.buy_params['adx-enabled']:
            conditions.append(dataframe["adx"] < self.buy_params['adx-value'])
        if self.buy_params['mfi-enabled']:
            conditions.append(dataframe['mfi'] < self.buy_params['mfi-value'])
        if self.buy_params['fastk-enabled']:
            conditions.append(dataframe['fastk'] < self.buy_params['fastk-value'])
        if self.buy_params['fastd-enabled']:
            conditions.append(dataframe['fastd'] < self.buy_params['fastd-value'])
        if self.buy_params['fastk-enabled'] == True & self.buy_params['fastd-enabled'] == True:
            conditions.append(qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
        
        # |
        # # try to get some sure things independent of resample
        # ((dataframe['rsi'] - dataframe['mfi']) < 10) &
        # (dataframe['mfi'] < 30) &
        # (dataframe['cci'] < -200)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1            
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['open'] >= dataframe['ema_high'])

        if self.sell_params['sell-fastd-enabled'] == True | self.sell_params['sell-fastk-enabled'] == True:
            conditions.append((qtpylib.crossed_above(dataframe['fastk'], self.sell_params['sell-fastk-value'])) |
                              (qtpylib.crossed_above(dataframe['fastd'], self.sell_params['sell-fastd-value'])))
        if self.sell_params['sell-cci-enabled'] == True:
            conditions.append(dataframe['cci'] > 100)
        if self.sell_params['sell-mfi-enabled'] == True:
            conditions.append(dataframe['mfi'] > self.sell_params['sell-mfi-value'])
        if self.sell_params['sell-adx-enabled'] == True:
            conditions.append(dataframe["adx"] < self.sell_params['sell-adx-value'])            

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "sell"] = 1              
        return dataframe