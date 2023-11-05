# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import timedelta, datetime, timezone

# --------------------------------


class Macd(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AwesomeMacd.cs

    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"

    # Optimal stoploss designed for the strategy
    stoploss = -0.1

    # Optimal timeframe for the strategy
    timeframe = '1h'

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the currentrate
        """
        if current_profit > 0.3: 
            return -0.01 + current_profit
        # if pair in ('ONE/USDT', 'MATIC/USDT', 'CHZ/USDT', 'ENJ/USDT'):
        #     return -0.10
        return 1

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
        # get access to all 5mirs available in whitelist.
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1d') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        macd = ta.MACD(dataframe)

        #---------------- INFORMATIVE ----------------
        inf_tf2 = '1d'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf2)
        # Get the 14 day rsi
        #6, 25

        macd1d = ta.MACD(informative, 12, 26, 9)
        informative['macdhist'] = macd1d['macdhist']
        informative['macd'] = macd1d['macd']
        informative['macdsignal'] = macd1d['macdsignal']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf2, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (
                        dataframe['macdhist_1d'] > 0
                         
                    )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['macdhist_1d'] < 0) 

            ),
            'sell'] = 1
        return dataframe
