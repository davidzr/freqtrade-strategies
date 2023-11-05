# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import RealParameter, IntParameter
from functools import reduce
from pandas import DataFrame



shma = 15 #short hull moving average
lhma = 35 #long hull moving average

# shma_c = 10 #short hull moving average de cierre
# lhma_c = 25 #long hull moving average de cierre

pwill = 12 #period williams r
pvol = 100 #period pvt
# pmv = 50 #period ema williams r

class e6v34(IStrategy):

    bwill = RealParameter(-25, -15, default=-20, space='buy')
    swill = RealParameter(-55, -35, default=-50, space='sell')

    # bwill = -20 #buy williams r
    # swill = -50 #sell williams r

    # ROI table:
    minimal_roi = {
        "0": 0.555,
        "300": 0.35,
        "500": 0.247,
        "1200": 0.0936,
        "2100": 0.04,
        "4000": 0
    }

    # Stoploss:
    stoploss = -0.54

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.18
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = True

    # Optimal timeframe use it in your config
    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMA - ex Moving Average
        dataframe[f'hma{shma}'] = qtpylib.hma(dataframe['close'], window=shma)
        dataframe[f'hma{lhma}'] = qtpylib.hma(dataframe['close'], window=lhma)
        # dataframe[f'hma{shma_c}'] = qtpylib.hma(dataframe['close'], window=shma_c)
        # dataframe[f'hma{lhma_c}'] = qtpylib.hma(dataframe['close'], window=lhma_c)
        dataframe['willr'] = ta.WILLR(dataframe['high'], dataframe['low'],dataframe['close'], timeperiod=pwill)
        # dataframe['will_mean'] = ta.EMA(dataframe, timeperiod=pmv, price='willr')
        dataframe['vol_mean'] = ta.EMA(dataframe, timeperiod=pvol, price='volume')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        conditions.append(dataframe[f'hma{shma}'].shift(1) - dataframe[f'hma{lhma}'].shift(1) < dataframe[f'hma{shma}'] - dataframe[f'hma{lhma}'])
        conditions.append(dataframe[f'hma{shma}'].shift(2) - dataframe[f'hma{lhma}'].shift(2) < dataframe[f'hma{shma}'].shift(1)  - dataframe[f'hma{lhma}'].shift(1) )

        conditions.append(dataframe['willr'] > self.bwill.value)
        conditions.append(dataframe['willr'].shift(1) < self.bwill.value)
        # conditions.append(dataframe['will_mean'] > -50)

        conditions.append(dataframe['volume'] > dataframe['vol_mean'])

        if conditions:
            dataframe.loc[
                reduce(lambda x,y: x&y, conditions),
                'buy']=1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # ( (dataframe[f'hma{lhma_c}'] > dataframe[f'hma{shma_c}']) &
                # (dataframe[f'hma{lhma_c}'].shift(1) < dataframe[f'hma{shma_c}'].shift(1)) ) |

                (dataframe['willr'] < self.swill.value) &
                (dataframe['willr'].shift(1) > self.swill.value)
            ),
            'sell'] = 1

        return dataframe