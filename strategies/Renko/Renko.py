import talib.abstract as ta
from pandas import DataFrame, Series, DatetimeIndex, merge
import pandas as pd
import numpy as np
#import pdb 
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy 
pd.set_option("display.precision", 10) 

class Renko(IStrategy):
 
    minimal_roi = {
        "0": 100
    }

    stoploss = -100

    timeframe = '15m'    
    
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.1
    ignore_roi_if_buy_signal = True
 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=5)
        brick_size = np.mean(dataframe['ATR'])
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ATR']
        df = dataframe[columns] 
        cdf = pd.DataFrame(
            columns=columns,
            data=[],
        ) 
        cdf.loc[0] = df.loc[0]
        close = df.loc[0]['close'] 
        volume = df.loc[0]['volume'] 
        cdf.iloc[0, 1:] = [close - brick_size, close, close - brick_size, close, volume, brick_size]
        cdf['trend'] = True  
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ATR', 'trend']

        for index, row in df.iterrows():  
            if not np.isnan(row['ATR']): brick_size = row['ATR'] 
            close = row['close']
            date = row['date'] 
            volume = row['volume'] 
            row_p1 = cdf.iloc[-1] 
            trend = row_p1['trend']
            close_p1 = row_p1['close'] 
            bricks = int(np.nan_to_num((close - close_p1) / brick_size))
            data = [] 
            if trend and bricks >= 1:
                for i in range(bricks):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, volume, brick_size, trend]
                    data.append(r)
                    close_p1 += brick_size
            elif trend and bricks <= -2:
                trend = not trend
                bricks += 1
                close_p1 -= brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, volume, brick_size, trend]
                    data.append(r)
                    close_p1 -= brick_size
            elif not trend and bricks <= -1:
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size, volume, brick_size, trend]
                    data.append(r)
                    close_p1 -= brick_size
            elif not trend and bricks >= 2:
                trend = not trend
                bricks -= 1
                close_p1 += brick_size
                for i in range(abs(bricks)):
                    r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size, volume, brick_size, trend]
                    data.append(r)
                    close_p1 += brick_size
            else:
                continue

            sdf = pd.DataFrame(data=data, columns=columns)
            cdf = pd.concat([cdf, sdf]) 

        renko_df = cdf.groupby(['date']).last()
        renko_df = renko_df.reset_index() 
        renko_df['previous-trend'] = renko_df.trend.shift(1)   
        renko_df['previous-trend2'] = renko_df.trend.shift(2)   

        return renko_df

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for index, row in dataframe.iterrows():  
            if row['previous-trend'] == False and row['trend'] == True:
                last_row = dataframe.loc[dataframe['date'] == row['date']][-1:] 
                dataframe.loc[dataframe.index== last_row.index.values[0], 'buy'] = 1
            if row['previous-trend'] == True and row['trend'] == True:
                last_row = dataframe.loc[dataframe['date'] == row['date']][-1:] 
                dataframe.loc[dataframe.index== last_row.index.values[0], 'buy'] = 1
            else:
                last_row = dataframe.loc[dataframe['date'] == row['date']][-1:] 
                dataframe.loc[dataframe.index== last_row.index.values[0], 'sell'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe