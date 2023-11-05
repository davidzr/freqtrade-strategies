# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import technical.indicators as ftt
from freqtrade.strategy import stoploss_from_open
from freqtrade.exchange import timeframe_to_minutes

logger = logging.getLogger(__name__)

# Obelisk_3EMA_StochRSI_ATR - 2021-04-10
#
# by Obelisk
# https://github.com/brookmiles/
#
# DO NOT RUN LIVE
#
# based on Trade Pro "76% Win Rate Highly Profitable Trading Strategy Proven 100 Trades - 3 EMA + Stochastic RSI + ATR"
# https://www.youtube.com/watch?v=7NM7bR2mL7U
#
# correctness/accuracy not guaranteed
#
# WARNING
#
# While this strategy is designed to be run at 1h, it should be backtested at 5m (or 1m).
# This is done to avoid misleading results produced using trailing stops and roi values at longer timeframes.
#
# When running at 5m, an informative pair at 1h will be used to generate signals equivalent to running at 1h.
#
# live / dryrun: use 1h
# backtest / hyperopt: use 5m or 1m

class Obelisk_3EMA_StochRSI_ATR(IStrategy):

    # Backtest or hyperopt at this timeframe
    timeframe = '5m'

    # Live or Dry-run at this timeframe
    informative_timeframe = '1h'

    startup_candle_count = 500

    # NOTE: this strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.99
    use_custom_stoploss = True

    custom_info = {}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        #RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        #StochRSI
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi  = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        dataframe.loc[
            (dataframe['ema8'] > dataframe['ema14']) &
            (dataframe['ema14'] > dataframe['ema50']) &
            qtpylib.crossed_above(dataframe['srsi_k'], dataframe['srsi_d'])
        ,
        'go_long'] = 1
        dataframe['go_long'].fillna(0, inplace=True)

        dataframe.loc[
            qtpylib.crossed_above(dataframe['go_long'], 0),
            'take_profit'] = dataframe['close'] + dataframe['atr'] * 2
        dataframe['take_profit'].fillna(method='ffill', inplace=True)

        dataframe.loc[
            qtpylib.crossed_above(dataframe['go_long'], 0),
            'stop_loss'] = dataframe['close'] - dataframe['atr'] * 3
        dataframe['stop_loss'].fillna(method='ffill', inplace=True)

        dataframe.loc[
            qtpylib.crossed_above(dataframe['go_long'], 0),
            'stop_pct'] = (dataframe['atr'] * 3) / dataframe['close']
        dataframe['stop_pct'].fillna(method='ffill', inplace=True)

        # add indicator mapped to correct DatetimeIndex to custom_info
        self.custom_info[metadata['pair']] = dataframe[['date', 'stop_pct', 'take_profit']].copy().set_index('date')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
            # don't overwrite the base dataframe's OHLCV information
            skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            qtpylib.crossed_above(dataframe['go_long'], 0)
        ,
        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sell'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        result = 1

        custom_info_pair = self.custom_info[pair]
        if custom_info_pair is not None:
            # using current_time/open_date directly will only work in backtesting/hyperopt.
            # in live / dry-run, we have to search for nearest row before
            tz = custom_info_pair.index.tz
            open_date = trade.open_date_utc if hasattr(trade, 'open_date_utc') else trade.open_date.replace(tzinfo=custom_info_pair.index.tz)
            open_date_mask = custom_info_pair.index.unique().get_loc(open_date, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if(open_df is None or len(open_df) == 0):
                logger.debug("No open_df :(")
                return 1 # oh well

            # stop out if we have reached our take profit limit
            take_profit = open_df['take_profit']
            if (take_profit is not None):
                if current_rate > take_profit:
                    logger.debug("take_profit={}, current={}".format(
                        take_profit,
                        current_rate
                        ))
                    return 0.001

            # keep trailing stoploss at -stop_pct from the open price
            stop_pct = open_df['stop_pct']
            if (stop_pct is not None):
                new_stop = stoploss_from_open(-stop_pct, current_profit)
                logger.debug("open={}, current={}, profit={}, stop_pct={}, stop={}".format(
                    current_rate / (1 + current_profit),
                    current_rate,
                    current_profit,
                    stop_pct,
                    current_rate * (1 - new_stop)))
                if new_stop > 0:
                    result = new_stop

        return result

    plot_config = {
        'main_plot': {
            'ema50': { 'color': 'orange' },
            'ema14': { 'color': 'blue' },
            'ema8': { 'color': 'purple' },
            'take_profit': { 'color': 'green' },
            'stop_loss': { 'color': 'red' },
        },
        'subplots': {
            "SRSI": {
                'srsi_k': {'color': 'blue'},
                'srsi_d': {'color': 'red'},
            },
            "ATR": {
                'atr': {'color': 'blue'},
            },
        }
    }
