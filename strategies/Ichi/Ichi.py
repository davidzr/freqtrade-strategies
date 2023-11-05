# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
from technical.indicators import ichimoku

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from functools import reduce
# from freqtrade.state import RunMode
import logging
import os

logger = logging.getLogger(__name__)

LOG_FILENAME = datetime.now().strftime('logfile_%d_%m_%Y.log')
os.system("rm " + LOG_FILENAME)

# This will have an impact on all the logging from FreqTrade, when using other strategies than this one !!! 
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s :: %(message)s')

logging.info("test")


class Ichi(IStrategy):
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    minimal_roi = {
        "60": 0,
        "45": 0.0025 / 2,
        "30": 0.003 / 2,
        "15": 0.005 / 2,
        "10": 0.075 / 2,
        "5": 0.01 / 2,
        "0": 0.02 / 2,
    }
    timeframe = '15m'
    stoploss = -0.20

    INTERFACE_VERSION = 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ichi = ichimoku(dataframe)
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']
        dataframe['chikou'] = ichi['chikou_span']

        # EMA
        #        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        #        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        #        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        #        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['open'].shift(1) < dataframe['senkou_b'].shift(1))
                        & (dataframe['close'].shift(1) > dataframe['senkou_b'].shift(1))
                        & (dataframe['open'] > dataframe['senkou_b'])
                        & (dataframe['close'] > dataframe['senkou_b'])

                    #				& (dataframe['cloud_green'] == True)
                    #				&(dataframe['tenkan'] > dataframe['kijun'])
                    #				&(dataframe['tenkan'].shift(1) < dataframe['kijun'].shift(1))
                    #				& (dataframe['close'].shift(-26) > dataframe['close'].shift(26))
                )
                #			 &
                #             (
                #                (dataframe['ema21'] > dataframe['ema55']) & # Wenn EMA21 > EMA51
                #                (dataframe['ema13'] > dataframe['ema21']) &
                #                (dataframe['ema8'] > dataframe['ema13'])
                #             )

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #        dataframe.loc[
        #            (
        #                (dataframe['close'].shift(-26) <= dataframe['close'].shift(26))
        #            ),
        #            'sell'] = 1
        return dataframe
