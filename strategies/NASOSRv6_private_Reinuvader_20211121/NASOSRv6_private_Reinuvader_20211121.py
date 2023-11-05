# THIS STRATEGY IS PRIVATE. DO NOT SHARE.
# Heavily borrowed code from MultiMA_TSL3, ClucHAnix_BB_RPB.
# Big thanks to all the giants whose shoulders this strat is standing on:
# @stash86
# @Al
# @Rallipanos
# @pluxury

from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.exchange import timeframe_to_minutes
import technical.indicators as ftt
from technical.indicators import zema

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 17,
    "ewo_high": 3.33,
    "ewo_high_2": -0.444,
    "ewo_low": -12.788,
    "lookback_candles": 1,
    "low_offset": 0.985,
    "low_offset_2": 0.935,
    "profit_threshold": 1.016,
    "clucha_rocr_1h": 0.47782,
    "clucha_bbdelta_close": 0.02206,
    "clucha_bbdelta_tail": 1.02515,
    "clucha_close_bblower": 0.03669,
    "clucha_closedelta_close": 0.04401,
    "base_nb_candles_buy_zema": 25,
    "base_nb_candles_buy_zema2": 53,
    "low_offset_zema": 0.958,
    "low_offset_zema2": 0.961,
    "base_nb_candles_buy_ema": 9,
    "base_nb_candles_buy_ema2": 75,
    "low_offset_ema": 1.067,
    "low_offset_ema2": 0.973,
    "high_offset_sell_ema": 0.994,
    "rsi_buy": 71
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 8,
    "high_offset": 1.01,
    "high_offset_2": 1.406,
    "ProfitLoss1": 0.005,
    "ProfitLoss2": 0.021,
    "ProfitMargin1": 0.018,
    "ProfitMargin2": 0.051,
    "pHSL": -0.15,
}


class NASOSRv6_private_Reinuvader_20211121(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 10
    }

    # Stoploss:
    stoploss = -0.15

    # SMAOffset
    base_nb_candles_buy = IntParameter(2, 26, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(2, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.85, 0.99, default=buy_params['low_offset'], space='buy', optimize=False)
    low_offset_2 = DecimalParameter(0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=False)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=False)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(1, 36, default=buy_params['lookback_candles'], space='buy', optimize=False)
    profit_threshold = DecimalParameter(1.0, 1.08, default=buy_params['profit_threshold'], space='buy', optimize=False)
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=False)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)
    rsi_buy = IntParameter(60, 82, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # trailing stoploss hyperopt parameters
    pHSL = DecimalParameter(-0.15, -0.08, default=sell_params['pHSL'], decimals=3, space='sell', optimize=True)
    ProfitMargin1 = DecimalParameter(0.009, 0.019, default=sell_params['ProfitMargin1'], decimals=3, space='sell', optimize=True)
    ProfitLoss1 = DecimalParameter(0.005, 0.012, default=sell_params['ProfitLoss1'], decimals=3, space='sell', optimize=True)
    ProfitMargin2 = DecimalParameter(0.033, 0.099, default=sell_params['ProfitMargin2'], decimals=3, space='sell', optimize=True)
    ProfitLoss2 = DecimalParameter(0.010, 0.025, default=sell_params['ProfitLoss2'], decimals=3, space='sell', optimize=True)

    # ClucHA
    clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=buy_params['clucha_bbdelta_close'], decimals=5, space='buy', optimize=False)
    clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=buy_params['clucha_bbdelta_tail'], decimals=5, space='buy', optimize=False)
    clucha_close_bblower = DecimalParameter(0.001, 0.05, default=buy_params['clucha_close_bblower'], decimals=5, space='buy', optimize=False)
    clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=buy_params['clucha_closedelta_close'], decimals=5, space='buy', optimize=False)
    clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=buy_params['clucha_rocr_1h'], decimals=5, space='buy', optimize=False)

    # Zema
    base_nb_candles_buy_zema = IntParameter(5, 80, default=buy_params['base_nb_candles_buy_zema'], space='buy', optimize=False)
    low_offset_zema = DecimalParameter(0.9, 0.99, default=buy_params['low_offset_zema'], space='buy', optimize=False)
    base_nb_candles_buy_zema2 = IntParameter(5, 80, default=buy_params['base_nb_candles_buy_zema2'], space='buy', optimize=False)
    low_offset_zema2 = DecimalParameter(0.9, 0.99, default=buy_params['low_offset_zema2'], space='buy', optimize=False)
    high_offset_sell_ema = DecimalParameter(0.99, 1.1, default=buy_params['high_offset_sell_ema'], space='buy', optimize=False)
    base_nb_candles_buy_ema = IntParameter(5, 80, default=buy_params['base_nb_candles_buy_ema'], space='buy', optimize=False)
    low_offset_ema = DecimalParameter(0.9, 1.1, default=buy_params['low_offset_ema'], space='buy', optimize=False)
    base_nb_candles_buy_ema2 = IntParameter(5, 80, default=buy_params['base_nb_candles_buy_ema2'], space='buy', optimize=False)
    low_offset_ema2 = DecimalParameter(0.9, 1.1, default=buy_params['low_offset_ema2'], space='buy', optimize=False)

    # Trailing stop:
    trailing_stop = False
    #trailing_stop_positive = 0.001
    #trailing_stop_positive_offset = 0.016
    #trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    #sell_profit_only = False
    #sell_profit_offset = -0.0001
    #ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    timeperiods = [
        # 50 // timeframe_minutes,
        # 85 // timeframe_minutes,
        180 // timeframe_minutes,
        360 // timeframe_minutes,
        420 // timeframe_minutes,
        560 // timeframe_minutes,
    ]

    process_only_new_candles = True
    startup_candle_count = 400
    use_custom_stoploss = True

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        if ((current_time - trade.open_date_utc).seconds / 60 > 1440):
            return 'unclog'

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        HSL = self.pHSL.value
        if (current_profit > self.ProfitMargin2.value):
            sl_profit = self.ProfitLoss2.value
        elif (current_profit > self.ProfitMargin1.value):
            sl_profit = self.ProfitLoss1.value + ((current_profit - self.ProfitMargin1.value) * (self.ProfitLoss2.value - self.ProfitLoss1.value) / (self.ProfitMargin2.value - self.ProfitMargin1.value))
        else:
            sl_profit = HSL

        return sl_profit

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        if hasattr(last_candle, 'sell_tag') and str(last_candle.sell_tag) != "nan":
            if (sell_reason == "sell_signal"):
                sell_reason = last_candle.sell_tag
            else:
                sell_reason += last_candle.sell_tag

        trade.sell_reason = f"{trade.buy_tag}->{sell_reason}"

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
                btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
          btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        inf_heikinashi = qtpylib.heikinashi(informative)
        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
        return informative

    def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        if length == 0:
            return ((df['open'] - df['close']) / df['close'])
        else:
            return ((df['open'].rolling(length).max() - df['close']) / df['close'])


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe


    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe
        dataframe['long_term_price_warning'] = np.where(
            (
                # 15% drop in 8h
                    (
                            dataframe['close'].rolling(96).max() * 0.85 > dataframe['close'].rolling(6).mean()
                    ) |
                    # 20% drop in 12h
                    (
                            dataframe['close'].rolling(144).max() * 0.8 > dataframe['close'].rolling(8).mean()
                    ) |
                    # 25% drop in 24h
                    (
                            dataframe['close'].rolling(288).max() * 0.75 > dataframe['close'].rolling(12).mean()
                    ) |
                    # 30% drop in 36h
                    (
                            dataframe['close'].rolling(432).max() * 0.7 > dataframe['close'].rolling(24).mean()
                    ) |
                    # over 35% increase in the past 24h and 36h
                    (
                            (df24h['close'].rolling(24).mean() * 1.42 < dataframe['close'].rolling(18).mean()) &
                            (df36h['close'].rolling(24).mean() * 1.42 < dataframe['close'].rolling(18).mean())
                    )
            ), 1, 0)

        for period in self.timeperiods:
            dataframe[f'rsi_{period}'] = normalize(ta.RSI(dataframe, timeperiod=period), 0, 100)
            dataframe[f'linangle_{period}'] = normalize(ta.LINEARREG_ANGLE(dataframe, timeperiod=period), -35, 35)

        # MACD
        dataframe['close_min'] = dataframe['close'].rolling(window=960 // self.timeframe_minutes).min()
        dataframe['close_max'] = dataframe['close'].rolling(window=960 // self.timeframe_minutes).max()
        dataframe['relative_price'] = normalize(dataframe['close'], dataframe['close_min'], dataframe['close_max'])

        #macd = ta.MACD(dataframe)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], fastperiod=21, slowperiod=45, signalperiod=16)
        dataframe['macdmin'] = dataframe['macd'].rolling(window=360 // self.timeframe_minutes).min()
        dataframe['macdmax'] = dataframe['macd'].rolling(window=360 // self.timeframe_minutes).max()
        dataframe['macd_norm'] = np.where(dataframe['macdmin'] == dataframe['macdmax'], 0, (2.0*(dataframe['macd']-dataframe['macdmin'])/(dataframe['macdmax']-dataframe['macdmin'])-1.0))

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=45 // self.timeframe_minutes)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=60 // self.timeframe_minutes)
        dataframe['uptrend'] = (dataframe['ema_fast'] > dataframe['ema_slow']).astype('int')
        dataframe['trendline'] = dataframe['linangle_72'] / dataframe['linangle_72'].shift(12)

        # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        dataframe = HA(dataframe, 4)

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        # # ClucHA
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['hema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_df = self.dp.get_pair_dataframe(pair=btc_info_pair, timeframe=self.timeframe)
        dataframe['btc_rsi'] = normalize(ta.RSI(btc_df, timeperiod=14), 0, 100)
        dataframe['btc_ema_45'] = ta.TEMA(btc_df, timeperiod=45 // self.timeframe_minutes)
        dataframe['btc_ema_60'] = ta.TEMA(btc_df, timeperiod=60 // self.timeframe_minutes)
        dataframe['btc_uptrend'] = (dataframe['btc_ema_45'] > dataframe['btc_ema_60']).astype('int')

        dataframe['zema_offset_buy'] = zema(dataframe, int(self.base_nb_candles_buy_zema.value)) *self.low_offset_zema.value
        dataframe['zema_offset_buy2'] = zema(dataframe, int(self.base_nb_candles_buy_zema2.value)) *self.low_offset_zema2.value
        dataframe['ema_sell'] = ta.EMA(dataframe, 5)
        dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value
        dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema2.value)) *self.low_offset_ema2.value

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []
        dont_buy_conditions.append(
            (
                # don't buy if there isn't some profit to be made
                (dataframe['close_1h'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dont_buy_conditions.append((dataframe['trendline'] < 0.995))
        dont_buy_conditions.append((dataframe['relative_price'] > 0.51))

        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < 35) &
                    (dataframe['rsi_36'] < 0.45) &
                    (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] > self.ewo_high.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < 35) &
                    (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                    (dataframe['EWO'] > self.ewo_high_2.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')


        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < 35) &
                    (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] < self.ewo_low.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        # This produces around 0.85% - 1.0% profitable trades but long durations so decided against it.
        # dataframe.loc[
        #     (
        #         (
        #             (
        #                 (dataframe['macd_norm'] < -0.9) &
        #                 (dataframe['linangle_17'] > 0.5)
        #             )
        #         ) &
        #         (dataframe['rsi_36'] < 0.55) &
        #         (dataframe['uptrend'] < 1)
        #     ),
        #     ['buy', 'buy_tag']] = (1, 'MacD')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        # This does not fit well the protections for EWO buys.
        dataframe.loc[
            (
                (dataframe['volume'] > 0) &
                (dataframe['rocr_1h'].gt(self.clucha_rocr_1h.value)) &
                (
                    (
                         (dataframe['lower'].shift().gt(0)) &
                         (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.clucha_bbdelta_close.value)) &
                         (dataframe['ha_closedelta'].gt(dataframe['ha_close'] * self.clucha_closedelta_close.value)) &
                         (dataframe['tail'].lt(dataframe['bbdelta'] * self.clucha_bbdelta_tail.value)) &
                         (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                         (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
                     ) |
                     (
                         (dataframe['ha_close'] < dataframe['hema_slow']) &
                         (dataframe['ha_close'] < self.clucha_close_bblower.value * dataframe['lower'])
                     )
                 )
            ),
            ['buy', 'buy_tag']] = (1, 'clucHA')

        # This does not fit well the protections for EWO buys.
        dataframe.loc[
            (
                (
                    ((dataframe['close'] < dataframe['zema_offset_buy']) & (dataframe['pm'] <= dataframe['pmax_thresh'])) |
                    ((dataframe['close'] < dataframe['zema_offset_buy2']) & (dataframe['pm'] > dataframe['pmax_thresh']))
                ) &
                (dataframe['volume'] > 0) &
                (dataframe['long_term_price_warning'] < 1) &
                (dataframe['trendline'] > 0.996) &
                (dataframe['btc_rsi'] > 0.49) &
                (dataframe['close'] < dataframe['Smooth_HA_L']) &
                (dataframe['close'] < (dataframe['ema_sell'] * self.high_offset_sell_ema.value)) &
                (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.10 )) &
                (dataframe['Smooth_HA_O'].shift(1) < dataframe['Smooth_HA_H'].shift(1)) &
                (dataframe['rsi_fast'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &
                (
                    (
                        (dataframe['close'] < dataframe['ema_offset_buy']) &
                        (dataframe['pm'] <= dataframe['pmax_thresh']) &
                        (
                            (dataframe['EWO'] < -19.632) |
                            (
                                (dataframe['EWO'] > 2.615) & (dataframe['rsi'] < 60)
                            )
                        )
                    ) |
                    (
                        (dataframe['close'] < dataframe['ema_offset_buy2']) &
                        (dataframe['pm'] > dataframe['pmax_thresh']) &
                        (
                            (dataframe['EWO'] < -19.955) |
                            (
                                (dataframe['EWO'] > 2.188) & (dataframe['rsi'] < 45)
                            )
                        )
                    )
                )
            ),
            ['buy', 'buy_tag']] = (1, 'zema')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['uptrend'] == 1) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            ) |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['uptrend'] == 1) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe


    def rollingNormalize(self, dataframe, name):
        df = dataframe.copy()
        df[name + '_nmin'] = df[name].rolling(window=1440 // self.timeframe_minutes).min()
        df[name + '_nmax'] = df[name].rolling(window=1440 // self.timeframe_minutes).max()
        return np.where(df[name + '_nmin'] == df[name + '_nmax'], 0, (2.0*(df[name]-df[name + '_nmin'])/(df[name + '_nmax']-df[name + '_nmin'])-1.0))


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)


def williams_r(dataframe: DataFrame, timeperiod: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=timeperiod).max()
    lowest_low = dataframe["low"].rolling(center=False, window=timeperiod).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{timeperiod} Williams %R",
        )

    return WR * -100


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
                basic_ub[i] < final_ub[i - 1]
                or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
                basic_lb[i] > final_lb[i - 1]
                or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                            and mavalue[i] <= final_ub[i])
            else final_lb[i] if (
                    pm_arr[i - 1] == final_ub[i - 1]
                    and mavalue[i] > final_ub[i]) else final_lb[i]
            if (pm_arr[i - 1] == final_lb[i - 1]
                and mavalue[i] >= final_lb[i]) else final_ub[i]
            if (pm_arr[i - 1] == final_lb[i - 1]
                and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)

    return df
