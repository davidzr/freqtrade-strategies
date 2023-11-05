# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import logging
import pandas_ta as pta

from pandas import DataFrame, Series
from datetime import datetime, timezone
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)



class UziChan2(IStrategy):

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.10
    timeframe = '1m'

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):     
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)        
        if current_profit*100 > 1: 
            logger.info(f'custom sell for {pair} at {current_rate}')
            return 'sell_1.2pc'
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        dataframe['perc'] = ((dataframe['high'] - dataframe['low']) / dataframe['low']*100)
        dataframe['avg3_perc'] = ta.EMA(dataframe['perc'], 3)
        dataframe['perc_norm'] = (dataframe['perc'] - dataframe['perc'].rolling(50).min())/(dataframe['perc'].rolling(50).max()-dataframe['perc'].rolling(50).min())

        # Uzirox's channel prezzo
        periodo = 15
        dataframe['uc_mid']  = pta.ssf(dataframe['close'],5)
        dataframe['uc_stdv'] = ta.STDDEV(dataframe['uc_mid'], periodo).round(5)
        dataframe['uc_low']  = ta.EMA(dataframe['uc_mid'] - dataframe['uc_stdv'],3).round(5)
        dataframe['uc_up']   = ta.EMA(dataframe['uc_mid'] + dataframe['uc_stdv'],3).round(5)            

        dataframe['co']      = ta.ADOSC(dataframe,fastperiod = 30, slowperiod = 100).round(3)


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe['close'] < dataframe['uc_low']) | (dataframe['open'] < dataframe['uc_low'])) &
                (dataframe['co'] > dataframe['co'].shift())
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['high'] > dataframe['uc_up']) &
                (dataframe['co'] > dataframe['co'].shift())
            ),
            'sell'] = 1
        return dataframe
     

class UziChanTB2(UziChan2):

    process_only_new_candles = True

    custom_info_trail_buy = dict()
    custom_info_trail_sell = dict()    

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_sell_order_enabled = True    
    #trailing_expire_seconds = 1800      #NOTE 5m timeframe
    trailing_expire_seconds = 1800/5    #NOTE 1m timeframe
    #trailing_expire_seconds = 1800*3    #NOTE 15m timeframe

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = True
    trailing_sell_uptrend_enabled = True    
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    trailing_sell_max_stop = 0.02   # stop trailing sell if current_price < starting_price * (1+trailing_buy_max_stop)
    trailing_sell_max_sell = 0.000  # sell if price between downlimit (=max of serie (current_price * (1 + trailing_sell_offset())) and (start_price * 1+trailing_sell_max_sell))

    abort_trailing_when_sell_signal_triggered = True


    init_trailing_buy_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,  
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    init_trailing_sell_dict = {
        'trailing_sell_order_started': False,
        'trailing_sell_order_downlimit': 0,        
        'start_trailing_sell_price': 0,
        'sell_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_sell_trailing': False,
    }    

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_buy_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_sell(self, pair, reinit=False):
        # returns trailing sell info for pair (init if necessary)
        if not pair in self.custom_info_trail_sell:
            self.custom_info_trail_sell[pair] = dict()
        if (reinit or not 'trailing_sell' in self.custom_info_trail_sell[pair]):
            self.custom_info_trail_sell[pair]['trailing_sell'] = self.init_trailing_sell_dict.copy()
        return self.custom_info_trail_sell[pair]['trailing_sell']


    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_buy_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")



    def trailing_sell_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_sell = self.trailing_sell(pair)

        duration = 0
        try:
            duration = (current_time - trailing_sell['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info("'\033[36m'SELL: "
                f"pair: {pair} : "
                f"start: {trailing_sell['start_trailing_sell_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"downlimit: {trailing_sell['trailing_sell_order_downlimit']:.4f}, "
                f"profit: {self.current_trailing_sell_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_sell['offset']}")

    def current_trailing_buy_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def current_trailing_sell_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_sell = self.trailing_sell(pair)
        if trailing_sell['trailing_sell_order_started']:
            return (current_price - trailing_sell['start_trailing_sell_price'])/ trailing_sell['start_trailing_sell_price']
            #return 0-((trailing_sell['start_trailing_sell_price'] - current_price) / trailing_sell['start_trailing_sell_price'])
        else:
            return 0


    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_buy_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt  = (last_candle['perc_norm']).round(5)
        default_offset = 0.0045 * (1 + adapt)        #NOTE: default_offset 0.0045 <--> 0.009
        

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    def trailing_sell_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_sell_profit_ratio = self.current_trailing_sell_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt  = (last_candle['perc_norm']).round(5)
        default_offset = 0.003 * (1 + adapt)        #NOTE: default_offset 0.003 <--> 0.006
        
        trailing_sell  = self.trailing_sell(pair)
        if not trailing_sell['trailing_sell_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration =  current_time - trailing_sell['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_sell_profit_ratio > 0) and (last_candle['sell'] != 0)):
                # more than 1h, price over first signal, sell signal still active -> sell
                return 'forcesell'
            else:
                # wait for next signal
                return None
        elif (self.trailing_sell_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_sell_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is falling, sell 
            return 'forcesell'

        if current_trailing_sell_profit_ratio > 0:
            # current price is lower than initial price
            return default_offset

        trailing_sell_offset = {
            # 0.06: 0.02,
            # 0.03: 0.01,
            0.1: default_offset,
        }

        for key in trailing_sell_offset:
            if current_trailing_sell_profit_ratio < key:
                return trailing_sell_offset[key]

        return default_offset

    # end of trailing sell parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])   
        self.trailing_sell(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
            val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
            
            if val:
                if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                    val = False
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if(len(dataframe) >= 1):
                        last_candle = dataframe.iloc[-1].squeeze()
                        current_price = rate
                        trailing_buy = self.trailing_buy(pair)
                        trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                        if trailing_buy['allow_trailing']:
                            if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                                # start trailing buy
                                
                                trailing_buy['trailing_buy_order_started'] = True
                                trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                                trailing_buy['start_trailing_price'] = last_candle['close']
                                trailing_buy['buy_tag'] = last_candle['buy_tag']
                                trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                                trailing_buy['offset'] = 0
                                
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                            elif trailing_buy['trailing_buy_order_started']:
                                if trailing_buy_offset == 'forcebuy':
                                    # buy in custom conditions
                                    val = True
                                    ratio = "%.2f" % ((self.current_trailing_buy_profit_ratio(pair, current_price)) * 100)
                                    self.trailing_buy_info(pair, current_price)
                                    logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                                elif trailing_buy_offset is None:
                                    # stop trailing buy custom conditions
                                    self.trailing_buy(pair, reinit=True)
                                    logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                                elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                    # update uplimit
                                    old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                    self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                    self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                    self.trailing_buy_info(pair, current_price)
                                    logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                                elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                    # buy ! current price > uplimit && lower thant starting price
                                    val = True
                                    ratio = "%.2f" % ((self.current_trailing_buy_profit_ratio(pair, current_price)) * 100)
                                    self.trailing_buy_info(pair, current_price)
                                    logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                                elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                    # stop trailing buy because price is too high
                                    self.trailing_buy(pair, reinit=True)
                                    self.trailing_buy_info(pair, current_price)
                                    logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                                else:
                                    # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                    self.trailing_buy_info(pair, current_price)
                                    logger.info(f'price too high for {pair} !')

                        else:
                            logger.info(f"Wait for next buy signal for {pair}")

                    if (val == True):
                        self.trailing_buy_info(pair, rate)
                        self.trailing_buy(pair, reinit=True)
                        logger.info(f'STOP trailing buy for {pair} because I buy it')
            
            return val


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)        
        
        if val:
            if self.trailing_sell_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_sell= self.trailing_sell(pair)
                    trailing_sell_offset = self.trailing_sell_offset(dataframe, pair, current_price)

                    if trailing_sell['allow_sell_trailing']:
                        if (not trailing_sell['trailing_sell_order_started'] and (last_candle['sell'] != 0)):
                            trailing_sell['trailing_sell_order_started'] = True
                            trailing_sell['trailing_sell_order_downlimit'] = last_candle['close']
                            trailing_sell['start_trailing_sell_price'] = last_candle['close']
                            trailing_sell['sell_tag'] = last_candle['sell_tag']
                            trailing_sell['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_sell['offset'] = 0
                            
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f'start trailing sell for {pair} at {last_candle["close"]}')

                        elif trailing_sell['trailing_sell_order_started']:
                            if trailing_sell_offset == 'forcesell':
                                # sell in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_sell_profit_ratio(pair, current_price)) * 100)
                                self.trailing_sell_info(pair, current_price)
                                logger.info(f"FORCESELL for {pair} ({ratio} %, {current_price})")

                            elif trailing_sell_offset is None:
                                # stop trailing sell custom conditions
                                self.trailing_sell(pair, reinit=True)
                                logger.info(f'STOP trailing sell for {pair} because "trailing sell offset" returned None')

                            elif current_price > trailing_sell['trailing_sell_order_downlimit']:
                                # update downlimit
                                old_downlimit = trailing_sell["trailing_sell_order_downlimit"]
                                self.custom_info_trail_sell[pair]['trailing_sell']['trailing_sell_order_downlimit'] = max(current_price * (1 - trailing_sell_offset), self.custom_info_trail_sell[pair]['trailing_sell']['trailing_sell_order_downlimit'])
                                self.custom_info_trail_sell[pair]['trailing_sell']['offset'] = trailing_sell_offset
                                self.trailing_sell_info(pair, current_price)
                                logger.info(f'update trailing sell for {pair} at {old_downlimit} -> {self.custom_info_trail_sell[pair]["trailing_sell"]["trailing_sell_order_downlimit"]}')

                            elif current_price > (trailing_sell['start_trailing_sell_price'] * (1 - self.trailing_sell_max_sell)):
                                # sell! current price < downlimit && higher than starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_sell_profit_ratio(pair, current_price)) * 100)
                                self.trailing_sell_info(pair, current_price)
                                logger.info(f"current price ({current_price}) < downlimit ({trailing_sell['trailing_sell_order_downlimit']}) but higher than starting price ({(trailing_sell['start_trailing_sell_price'] * (1 + self.trailing_sell_max_sell))}). OK for {pair} ({ratio} %)")

                            elif current_price < (trailing_sell['start_trailing_sell_price'] * (1 - self.trailing_sell_max_stop)):
                                # stop trailing, sell fast, price too low
                                val = True                                
                                self.trailing_sell_info(pair, current_price)
                                logger.info(f'STOP trailing sell for {pair} because of the price is much lower than starting price * {1 + self.trailing_sell_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_sell_info(pair, current_price)
                                logger.info(f'price too low for {pair} !')

                    else:
                        logger.info(f"Wait for next sell signal for {pair}")

                if (val == True):
                    self.trailing_sell_info(pair, rate)
                    self.trailing_sell(pair, reinit=True)
                    logger.info(f'STOP trailing sell for {pair} because I SOLD it')

        if sell_reason != 'sell_signal':
            val = True

        return val


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'): 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"                        
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_sell_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.abort_trailing_when_sell_signal_triggered and self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-1].squeeze()
            if (last_candle['sell'] != 0):
                trailing_buy = self.trailing_buy(metadata['pair'])
                if trailing_buy['trailing_buy_order_started']:
                    logger.info(f"Sell signal for {metadata['pair']} is triggered!!! Abort trailing")
                    self.trailing_buy(metadata['pair'], reinit=True)        

        if self.trailing_sell_order_enabled and self.config['runmode'].value in ('live', 'dry_run'): 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_sell = self.trailing_sell(metadata['pair'])
            if (last_candle['sell'] != 0):
                if not trailing_sell['trailing_sell_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    #if not open_trades: 
                    if open_trades:
                        logger.info(f"Set 'allow_SELL_trailing' to True for {metadata['pair']} to start *SELL* trailing")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_sell['allow_sell_trailing'] = True
                        initial_sell_tag = last_candle['sell_tag'] if 'sell_tag' in last_candle else 'sell signal'
                        dataframe.loc[:, 'sell_tag'] = f"{initial_sell_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_sell['trailing_sell_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger sell signal!")
                    dataframe.loc[:,'sell'] = 1
                    dataframe.loc[:, 'sell_tag'] = trailing_sell['sell_tag']

        return dataframe


    plot_config = {
      'main_plot':{
            'uc_up':{'color':'gray'},
            'uc_mid':{'color':'green'},
            'uc_low' :{'color':'gray'},        
            },
      'subplots': {                 
      }  
    }     