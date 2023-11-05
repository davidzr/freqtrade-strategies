import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class ClucHAwerk(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01021,
        'bbdelta-tail': 0.88118,
        'close-bblower': 0.0022,
        'closedelta-close': 0.00519,
        'rocr-1h': 0.50931,
        'volume': 35
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 1.01283, 
        'sell-rocr-1h': 0.95269
    }

    # ROI table:
    minimal_roi = {
        "0": 0.11054,
        "2": 0.05569,
        "10": 0.03055,
        "16": 0.02311,
        "82": 0.01267,
        "238": 0.00301,
        "480": 0
    }

    # Stoploss:
    stoploss = -0.02139

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.09291
    trailing_stop_positive_offset = 0.10651
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """
    
    timeframe = '1m'

    startup_candle_count: int = 168

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']


        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(dataframe['ha_close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        bollinger = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)
        
        inf_tf = '1h'
        
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((      
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                    dataframe['ha_close'].lt(dataframe['lower'].shift()) &
                    dataframe['ha_close'].le(dataframe['ha_close'].shift())
            ) |
            (       
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            dataframe['rocr_1h'].lt(params['sell-rocr-1h']) &
            ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe

class ClucHAwerk_ETH(ClucHAwerk):

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.02
    ignore_roi_if_buy_signal = True

    # hyperopt --config user_data/config-backtest-ETH.json --hyperopt ClucHAwerkHyperopt_ETH --hyperopt-loss SortinoHyperOptLoss --strategy ClucHAwerk_ETH -e 500 --spaces buy --timeframe 1m --timerange 20210101-
    # 483/500:   1880 trades. 1054/0/826 Wins/Draws/Losses. Avg profit   0.76%. Median profit   1.12%. Total profit  0.14286503 ETH ( 1426.37Σ%). Avg duration  45.3 min. Objective: -623.10584
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01566,
        'bbdelta-tail': 0.8478,
        'close-bblower': 0.00998,
        'closedelta-close': 0.00614,
        'rocr-1h': 0.61579,
        'volume': 27
    }
	
	# hyperopt --config user_data/config-backtest-ETH.json --hyperopt ClucHAwerkHyperopt_ETH --hyperopt-loss SortinoHyperOptLoss --strategy ClucHAwerk_ETH -e 500 --spaces sell --timeframe 1m --timerange 20210101-
	# 210/500:   1881 trades. 1055/0/826 Wins/Draws/Losses. Avg profit   0.76%. Median profit   1.12%. Total profit  0.14316974 ETH ( 1429.41Σ%). Avg duration  45.3 min. Objective: -624.58371
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 1.02894, 
		'sell-rocr-1h': 0.94925
    }
	
	# hyperopt --config user_data/config-backtest-ETH.json --hyperopt ClucHAwerkHyperopt_ETH --hyperopt-loss SharpeHyperOptLoss --strategy ClucHAwerk_ETH -e 250 --spaces roi --timeframe 1m --timerange 20210101-
    # 479/500:   1890 trades. 1209/0/681 Wins/Draws/Losses. Avg profit   0.86%. Median profit   0.74%. Total profit  0.16228997 ETH ( 1620.31Σ%). Avg duration  45.6 min. Objective: -258.91551
    # ROI table:
    minimal_roi = {
        "0": 0.14414,
        "13": 0.10123,
        "20": 0.03256,
        "47": 0.0177,
        "132": 0.01016,
        "177": 0.00328,
        "277": 0
    }
	
	# hyperopt --config user_data/config-backtest-ETH.json --hyperopt ClucHAwerkHyperopt_ETH --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_ETH -e 500 --spaces stoploss --timeframe 1m --timerange 20210101- 
    # 292/500:   1890 trades. 1209/0/681 Wins/Draws/Losses. Avg profit   0.86%. Median profit   0.74%. Total profit  0.16251312 ETH ( 1622.53Σ%). Avg duration  45.6 min. Objective: -4.40843
    # Stoploss:
    stoploss = -0.02
	
	# hyperopt --config user_data/config-backtest-ETH.json --hyperopt ClucHAwerkHyperopt_ETH --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_ETH -e 500 --spaces trailing --timeframe 1m --timerange 20210101- 
    # 443/500:   1883 trades. 1193/0/690 Wins/Draws/Losses. Avg profit   0.86%. Median profit   0.76%. Total profit  0.16275524 ETH ( 1624.95Σ%). Avg duration  46.0 min. Objective: -4.41651
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01018
    trailing_stop_positive_offset = 0.01203
    trailing_only_offset_is_reached = False

class ClucHAwerk_BTC(ClucHAwerk):

	# hyperopt --config user_data/config-backtest-BTC.json --hyperopt ClucHAwerkHyperopt_BTC --hyperopt-loss SortinoHyperOptLoss --strategy ClucHAwerk_BTC -e 500 --spaces buy --timeframe 1m --timerange 20210101-
	# 412/500:    572 trades. 505/17/50 Wins/Draws/Losses. Avg profit   1.80%. Median profit   2.27%. Total profit  0.01028714 BTC ( 1027.06Σ%). Avg duration 281.3 min. Objective: -152.09192
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01192,
        'bbdelta-tail': 0.96183,
        'close-bblower': 0.01212,
        'closedelta-close': 0.01039,
        'rocr-1h': 0.53422,
        'volume': 27
    }
	
	# hyperopt --config user_data/config-backtest-BTC.json --hyperopt ClucHAwerkHyperopt_BTC --hyperopt-loss SharpeHyperOptLoss --strategy ClucHAwerk_BTC -e 500 --spaces sell --timeframe 1m --timerange 20210101-
    # 284/500:    575 trades. 508/17/50 Wins/Draws/Losses. Avg profit   1.80%. Median profit   2.25%. Total profit  0.01036281 BTC ( 1034.62Σ%). Avg duration 279.4 min. Objective: -100.84858
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.98016, 
		'sell-rocr-1h': 0.86823
    }
	
	# hyperopt --config user_data/config-backtest-BTC.json --hyperopt ClucHAwerkHyperopt_BTC --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_BTC -e 500 --spaces roi --timeframe 1m --timerange 20210101- 
    # 304/500:    627 trades. 563/10/54 Wins/Draws/Losses. Avg profit   1.71%. Median profit   2.18%. Total profit  0.01075130 BTC ( 1073.41Σ%). Avg duration 248.0 min. Objective: -2.57804
    # ROI table:
    minimal_roi = {
        "0": 0.18105,
        "9": 0.10391,
        "49": 0.0447,
        "53": 0.02747,
        "141": 0.01265,
        "312": 0.00499,
        "466": 0
    }
	
	# hyperopt --config user_data/config-backtest-BTC.json --hyperopt ClucHAwerkHyperopt_BTC --hyperopt-loss SharpeHyperOptLoss --strategy ClucHAwerk_BTC -e 250 --spaces stoploss --timeframe 1m --timerange 20210101- 
    # 192/250:    568 trades. 505/17/46 Wins/Draws/Losses. Avg profit   1.80%. Median profit   2.18%. Total profit  0.01023698 BTC ( 1022.05Σ%). Avg duration 280.1 min. Objective: -99.33155
    # Stoploss:
    stoploss = -0.11356
	
	# hyperopt --config user_data/config-backtest-BTC.json --hyperopt ClucHAwerkHyperopt_BTC --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_BTC -e 500 --spaces trailing --timeframe 1m --timerange 20210101-
    # 313/500:    637 trades. 573/10/54 Wins/Draws/Losses. Avg profit   1.69%. Median profit   2.50%. Total profit  0.01079683 BTC ( 1077.96Σ%). Avg duration 242.7 min. Objective: -2.59319
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01008
    trailing_stop_positive_offset = 0.04585
    trailing_only_offset_is_reached = False

class ClucHAwerk_USD(ClucHAwerk):

	# hyperopt --config user_data/config-backtest-USD.json --hyperopt ClucHAwerkHyperopt_USD --hyperopt-loss SharpeHyperOptLoss --strategy ClucHAwerk_USD -e 500 --spaces buy --timeframe 1m --timerange 20210101-
    # 470/500:    680 trades. 631/27/22 Wins/Draws/Losses. Avg profit   2.91%. Median profit   2.93%. Total profit  991.61804628 USD ( 1980.07Σ%). Avg duration 184.2 min. Objective: -186.95550
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01806,
        'bbdelta-tail': 0.85912,
        'close-bblower': 0.01158,
        'closedelta-close': 0.01466,
        'rocr-1h': 0.51901,
        'volume': 26
    }
	
	# hyperopt --config user_data/config-backtest-USD.json --hyperopt ClucHAwerkHyperopt_USD --hyperopt-loss SortinoHyperOptLoss --strategy ClucHAwerk_USD -e 500 --spaces sell --timeframe 1m --timerange 20210101- 
    # 1/500:    679 trades. 630/27/22 Wins/Draws/Losses. Avg profit   2.90%. Median profit   2.93%. Total profit  986.25885773 USD ( 1969.37Σ%). Avg duration 184.6 min. Objective: -277.19845
    # Sell hyperspace params:
    sell_params = {
		'sell-bbmiddle-close': 1.06163, 
		'sell-rocr-1h': 0.63285
    }

	# hyperopt --config user_data/config-backtest-USD.json --hyperopt ClucHAwerkHyperopt_USD --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_USD -e 500 --spaces roi --timeframe 1m --timerange 20210101- 
    # 334/500:    715 trades. 674/22/19 Wins/Draws/Losses. Avg profit   2.90%. Median profit   2.80%. Total profit  1037.85838537 USD ( 2072.40Σ%). Avg duration 166.5 min. Objective: -5.90800
    # ROI table:
    minimal_roi = {
        "0": 0.19315,
        "13": 0.13189,
        "24": 0.08358,
        "103": 0.03894,
        "148": 0.0148,
        "201": 0.00506,
        "447": 0
    }

	# hyperopt --config user_data/config-backtest-USD.json --hyperopt ClucHAwerkHyperopt_USD --hyperopt-loss SharpeHyperOptLoss --strategy ClucHAwerk_USD -e 500 --spaces stoploss --timeframe 1m --timerange 20210101- 
    # 352/500:    729 trades. 688/22/19 Wins/Draws/Losses. Avg profit   2.91%. Median profit   2.79%. Total profit  1060.61902930 USD ( 2117.85Σ%). Avg duration 167.6 min. Objective: -198.19711
    # Stoploss:
    stoploss = -0.17725

	# hyperopt --config user_data/config-backtest-USD.json --hyperopt ClucHAwerkHyperopt_USD --hyperopt-loss OnlyProfitHyperOptLoss --strategy ClucHAwerk_USD -e 500 --spaces trailing --timeframe 1m --timerange 20210101- 
    # 366/500:    730 trades. 689/22/19 Wins/Draws/Losses. Avg profit   2.91%. Median profit   2.80%. Total profit  1062.06091250 USD ( 2120.73Σ%). Avg duration 167.3 min. Objective: -6.06910
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02946
    trailing_only_offset_is_reached = False