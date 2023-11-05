# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series  # noqa

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime


class Uptrend(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        'buy_rsi_uplimit': 50,

    }

    buy_rsi_uplimit = IntParameter(50, 90, default=buy_params['buy_rsi_uplimit'], optimize=False, space='buy')

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.1,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.015
        elif (current_profit > 0.015):
            sl_new = 0.0075

        return sl_new


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)

        dataframe['mama_diff'] = dataframe['mama'] - dataframe['fama']
        dataframe['mama_diff_ratio'] = dataframe['mama_diff'] / dataframe['hl2']

        dataframe['zero'] = 0

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # EMA 50
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)

        # EMA 200
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=200)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 80) &
                (dataframe['mama'] >  dataframe['fama']) & # uptrend
                (dataframe['mama_diff_ratio'] > 0.04) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['mama_diff_ratio'] < 0.01) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe


import random
from functools import reduce

class SuperBuy(Uptrend):
    """
	Idea is to build random buy signales from populate_indicators, with luck we'll get a good buy signal
	"""

    generator = IntParameter(0, 100000000000, default=99295874569, optimize=True, space='buy')  # generate unique matrix of conditions for your dataframe
    operators_used_to_compare_between_columns = IntParameter(0, 3, default=3, optimize=True, space='buy')  # number of conditions you will keep to build buy signal
    operators_used_to_with_best_point = IntParameter(0, 3, default=1, optimize=True, space='buy')  # number of conditions you will keep to build buy signal
    condition_selector = IntParameter(0, 100, default=50, optimize=True, space='buy')  # how to select the desired conditions beteween all conditions generated (seed random)

    best_buy_point = None
    best_buy_point_dict = dict()
    bad_buy_point_dict = dict()
    all_points_dict = dict()
    buy_signal_already_printed = False

    columns = []
    columns_to_compare_to_best_point = []
    columns_to_compare_to_volume = []
    columns_to_compare_to_price = []

    operators = {
        0: '<',
        1: '>',
        2: '<=',
        3: '>=',
        4: '==',
        5: '!='
    }

    top_index_criteria = {

        # best point criteria
        'min_close_hh_ratio': 0.08,
        'max_candles_to_get_ratio': 8,
        'candles_after_dip_to_buy': 0,

        # parameters selection criteria
        'select_parameter_if_in_more_than_x_percent_of_best_points': 97,
        'select_prameter_if_prop_is_x_percent_higher_in_best_points': 10,
    }

    def find_best_entry_point(self, dataframe: DataFrame, metadata: dict):
        lookahead_candles = self.top_index_criteria['max_candles_to_get_ratio']

        workdataframe = dataframe.copy()
        workdataframe['higher_high'] = workdataframe['high'].rolling(lookahead_candles).max()
        workdataframe['close_shifted_lookehead'] = workdataframe['close'].shift(lookahead_candles)
        workdataframe['higher_high_close_ratio'] = workdataframe['higher_high'] / workdataframe['close_shifted_lookehead']

        df_mask = workdataframe['higher_high_close_ratio'] >= 1 + self.top_index_criteria['min_close_hh_ratio']
        # print(1 + self.top_index_criteria['min_close_hh_ratio'])
        filtered_df = workdataframe[df_mask]

        filtered_df = filtered_df.sort_values(by=["higher_high_close_ratio"], ascending=False)
        filtered_df["shifted_index"] = filtered_df.index - lookahead_candles + self.top_index_criteria['candles_after_dip_to_buy']

        if filtered_df.empty:
            if self.config['runmode'].value != 'hyperopt':
                print("No entry point found for {}".format(metadata['pair']))
            return filtered_df, workdataframe
        # print(metadata['pair'])
        # print(filtered_df[["date", "shifted_index", "higher_high_close_ratio", "close_shifted_lookehead", "close", "higher_high"]])
        return filtered_df, workdataframe.drop(filtered_df['shifted_index'])

    def common_points_for_every_best_entry(self, dataframe: DataFrame, metadata: dict, columns: list) -> list:
        full_pairlist = self.dp.current_whitelist()
        current_pair = metadata['pair']

        if current_pair not in self.best_buy_point_dict:
            self.best_buy_point_dict[current_pair], self.bad_buy_point_dict[current_pair] = self.find_best_entry_point(dataframe, metadata)
            self.all_points_dict[current_pair] = dataframe.copy()

        for pair in full_pairlist:
            current_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if (pair not in self.best_buy_point_dict) and not current_df.empty:
                # print("No entry point found for {}".format(pair))
                return []

        all_best_points = None
        all_bad_points = None
        all_points = None
        for pair in full_pairlist:
            # NO DATA FOR THIS PAIR
            if not pair in self.best_buy_point_dict:
                continue
            if all_best_points is None:
                all_best_points = self.best_buy_point_dict[pair]
                all_bad_points = self.bad_buy_point_dict[pair]
                all_points = self.all_points_dict[pair]
            else:
                all_best_points = all_best_points.append(self.best_buy_point_dict[pair])
                all_bad_points = all_bad_points.append(self.bad_buy_point_dict[pair])
                all_points = all_points.append(self.all_points_dict[pair])

        print("HERE COMMON VALUES FOR ALL BEST POINTS !!!!!!!!!!")
        res = list()
        best_indicators = []
        for column in columns:
            all_points_values_count = all_points[column].value_counts()
            all_bad_points_values_count = all_bad_points[column].value_counts()
            count = all_best_points[column].value_counts()
            # keep only values in more than x% of all points
            df_mask = count >= 1 / 100 * all_best_points.shape[0]
            count = count[df_mask]

            if not count.empty and not column in ['buy', 'buy_tag']:
                count_normalized = count / all_best_points.shape[0]
                all_bad_points_values_count_normalized = all_bad_points_values_count / all_bad_points.shape[0]
                df_all = count.to_frame(name='best_points').join(all_bad_points_values_count.to_frame(name='bad_points'))
                df_all['best_point_percent'] = count_normalized
                df_all['bad_point_percent'] = all_bad_points_values_count_normalized
                df_all['part_of_best_points'] = 100 * df_all['best_points'] / (df_all['best_points'] + df_all['bad_points'])
                df_all['part_of_best_points_percent'] = count_normalized / all_bad_points_values_count_normalized
                print(column)
                print(df_all)
                values = df_all.query(
                    f"part_of_best_points > 3 & "  # the part of best points should be at least 3% for the value
                    "part_of_best_points_percent > 13 & "  # proportion of value is X* more important in best points than in bad points
                    "best_points > 12"  # minimum number of the value (because we don't want close=1.121213243482902183 as result)
                ).index.tolist()  # & df_all["part_of_best_points_percent"] > 10)]
                print(values)
                for elt in values:
                    best_indicators.append(f"dataframe['{column}'] == {elt}")

            if column in ['buy', 'buy_tag']:
                print(column)
                print(all_best_points[column].value_counts())
            elif not count.empty:
                res.append({
                    'column': column,
                    'value': count.index[0],
                    'ratio_for_best': count.iloc[0] / all_best_points.shape[0],
                    'ratio_for_all': all_points_values_count[count.index[0]] / all_points.shape[0],
                    'ratio_diff': count.iloc[0] / all_best_points.shape[0] - all_points_values_count[count.index[0]] / all_points.shape[0]
                })
        for item in sorted(res, key=lambda x: x['ratio_diff']):
            print(f"({item['column']} == {item['value']}), {100 * item['ratio_for_best']:.2f}% in best vs {100 * item['ratio_for_all']:.2f}% average")
        print("END OF COMMON VALUES FOR ALL BEST POINTS !!!!!!!!!!")

        print("Suggested buy signal:")
        print("( # main buy signals found")
        for item in best_indicators:
            print(f"    ({item})|")
        print(")")
        print("& ( # protections")
        for item in sorted(res, key=lambda x: x['ratio_diff']):
            if (((100 * item['ratio_for_best']) - (100 * item['ratio_for_all'])) > self.top_index_criteria['select_prameter_if_prop_is_x_percent_higher_in_best_points']) and (100 * item['ratio_for_best']) > self.top_index_criteria['select_parameter_if_in_more_than_x_percent_of_best_points']:
                print(f"(dataframe['{item['column']}'] == {item['value']}) &")
        print(")")
        return []

    def is_same_dimension_as_price(self, dataframe: DataFrame, column_name: str) -> bool:
        if dataframe['close'].dtype != dataframe[column_name].dtype:
            # prevent impossible comparisons
            return False
        return (dataframe[column_name].max() <= dataframe['high'].max() and dataframe[column_name].min() >= dataframe['low'].min())

    def is_same_dimension_as_volume(self, dataframe: DataFrame, column_name: str) -> bool:
        if dataframe['volume'].dtype != dataframe[column_name].dtype:
            # prevent impossible comparisons
            return False
        if 'volume' in column_name:
            return True
        return False

    def generate_superbuy_signal(self, dataframe: DataFrame, metadata: dict) -> list:
        # every indicators names
        columns = list(dataframe.columns)
        columns.remove('date')
        columns.remove('sell')
        columns.remove('buy')
        columns.remove('buy_tag')
        columns = [column for column in columns if not 'date' in column]

        # generated random conditions
        buy_conds = []

        # operators we will use as a string "123423232" which will be used to sequentially pick in operators
        generators = ""
        base_generators = str(self.generator.value)
        while len(generators) < len(columns) * len(columns):
            generators = generators + base_generators

        # get best buy point for first pair, will indicators will be used for each pair
        # THE PAIR YOU WANT TO USE AS REFERENCE MUST BE FIRST IN YOUR PAIRLIST !!!!!!!!
        if self.best_buy_point is None:
            try:
                top_index, _ = self.find_best_entry_point(dataframe, metadata)["shifted_index"].iloc[0]
                self.best_buy_point = dataframe.iloc[top_index]
                print(f"pair used as reference is {metadata['pair']}")
                print(top_index)
            except:
                self.best_buy_point = None
                pass

        # sort columns by category
        if len(self.columns_to_compare_to_best_point) == 0 and len(self.columns_to_compare_to_volume) == 0 and len(self.columns_to_compare_to_price) == 0:
            self.columns = columns
            for column in columns:
                if self.is_same_dimension_as_price(dataframe, column):
                    self.columns_to_compare_to_price.append(column)
                elif self.is_same_dimension_as_volume(dataframe, column):
                    self.columns_to_compare_to_volume.append(column)
                else:
                    self.columns_to_compare_to_best_point.append(column)
            print(f"columns_to_compare_to_price : {self.columns_to_compare_to_price}")
            print(f"columns_to_compare_to_volume : {self.columns_to_compare_to_volume}")

            # remove NAN columns for best point...
            if self.best_buy_point is not None:
                for column in self.columns_to_compare_to_best_point:
                    if str(self.best_buy_point[column]) == 'nan':
                        self.columns_to_compare_to_best_point.remove(column)
            print(f"columns_to_compare_to_best_point : {self.columns_to_compare_to_best_point}")

        # generate matrix of all operators for all combinations of columns and create buy conditions
        index = 0
        for left_elt in self.columns_to_compare_to_price:
            for right_elt in self.columns_to_compare_to_price:
                if index > len(generators):
                    break
                generator = generators[index]
                index += 1
                if left_elt == right_elt:
                    continue
                if int(generator) not in self.operators:
                    # pass if no operator is selected
                    continue
                # print("(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])")
                buy_conds.append(
                    "(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])"
                )

        for left_elt in self.columns_to_compare_to_volume:
            for right_elt in self.columns_to_compare_to_volume:
                if index > len(generators):
                    break
                generator = generators[index]
                index += 1
                if left_elt == right_elt:
                    continue
                if int(generator) not in self.operators:
                    # pass if no operator is selected
                    continue
                # print("(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])")
                buy_conds.append(
                    "(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])"
                )

        buy_conds_best_point = []
        # generate buy conditions with best buy point
        for column in self.columns_to_compare_to_best_point:
            if self.best_buy_point is None:
                continue
            if index > len(generators):
                break
            generator = generators[index]
            index += 1
            if int(generator) not in self.operators:
                # pass if no operator is selected
                continue
            # print("(dataframe['" + column + "'] " + self.operators[int(generator)] + " best_buy_point['" + column + "']))")
            # print(eval("best_buy_point['" + column + "']"))
            buy_conds_best_point.append(
                "(dataframe['" + column + "'] " + self.operators[int(generator)] + " " + str(self.best_buy_point[column]) + ")"
            )

        # select a few buy conditions
        random.seed(self.condition_selector.value)
        try:
            buy_conds = random.sample(buy_conds, self.operators_used_to_compare_between_columns.value)
        except ValueError:
            print("not enough conditions to compare between columns")
            # Sample larger than population or is negative
            pass
        try:
            buy_conds += random.sample(buy_conds_best_point, self.operators_used_to_with_best_point.value)
        except ValueError as e:
            if self.config['runmode'].value != 'hyperopt':
                print("not enough conditions to compare with best point")
            # Sample larger than population or is negative
            pass
        if self.config['runmode'].value in ('backtest', 'hyperopt') and self.buy_signal_already_printed != buy_conds:
            print(buy_conds)
            self.buy_signal_already_printed = buy_conds
        try:
            buy_conds = [eval(buy_cond, globals(), {'dataframe': dataframe, 'best_buy_point': self.best_buy_point}) for buy_cond in buy_conds]
        except:
            return []
        return buy_conds

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_conds = self.generate_superbuy_signal(dataframe, metadata)

        if self.config['runmode'].value in ('backtest'):  # backtest, we want to check must common buy tags...
            dataframe = super().populate_buy_trend(dataframe, metadata)  # get buy tags
            self.common_points_for_every_best_entry(dataframe, metadata, self.columns + ['buy', 'buy_tag'])
        elif self.config['runmode'].value in ('hyperopt'):  # hyperopt, we want to test new buy signals
            is_additional_check = (
                    (  # main buy signals found
                        (dataframe['not_res1_1h'] == True)
                    )
                    & (  # protections
                            (dataframe['rsi_fast_lower_20'] == 0) &
                            (dataframe['rsi_fast_lower_30'] == 0) &
                            (dataframe['r_14_lower_minus_80'] == 0) &
                            (dataframe['r_32_lower_minus_80'] == 0) &
                            (dataframe['r_96_lower_minus_80'] == 0)
                    ) &
                    (
                        # (dataframe['ema_50_lin'] < dataframe['ema_26_lin']) |
                        # (dataframe['sma_15'] > dataframe['ema_50_1h']) |
                        # (dataframe['ema_slow'] <= dataframe['sup1']) |
                        (dataframe['res1'] >= dataframe['bb_upperband2_1h'])
                    )
            )

            if buy_conds:
                dataframe.loc[
                    is_additional_check
                    &
                    reduce(lambda x, y: x & y, buy_conds)

                    , 'buy'] = 1
        # THIS STRAT SHOULD NOT BE USED IN LIVE/DRYRUN MODE
        return dataframe
