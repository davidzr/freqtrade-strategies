# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
# --------------------------------

"""
                   ,-'"-.
             __...| .".  |
        ,--+"     ' |   ,'
       | .'   ..--,  `-' `.
       |/    |  ,' |       :
       |\...-+-".._|       |
     ,"            `--.     `.     _..-'+"/__
    /   .              |      :,-"'     `" |_'
 ..| .    _,....___,'  |    ,'            /\
..\'.__.-'  /V     |   '                ,'""
`. |  `:  \.       |  .               ,'         ,.-.
  `:       |       |  '             .^.        ,' ,"`.
    `.     |       | /               _.\.---..'  /   |     ,-,.
      `._  A      / j              ."       /   /    |   .',' |
         `. `...-' ,'             /        /._ /     | ,' /   |
           |"-----'             ,'        /   /-.__  |'  /    |
           | _.--'"'""`.       .         /   /     `"^-.,     |
           |"       ____\     j             j            `"--.|
           |  _.-""'     \    |             |                j
         _,+."_           \   |             |                |
        '    . `.     _.-"'.     ,          |                '
       |_    | `.`. ,'      `.   |          |               .
       | `-. |  ,'.\         .\   \         |              /
       |\   ;+-'   "\      ,'  `.  \        |             /
       '\\."         \ _.-'     ,`. \       '            /
        \\\           :       .'   `.`._     \          / `-..-.
         ``.          |    _." _...,:.._`.    `._     ,'   -. \'
          `.`.        |`".'__.'           `,...__"--`/  |   / |
            `.`.     _'    \|             ,'       ,'_  `..'  |..__,.
              `._`--".'     \`._      _,-'       ,' `-'  /    | .  ,'
                 `""'        `. `"'""'   ,-" _,-'    _ .'     '  `' `.
                               `-.._____:  |"       _," ."  ,'__,.."'
                                         `.|-...,.<'    `,_""'`./
                                             `.'   `"--'" mh
SLOWBRO v100

"""


class Slowbro(IStrategy):

    minimal_roi = {
         "0": 0.10,
         "1440": 0.20,
         "2880": 0.30,
         "10080": 1.0
    }

    # Stoploss:
    stoploss = -0.99

    timeframe = '1h'
    inf_timeframe = '1d'

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30
    process_only_new_candles = False

    def informative_pairs(self):
        # add all whitelisted pairs on informative timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['30d-low'] = informative['close'].rolling(30).min()
        informative['30d-high'] = informative['close'].rolling(30).max()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'],dataframe[f"30d-low_{self.inf_timeframe}"])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'],dataframe[f"30d-high_{self.inf_timeframe}"])
            ),
            'sell'] = 1

        return dataframe
