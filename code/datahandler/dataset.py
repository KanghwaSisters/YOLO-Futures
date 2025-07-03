import pandas as pd
from torch.utils.data import Dataset
from datetime import timedelta

from indicator_ftns import *

class FuturesDataset(Dataset):
    def __init__(self, df, window_size):
        # df 
        grouped_df = self._make_group(df)
        total_df = self._add_technical_indicators(grouped_df)
        cleaned_df = self._remove_Nan(total_df)
        self.states, self.close_prices, self.timesteps = self._split_dataset(cleaned_df, window_size)

        # inner info 
        self.window_size = window_size

        # outer info 
        self.indices = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.close_prices[idx], self.timesteps[idx] # (total_fixed_state, close_price, timestep)
    
    def _remove_Nan(self, df):
        return df[~df.isnull().any(axis=1)]
    
    def _make_group(self, df):
        grouped_df = df.copy()
        # 연속 여부 계산 
        grouped_df['continues'] = (grouped_df.index + timedelta(minutes=1)).isin(grouped_df.index)

        # 연속이 끊기기 직전 idx가 기준이 되게 breakpoint 지정
        breakpoint = ~grouped_df['continues'].shift(fill_value=True)

        # 누적합을 기준으로 그룹핑 
        grouped_df['group_id'] = breakpoint.cumsum()

        # 장 마감 시간을 제외한 나머지 데이터만 이용 
        selected_df = grouped_df[grouped_df['continues']]

        return selected_df.drop(['date', 'time','continues'], axis=1)


    def _split_dataset(self, df, window_size):

        df = df.drop(['prevClose'], axis=1)
        self.indices = df.columns.to_list()

        states = []
        close_prices = []
        timesteps = []

        for _, group in df.groupby([df.index.date, 'group_id']):
            
            if len(group) >= window_size:
                total_iteration = len(group) - window_size + 1

            for i in range(total_iteration):
                state = np.array(group.iloc[i:i+window_size])
                close = group['close'].iloc[i + window_size - 1]
                time = group.index[i + window_size - 1]

                states.append(state)
                close_prices.append(close)
                timesteps.append(time)

        return states, close_prices, timesteps

    def _add_technical_indicators(self, df):
        result = []
        for _, group in df.groupby([df.index.date, 'group_id']):
            group = group.copy()
            group = add_basic_indicators(group)
            group = add_trend_indicators(group)
            group = add_momentum_indicators(group)
            group = add_volume_indicators(group)
            group = add_volatility_indicators(group)
            result.append(group)

        return pd.concat(result)
        