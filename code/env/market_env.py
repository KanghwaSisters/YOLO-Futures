import pandas as pd

from env.done_ftn import *
from env.reward_ftn import *
from datahandler.dataset import *
from account import *

class MarketEnvironment:
    def __init__(self, full_df:pd.DataFrame, date_range:tuple, window_size:int, state_type, reward_ftn, done_ftn, start_budget, position_cap=None, scaler=None):
        ## inner infomation 
        self._full_df = full_df
        self._date_range = date_range

        # set sliced df 
        self.df = self._slice_by_date(full_df, date_range)

        # dataset, data iterator 
        self.scaler = scaler
        self.dataset = FuturesDataset(self.df, window_size, self.scaler)
        self.data_iterator = iter(self.dataset)

        # set state frame 
        self.state = state_type
        self.state.get_dataset_indices(self.dataset.indices)
        self.next_state = None 

        # 시장 정보 
        self.previous_price = None

        # current info 
        self.current_timestep = date_range[0]

        # 계좌
        self.account = Account(start_budget, self.current_timestep, position_cap, self.current_timestep)

        # penalty 
        self.hold_over_penalty = -0.05
        self.margin_call_penalty = -1.0

        # useful ftn 
        self.sign = lambda x: (x > 0) - (x < 0)
        self.get_reward = reward_ftn      # reward를 계산하는 함수 
        self.get_done = done_ftn

    def _slice_by_date(self, full_df, date_range):
        full_df = full_df.copy()
        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()

        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return full_df[(full_df.index >= start) & (full_df.index <= end)]


    def step(self, action: int):
        # 1. 다음 상태 먼저 받기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)

        # daily settlement (하루 장 마감 시 당일 청산)
        if self.current_timestep.date != next_timestep.date:
            self.account.daily_settlement(self.previous_price)

        # 2. 현재 가격 업데이트
        current_price = close_price

        # 3,4. 포지션, 계좌 업데이트
        self.account.step(action, close_price, next_timestep)

        # 5. State, reward, done
        next_state = self.state(next_fixed_state, 
                                current_position=self.account.current_position, 
                                execution_strength=self.account.execution_strength)
        
        reward = self.get_reward(unrealized_pnl=self.account.unrealized_pnl, 
                                 prev_unrealized_pnl=self.account.prev_unrealized_pnl,
                                 current_budget=self.account.available_balance) # 현재 예산 = 가용잔고
        
        done = self.get_done(current_timestep=self.current_timestep,
                             next_timestep=next_timestep, 
                             max_strength=self.account.position_cap, 
                             current_strength=self.account.execution_strength)
        
        # if self.position_cap is not None:
        #     if self.execution_strength > self.position_cap:
        #         done = True 

        # 6. Update
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        return next_state, reward, done
    
    def reset(self): 
        # 진짜 0스텝부터 돌아가는 매서드 
        self.data_iterator = iter(self.dataset) # init iterator 

        fixed_state, close_price, timestep = next(self.data_iterator)

        self.previous_price = close_price
        self.current_timestep = timestep

        self.account.reset()

        return self.state(fixed_state, 
                          current_position=self.account.current_position, 
                          execution_strength=self.account.execution_strength)
    
    def conti(self):
        # done 이후부터 돌아가는 코드 (일반적인 reset의 기능과 유사하다.)
        return self.next_state
    
    def render(self, state, action, next_state):
        close_idx = self.dataset.indices.index('close')
        # memory : 제대로 예측이 되는지 보여줄 수 있는 지표여야 한다. 
        pass 

    def __str__(self):
        return (
            f"=== Futures Trading Environment ===\n"
            f"⏱️  Current Timestep   : {self.current_timestep}\n"
            f"📈  Previous Close     : {self.previous_price:.2f}\n"
            f"💼  Current Position   : {self.position_dict[self.current_position]} ({self.current_position})\n"
            f"📊  Execution Strength : {self.execution_strength}/{self.position_cap}\n"
            f"📉  Unrealized PnL     : {self.unrealized_pnl:.2f} KRW\n"
            f"💰  Current Budget     : {self.current_budget:.2f} KRW\n"
            f"💵  Rate of Return     : {self.current_budget / self.init_budget * 100:.2f} %\n"
            f"⚖️  Avg Entry Price    : {self.average_entry:.2f}\n"
            f"==================================\n"
        )
    