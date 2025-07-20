import pandas as pd

from env.done_ftn import *
from env.reward_ftn import *
from datahandler.dataset import *

class FuturesEnvironment:
    def __init__(self, full_df:pd.DataFrame, date_range:tuple, window_size:int, state_type, reward_ftn, done_ftn, start_budget, n_actions, position_cap=float('inf'), scaler=None):
        ## inner infomation 
        self._full_df = full_df
        self._date_range = date_range
        self.n_actions = n_actions

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

        # 체결 계약 수, 현재 포지션 정보 
        self.current_position = 0     
        self.position_dict = {-1 : 'short', 0 : 'hold', 1 : 'long'}
        self.execution_strength = 0        # 체결 계약 수 
        self.position_cap = position_cap   # 최대 계약 수 : 상한 
        self.single_execution_cap = self.n_actions // 2

        # 시장 정보 
        self.previous_price = None
        self.prev_unrealized_pnl = 0
        # self.maintenance_margin = 
        self.average_entry = 0
        self.contract_unit = 50000         # 거래 단위가 1포인트 당 5만원 (미니 선물)

        # current info 
        # -[ type of info ]-------------------------------------
        # '' : done=False, 'margin_call' : 마진콜, 
        # 'end_of_data' : 마지막 데이터, 'bankrupt' : 도부, 
        # 'maturity_date' : 만기일, 'max_contract' : 최대 계약수 도달 
        # ------------------------------------------------------
        self.info = ''      
        self.mask = [1] *  self.n_actions      # shape [n_actions] with 1 (valid) or 0 (invalid)
        self.init_budget = start_budget
        self.current_budget = start_budget
        self.unrealized_pnl = 0
        self.current_timestep = date_range[0]

        # penalty 
        self.hold_over_penalty = -0.05
        self.margin_call_penalty = -1.0

        # useful ftn 
        self.sign = lambda x: (x > 0) - (x < 0)
        self.get_reward = reward_ftn      # reward를 계산하는 함수 
        self.get_done = done_ftn

    def get_mask(self):
        remaining_strength = self.position_cap - self.execution_strength

        if self.info == 'max_contract':
            if self.current_position == -1: # short 
                mask = [0] * self.single_execution_cap + [1] * (self.single_execution_cap+1)
            elif self.current_position == 1: # long 
                mask = [1] * (self.single_execution_cap+1) + [0] * self.single_execution_cap 

        elif (remaining_strength) < self.single_execution_cap:
            # 최대 체결 가능 계약수에 근접하여 일부 행동에 제약이 있다. 
            restricted_action = self.single_execution_cap - remaining_strength 

            if self.current_position == -1: # short 
                mask = [0] * restricted_action + [1] * (self.n_actions - restricted_action)
            elif self.current_position == 1: # long 
                mask = [1] * (self.n_actions - restricted_action) + [0] * restricted_action

        else:
            mask = [1] *  self.n_actions

        return mask

    def _slice_by_date(self, full_df, date_range):
        full_df = full_df.copy()
        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()

        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return full_df[(full_df.index >= start) & (full_df.index <= end)]

    def _cal_ave_entry_price(self, current_price, prev_execution, new_execution, action):
        # 평균 진입가 업데이트 (새 포지션이 생길 경우)
        remaining_execution = new_execution

        if remaining_execution != 0:
            if self.sign(new_execution) == self.sign(action):
                # 평균 가격 갱신
                prev_value = self.average_entry * abs(prev_execution)
                new_value = current_price * abs(action)
                total_value = prev_value + new_value
                total_contracts = abs(prev_execution + action)
                self.average_entry = total_value / total_contracts

        else:
            self.average_entry = 0

    def _get_realized_pnl(self, current_price, prev_execution, action):
        if self.previous_price is None:
            return 0  # 첫 스텝이면 계산 불가
        
        # 청산된 계약 수
        liquidation = min(abs(prev_execution), abs(action)) if self.sign(prev_execution) != self.sign(action) else 0

        # 실현 손익 계산
        realized_pnl = 0
        if liquidation > 0:
            price_diff = (current_price - self.average_entry) * self.sign(prev_execution)
            realized_pnl = price_diff * liquidation * self.contract_unit

        return realized_pnl
    
    def _get_unrealized_pnl(self):
        # 미실현 손익(Unrealized Profit/Loss)

        if self.previous_price is None or self.execution_strength == 0:
            return 0.0  # 포지션이 없거나 가격 정보가 없을 경우

        price_diff = self.previous_price - self.average_entry
        direction = self.current_position  # +1 for long, -1 for short
        contracts = self.execution_strength

        unrealized_pnl = price_diff * direction * contracts * self.contract_unit
        return unrealized_pnl
    
    def get_current_position_strength(self, action):
        # 이전 체결 데이터를 가져와 현재 행동을 반영한다. 
        previous_execution = self.current_position * self.execution_strength

        # 행동 이후의 체결 계약 수와 포지션을 업데이트한다. 
        current_execution = previous_execution + action

        if current_execution == 0:
            return 0, 0
        
        execution_strength = abs(current_execution)
        current_position = self.sign(current_execution)

        return current_position, execution_strength
    
    def _force_liquidate_all_positions(self):
        # 현재 정보 
        execution = self.current_position * self.execution_strength
        current_price = self.previous_price

        pnl = self._get_realized_pnl(current_price, execution, -execution)
        self.current_budget += pnl 

    def step(self, action: int):
        # 1. 다음 상태 먼저 받기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)

        # 2. 현재 가격 업데이트
        current_price = close_price

        # 3. 포지션 관련 업데이트
        prev_execution = self.current_position * self.execution_strength
        new_execution = prev_execution + action

        realized_pnl = self._get_realized_pnl(self.previous_price, prev_execution, action)
        self._cal_ave_entry_price(self.previous_price, prev_execution, new_execution, action)

        self.current_position, self.execution_strength = self.get_current_position_strength(action)

        # 4. PnL 및 Budget
        self.prev_unrealized_pnl = self.unrealized_pnl
        self.unrealized_pnl = self._get_unrealized_pnl()
        self.current_budget += realized_pnl

        # 5. State, reward, done, mask 
        next_state = self.state(next_fixed_state, 
                                current_position=self.current_position, 
                                execution_strength=self.execution_strength)
        
        reward = self.get_reward(unrealized_pnl=self.unrealized_pnl, 
                                 prev_unrealized_pnl=self.prev_unrealized_pnl,
                                 current_budget=self.current_budget)
        
        done = self.get_done(current_timestep=self.current_timestep, next_timestep=next_timestep, 
                             max_strength=self.position_cap, current_strength=self.execution_strength)
        
        self.mask = self.get_mask()

        # 6. Update
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        # 7. Handle done 
        done = True if self.is_dataset_reached_end(self.current_timestep) else False # 마지막 데이터에 대한 종료 
        done = True if self.is_near_margin_call() else False
        done = True if self.is_maturity_date() else False
        done = True if self.is_bankrupt() else False

        return next_state, reward, done
    
    def is_dataset_reached_end(self, current_timestep):
        flag = self.dataset.reach_end(current_timestep)
        self.info = 'end_of_data' if flag is True else ''
        return flag

    def is_near_margin_call(self):
        flag = False
        self.info = 'margin_call' if flag is True else ''
        return flag 
    
    def is_maturity_date(self):
        flag = False 
        self.info = 'maturity_date' if flag is True else ''
        return flag 
    
    def is_bankrupt(self):
        flag = False 
        self.info = 'bankrupt' if flag is True else ''
        return flag 

    def reset(self): 
        # 진짜 0스텝부터 돌아가는 매서드 
        self.data_iterator = iter(self.dataset) # init iterator 

        self.current_position = 0          # [-1 : long, 0 : hold, 1 : short] 
        self.execution_strength = 0        # 체결 계약 수 
        self.average_entry = 0

        fixed_state, close_price, timestep = next(self.data_iterator)

        self.previous_price = close_price
        self.current_timestep = timestep
        self.mask = [1] *  self.n_actions

        return self.state(fixed_state, 
                          current_position=self.current_position, 
                          execution_strength=self.execution_strength)
    
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
            f"💰  Current Budget     : {self.current_budget:.2f}/{self.init_budget} KRW\n"
            f"💵  Rate of Return     : {self.current_budget / self.init_budget * 100:.2f} %\n"
            f"⚖️  Avg Entry Price    : {self.average_entry:.2f}\n"
            f"==================================\n"
        )
    