import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from datahandler.dataset import *
from env.done_ftn import *
from env.reward_ftn import *
from env.account import *
from env.maturity_ftn import *
from env.env_f import *

# GOTRandomEnv을 위한 것 
from env.EpisodeData import *


class GoalOrTimeoutEnv(FuturesEnvironment):
    def __init__(self, 
                 full_df: pd.DataFrame, 
                 date_range: tuple, 
                 window_size: int, 
                 state_type, 
                 reward_ftn, 
                 done_ftn, 
                 start_budget: float,
                 n_actions: int, 
                 position_cap: float = float('inf'),
                 scaler=None,
                 # 고유 속성 
                 max_step=3000,
                 net_pnl_ratio=0.05,
                 # 거래 비용 및 리스크 파라미터
                 transaction_cost: float = 0.0005,
                 slippage_factor: float = 0.0001,
                 margin_requirement: float = 0.1,
                 max_drawdown_limit: float = 0.2,
                 intraday_only: bool = False,
                 risk_lookback: int = 20):

        super().__init__(
                full_df, date_range,  window_size, state_type, 
                reward_ftn, done_ftn, start_budget, n_actions, position_cap,
                scaler, transaction_cost, slippage_factor, margin_requirement,
                max_drawdown_limit, intraday_only, risk_lookback)

        self.max_step = max_step
        self.net_pnl_ratio = net_pnl_ratio

        self.done_status_list = ['end_of_data', 'maturity_data', 
                                 'max_step', #'goal_profit',
                                 'bankrupt']
        self.liquidate_status_list = ['end_of_data', 'maturity_data', 
                                      'max_step', # 'goal_profit',
                                      'margin_call', 'bankrupt']
    
    def step(self, action: int) -> Tuple[Any, float, bool]:
        """환경 스텝 실행"""
        # 0. 초기화 
        self.account.net_realized_pnl = 0
        self.account.net_realized_pnl_without_cost = 0

        # 1. 다음 데이터 가져오기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)
        current_price = close_price
        
        # 2. 계좌 업데이트 (거래 실행)
        net_realized_pnl, cost = self.account.step(action, current_price, next_timestep)
        self.maintained_steps += 1 
        
        # ===========================================
        self._check_end_of_day(next_timestep, self.current_timestep)
        self._check_near_margin_call()
        # ===========================================
        
        # 6. 시장 상태 업데이트
        self._update_market_conditions()
        
        # 7. 계좌 상태 확인
        self._check_account_status()
        
        # 8. 종료 조건 확인
        done, self.info = self._check_termination_conditions(next_timestep)
        
        # 9. 강제 청산 처리
        forced_liquidation_pnl = 0.0
        if self.info in self.liquidate_status_list:
            forced_liquidation_pnl, _cost = self._force_liquidate_all_positions(current_price)
            
            # 강제 청산 후 자산 계산
            current_equity = self.account.available_balance + self.account.unrealized_pnl
            current_equity = max(current_equity, 1.0)
            
            daily_return = self.performance_tracker.update_equity(current_equity)
             
            # 최종 수익률 계산 및 업데이트
            if len(self.performance_tracker.equity_history) > 1:
                final_return = (current_equity - self.performance_tracker.equity_history[-2]) / \
                              max(self.performance_tracker.equity_history[-2], 1.0)
            else:
                final_return = 0.0
            
            self.risk_metrics.update(
                pnl=forced_liquidation_pnl,
                returns=final_return,
                current_equity=current_equity
            )
            
            net_realized_pnl += forced_liquidation_pnl
            cost+= _cost
            
        else:
            # ====확인요망!!!!!!!!!!!!!!!================
            current_equity = self.account.available_balance + self.account.unrealized_pnl
            current_equity = max(current_equity, 1.0)  # 음수 방지
            
            # 4. 성과 추적 업데이트
            daily_return = self.performance_tracker.update_equity(current_equity)
            
            self.risk_metrics.update(
                pnl=net_realized_pnl,
                returns=daily_return,
                current_equity=current_equity
            )
        
        # =======================================================
        self.performance_tracker.update_trade(
            action=action,
            net_pnl=net_realized_pnl,
            cost=cost,
            current_price=current_price,
            current_timestep=str(self.current_timestep),
            current_equity=current_equity,
            position=self.account.current_position,
            execution_strength=self.account.execution_strength
        )
        
        # 5. 리스크 메트릭 업데이트
        self.risk_metrics.update(
            pnl=net_realized_pnl,
            returns=daily_return,
            current_equity=current_equity
        )
        
        # 실제 거래 발생 시 거래 결과 업데이트
        if action != 0:
            if net_realized_pnl != 0:  # 실현손익 발생
                self.risk_metrics.update_trade_result(net_realized_pnl)
            else:  # 미실현손익 변화 평가
                prev_unrealized = getattr(self.account, 'prev_unrealized_pnl', 0)
                current_unrealized = self.account.unrealized_pnl
                unrealized_change = current_unrealized - prev_unrealized
                
                if self.account.current_position != 0 and abs(unrealized_change) > 1000:
                    self.risk_metrics.update_trade_result(unrealized_change)
        
        # =======================================================
        
        # 10. 에피소드 완료 처리
        if done:
            self.performance_tracker.complete_episode(current_equity)
        
        # 당일 마지막 거래에서 미실현 손익 >> 실현 손익으로 전환 
        if is_day_changed(next_timestep=next_timestep, current_timestep=self.current_timestep):
            self.account.daily_settlement(current_price)

        # 성과 지표 
        perf = self.get_performance_summary() # 'cost_ratio', 'market_regime', 'volatility_regime'
        
        # 11. 보상 계산
        reward = self.get_reward(
            unrealized_pnl=self.account.unrealized_pnl,
            prev_unrealized_pnl=self.account.prev_unrealized_pnl,
            current_budget=self.account.available_balance,
            transaction_cost=cost,
            risk_metrics=self.risk_metrics,
            market_regime=self.market_state_manager.market_regime,
            daily_return=daily_return,
            net_realized_pnl=net_realized_pnl,
            realized_pnl=net_realized_pnl,
            prev_position=self.account.prev_position,
            current_position=self.account.current_position,
            execution_strength=self.account.execution_strength,
            equity=self.account.available_balance,
            initial_budget=self.account.initial_budget
        )
        
        # 12. 다음 상태 생성
        next_state = self.state(
            next_fixed_state,
            current_position=self.account.current_position,
            execution_strength=self.account.execution_strength,
            n_days_before_ma=self._get_n_days_before_maturity(self.current_timestep),
            realized_pnl=self.account.realized_pnl / self.account.contract_unit ,                       # (pt)
            unrealized_pnl=self.account.unrealized_pnl / self.account.contract_unit ,                   # (pt)
            available_balance=self.account.available_balance / self.account.contract_unit,              # (pt)
            cost_ratio=perf['cost_ratio'],
            market_regime=perf['market_regime']
        )

        # 12. action space에 대한 마스크 생성 
        self.mask = self.get_mask()

        # 업데이트 
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        self.account.net_realized_pnl = net_realized_pnl
        self.account.net_realized_pnl_without_cost = net_realized_pnl + cost

        return next_state, reward, done

    def _check_end_of_day(self, next_timestep, current_timestep):
        if is_day_changed(next_timestep=next_timestep, current_timestep=self.current_timestep):
            self.info = 'end_of_day'

    def _over_pnl_ratio_threshold(self):
        return self.account.realized_pnl_ratio >= self.net_pnl_ratio

    def _is_maturity_data(self, next_timestep, current_timestep):
        is_maturity_date = self.current_timestep.date() in self.maturity_list
        day_changed = is_day_changed(next_timestep=next_timestep, current_timestep=current_timestep)
        return is_maturity_date and day_changed

    def _check_termination_conditions(self, next_timestep) -> Tuple[bool, str]:
        """종료 조건 확인"""
        # 01. 마지막 데이터인가? 
        if self.dataset.reach_end(next_timestep):
            return True, 'end_of_data'

        # 02. 최대 step 수에 도달했는가? 
        if reach_max_step(maintained_steps=self.maintained_steps, 
                          max_step=self.max_step):
            return True, 'max_step'

        # 03. 기대 수익의 임계점을 넘었는가? 
        if self._over_pnl_ratio_threshold():
            return False, 'goal_profit'
        
        # 04. 파산상황인가? 
        if self.account.available_balance <= 0:
            return True, 'bankrupt'
        
        # 05. 만기일인가? 
        if self._is_maturity_data(next_timestep, self.current_timestep):
            return True, 'maturity_data'
        
        # 06. etc
        return False, ''

class GOTRandomEnv(GoalOrTimeoutEnv):
    def __init__(self, 
                 full_df: pd.DataFrame, 
                 date_range: tuple, 
                 window_size: int, 
                 state_type, 
                 reward_ftn, 
                 done_ftn, 
                 start_budget: float,
                 n_actions: int, 
                 position_cap: float = float('inf'),
                 scaler=None,
                 # 고유 속성 
                 max_step=3000,
                 net_pnl_ratio=0.05,
                 # 거래 비용 및 리스크 파라미터
                 transaction_cost: float = 0.0005,
                 slippage_factor: float = 0.0001,
                 margin_requirement: float = 0.1,
                 max_drawdown_limit: float = 0.2,
                 intraday_only: bool = False,
                 risk_lookback: int = 20):



        self.max_step = max_step
        self.net_pnl_ratio = net_pnl_ratio

        self.done_status_list = ['end_of_data', 'maturity_data', 
                                 'max_step', #'goal_profit',
                                 'bankrupt']
        self.liquidate_status_list = ['end_of_data', 'maturity_data', 
                                      'max_step', # 'goal_profit',
                                      'margin_call', 'bankrupt']
    

        # === 기본 환경 설정 ===
        self._full_df = full_df
        self._date_range = date_range
        self.n_actions = n_actions
        self.df = self._slice_by_date(full_df, date_range)
        self.max_step = max_step
        
        self.scaler = scaler
        self.window_size = window_size

        # define dataset 
        self.base_dataset = FuturesDataset(self.df, window_size, self.scaler)               # 해당 timestep의 전체 데이터셋
        self.episode_dataset = EpisodeDataset(self.base_dataset, window_len=max_step)   # 에피소드로 데이터셋을 묶은 애
        self.episode_loader = EpisodeDataloader(self.episode_dataset, shuffle=True)         # 그걸 섞고 관리하는 애 
        self.dataset = next(self.episode_loader)
        self.episode_iterator = iter(self.dataset)                                   # 콜하면 하나의 에피소드[MiniFuturesDataset]
        
        # 상태 관리
        self.state = state_type
        self.next_state = None
        
        # === 포지션 및 거래 설정 ===
        self.position_dict = {-1: 'short', 0: 'hold', 1: 'long'}
        self.position_cap = position_cap
        self.single_execution_cap = self.n_actions // 2
        
        # === 시장 정보 ===
        self.previous_price = None
        self.contract_unit = 50000  # 미니 선물 계약 단위
        self.current_timestep = date_range[0]
        self.start_budget = start_budget
        
        # 만기일 계산
        mask = self._full_df.index >= pd.to_datetime(self._date_range[0])
        dates = self._full_df.loc[mask].index.normalize().unique()

        self.maturity_list = calculate_maturity(dates)
        # print(self.maturity_list)
        self.maturity_iter = iter(self.maturity_list)
        self.latest_maturity_day = next(self.maturity_iter)
        
        # === 계좌 및 거래 비용 설정 ===
        self.account = Account(start_budget, position_cap, self.current_timestep, 
                              transaction_cost, slippage_factor)
        self.transaction_cost = transaction_cost
        self.slippage_factor = slippage_factor
        self.margin_requirement = margin_requirement
        self.max_drawdown_limit = max_drawdown_limit
        self.intraday_only = intraday_only
        
        # === 관리 객체 초기화 ===
        self.market_state_manager = MarketStateManager()
        self.performance_tracker = PerformanceTracker(self.start_budget)
        self.risk_metrics = RiskMetrics(risk_lookback)
        self.winning_trades = 0
        
        # === 환경 상태 변수 ===
        self.info = ''
        self.mask = np.ones(self.n_actions, dtype=np.int32).tolist()
        
        # === 페널티 설정 ===
        self.hold_over_penalty = -0.05
        self.margin_call_penalty = -1.0
        
        # === 외부 함수 ===
        self.sign = lambda x: (x > 0) - (x < 0)
        self.get_reward = reward_ftn
        self.get_done = done_ftn

        # 성과 추적용 리스트
        self.daily_returns = []
        self.trade_history = []
        self.maintained_steps = 0


    def reset(self):
        """환경 초기화"""
        # 1. 계좌 초기화
        self.account.reset()
        
        # 2. 관리 객체들 초기화
        self.performance_tracker.reset()
        self.risk_metrics.reset()
        
        # 3. 환경 상태 초기화
        self.info = ''
        self.maintained_steps = 0
        self.mask = [1] * self.n_actions
        
        # 4. 데이터 이터레이터 재설정
        try:
            # 다음 에피소드 하나 꺼내서 이터레이터 생성
            self.dataset = next(self.episode_loader)
            
        except StopIteration:
            # 에피소드 순회가 끝났다면 다시 섞어서 리셋
            self.episode_loader.shuffle_indices()
            self.episode_loader = iter(self.episode_loader)
            self.dataset = next(self.episode_loader)
        
        self.episode_iterator = iter(self.dataset)
        self.data_iterator = iter(self.episode_iterator)

        fixed_state, close_price, timestep = next(self.data_iterator)
        
        # 5. 초기 시장 정보 설정
        self.previous_price = close_price
        self.current_timestep = timestep
        
        # 6. 초기 자산 가치 기록
        initial_equity = self.account.available_balance + self.account.unrealized_pnl
        initial_equity = max(initial_equity, 1.0)
        
        # 성과 추적기에 초기 자산 설정
        self.performance_tracker.episode_start_equity = initial_equity
        self.performance_tracker.update_equity(initial_equity)
        
        # 리스크 메트릭에 초기 자산 설정
        self.risk_metrics.update(
            pnl=0.0,
            returns=0.0,
            current_equity=initial_equity
        )

        # 만기일 객체 초기화 
        self.maturity_iter = iter(self.maturity_list)
        self.latest_maturity_day = next(self.maturity_iter)
        
        # 7. 초기 상태 생성
        initial_state = self.state(
            fixed_state,
            current_position=self.account.current_position,
            execution_strength=self.account.execution_strength,
            n_days_before_ma=self._get_n_days_before_maturity(self.current_timestep),
            realized_pnl=self.account.realized_pnl / self.account.contract_unit ,                       # (pt)
            unrealized_pnl=self.account.unrealized_pnl / self.account.contract_unit ,                   # (pt)
            available_balance=self.account.available_balance / self.account.contract_unit,              # (pt)
            cost_ratio=0,
            market_regime=0
        )
        
        return initial_state
        
    def _check_termination_conditions(self, next_timestep) -> Tuple[bool, str]:
        """종료 조건 확인"""
        # 01. 최대 step 수에 도달했는가? 
        if reach_max_step(maintained_steps=self.maintained_steps, 
                          max_step=self.max_step):
            return True, 'max_step'

        # 02. 기대 수익의 임계점을 넘었는가? 
        if self._over_pnl_ratio_threshold():
            return False, 'goal_profit'
        
        # 03. 파산상황인가? 
        if self.account.available_balance <= 0:
            return True, 'bankrupt'
        
        # 04. 만기일인가? 
        if self._is_maturity_data(next_timestep, self.current_timestep):
            return True, 'maturity_data'
        
        # 05. etc
        return False, ''
