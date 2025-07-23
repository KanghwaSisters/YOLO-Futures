import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from datahandler.dataset import *
from env.done_ftn import *
from env.reward_ftn import *
from env.account import *
from env.maturity_ftn import *

# 시장 상태 구분 Enum (강세장, 약세장, 횡보장)
class MarketRegime(Enum):
    BULL = 1
    BEAR = -1
    SIDEWAYS = 0

# 리스크 관련 지표 계산 클래스
class RiskMetrics:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.returns_history = []
        self.pnl_history = []
        
    def update(self, pnl: float, returns: float):
        # 최근 손익 및 수익률 기록 업데이트
        self.pnl_history.append(pnl)
        self.returns_history.append(returns)
        
        # lookback 기간 초과된 데이터 제거
        if len(self.pnl_history) > self.lookback_period:
            self.pnl_history.pop(0)
            self.returns_history.pop(0)
    
    def get_sharpe_ratio(self) -> float:
        # 샤프비율 계산 (평균수익/표준편차)
        if len(self.returns_history) < 2:
            return 0.0
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 표준편차가 0에 가까우면 0 반환 (division by zero 방지)
        if std_return < 1e-8:
            return 0.0
        
        sharpe = mean_return / std_return
        # 극값 제한
        return np.clip(sharpe, -10.0, 10.0)
    
    def get_max_drawdown(self) -> float:
        # 최대 낙폭 계산 (최대 누적손실)
        if len(self.pnl_history) < 2:
            return 0.0
        cumulative = np.cumsum(self.pnl_history)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (peak + 1e-8)
        return np.min(drawdown)
    
    def get_volatility(self) -> float:
        # 수익률 변동성 계산 (표준편차)
        if len(self.returns_history) < 2:
            return 0.0
        return np.std(self.returns_history)

# 선물 트레이딩 환경 클래스
class FuturesEnvironment:
    def __init__(self, 
                 full_df: pd.DataFrame, 
                 date_range: tuple, 
                 window_size: int, 
                 state_type, 
                 reward_ftn, 
                 done_ftn, 
                 start_budget: float,
                 n_actions, 
                 position_cap: float = float('inf'),
                 scaler=None,
                 # 추가 파라미터
                 transaction_cost: float = 0.0005,  # 거래 비용 비율
                 slippage_factor: float = 0.0001,   # 슬리피지 비율
                 margin_requirement: float = 0.1,   # 증거금 비율
                 max_drawdown_limit: float = 0.2,   # 최대 허용 손실 비율
                 intraday_only: bool = False,       # 당일 청산 여부
                 risk_lookback: int = 20):          # 리스크 계산 기간
        
        # 데이터프레임을 날짜 기준으로 슬라이싱하여 환경 데이터셋 생성
        self._full_df = full_df
        self._date_range = date_range
        self.n_actions = n_actions
        self.df = self._slice_by_date(full_df, date_range)
        
        self.scaler = scaler
        self.window_size = window_size
        
        self.dataset = FuturesDataset(self.df, window_size, self.scaler)
        self.data_iterator = iter(self.dataset)
        
        self.state = state_type
        self.state.get_dataset_indices(self.dataset.indices)
        self.next_state = None

        # 포지션 제한
        self.position_dict = {-1 : 'short', 0 : 'hold', 1 : 'long'}
        self.position_cap = position_cap   # 최대 계약 수 : 상한 
        self.single_execution_cap = self.n_actions // 2

        # 시장 정보 
        self.previous_price = None      # 현재 시장 가격
        self.contract_unit = 50000      # 거래 단위가 1포인트 당 5만원 (미니 선물)
        self.current_timestep = date_range[0]   # 현재 타임스텝 추적

        # 만기일 리스트
        self.maturity_list = calculate_maturity(self.df.index)

        # 계좌
        self.account = Account(start_budget, position_cap, self.current_timestep, transaction_cost, slippage_factor)

        # 현재 타임스텝 추적
        self.current_timestep = date_range[0]
        
        # ===== 기존 코드 호환성을 위한 속성 추가 =====
        # current info 
        # -[ type of info ]-------------------------------------
        # '' : done=False, 'margin_call' : 마진콜, 
        # 'end_of_data' : 마지막 데이터, 'bankrupt' : 도부, 
        # 'maturity_date' : 만기일, 'max_contract' : 최대 계약수 도달 
        # ------------------------------------------------------
        self.info = ''      
        self.mask = [1] *  self.n_actions      # shape [n_actions] with 1 (valid) or 0 (invalid)

        # penalty 
        self.hold_over_penalty = -0.05
        self.margin_call_penalty = -1.0
        # ==============================================
        
        # 추가 기능 관련 변수 초기화
        self.transaction_cost = transaction_cost
        self.slippage_factor = slippage_factor
        self.margin_requirement = margin_requirement
        self.max_drawdown_limit = max_drawdown_limit
        self.intraday_only = intraday_only
        
        # 리스크 계산용 객체
        self.risk_metrics = RiskMetrics(risk_lookback)
        # self.total_trades = 0
        self.winning_trades = 0
        self.total_transaction_costs = 0
        
        # 시장 상태 초기값
        self.market_features = {
            'market_regime': 0,
            'volatility_regime': {'low': -1, 'normal': 0, 'high': 1}['low'],
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'win_rate': 0,
            'total_trades': 0,
            'transaction_cost_ratio': 0
        }
        self.market_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = 'normal'  # low, normal, high
        
        # 외부에서 주입된 함수
        self.sign = lambda x: (x > 0) - (x < 0)
        self.get_reward = reward_ftn
        self.get_done = done_ftn

        # 성과 추적용 리스트
        self.daily_returns = []
        self.trade_history = []
        
    def get_mask(self):
        def restrict_actions_by_position():
            if self.account.current_position == -1: # short 
                mask = [0] * self.single_execution_cap + [1] * (self.single_execution_cap+1)
            elif self.account.current_position == 1: # long 
                mask = [1] * (self.single_execution_cap+1) + [0] * self.single_execution_cap 
            else:
                mask = [1] * self.n_actions  
            return mask

        # 가용 계약 수 
        remaining_strength = self.position_cap - self.account.execution_strength

        if self.position_cap == remaining_strength:
            # 최대 체결 가능 계약수에 도달했을 때 
            mask = restrict_actions_by_position()

        elif self.info == 'insufficient':
            # 자본금 부족으로 새로운 포지션을 체결할 수 없을 때 
            mask = restrict_actions_by_position()

        elif (remaining_strength) < self.single_execution_cap:
            # 최대 체결 가능 계약수에 근접하여 일부 행동에 제약이 있다. 
            restricted_action = self.single_execution_cap - remaining_strength 

            if self.account.current_position == -1: # short 
                mask = [0] * restricted_action + [1] * (self.n_actions - restricted_action)
            elif self.account.current_position == 1: # long 
                mask = [1] * (self.n_actions - restricted_action) + [0] * restricted_action

        else:
            mask = [1] *  self.n_actions

        if len(mask) != 9:  # 예: 3이 정상이라고 가정
            print(f"❗️[Warning] n_actions changed: {mask}, current timestep: {self.current_timestep}, info: {self.info}")
            print(remaining_strength)
            print(self.position_cap)
            print(self.single_execution_cap)
            print(self.n_actions)
            import traceback; traceback.print_stack()

        return mask
    
    def _slice_by_date(self, full_df, date_range):
        full_df = full_df.copy()
        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()
        
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return full_df[(full_df.index >= start) & (full_df.index <= end)]
    
    def _update_market_regime(self, price_data: np.ndarray):
        """가격 데이터를 바탕으로 시장 상태(강세, 약세, 횡보) 및 변동성 상태 갱신"""
        if len(price_data) < 20:
            return
        
        short_ma = np.mean(price_data[-5:])
        long_ma = np.mean(price_data[-20:])
        
        if short_ma > long_ma * 1.02:
            self.market_regime = MarketRegime.BULL
        elif short_ma < long_ma * 0.98:
            self.market_regime = MarketRegime.BEAR
        else:
            self.market_regime = MarketRegime.SIDEWAYS
        
        volatility = np.std(price_data[-10:]) / np.mean(price_data[-10:])
        if volatility > 0.02:
            self.volatility_regime = 'high'
        elif volatility < 0.01:
            self.volatility_regime = 'low'
        else:
            self.volatility_regime = 'normal'
    
    def _force_liquidate_all_positions(self):
        """리스크 제한 초과 시 모든 포지션 강제 청산"""
        if self.account.execution_strength == 0:
            return

        reversed_execution = -self.account.execution_strength * self.account.current_position
        net_pnl, cost = self.account.step(reversed_execution, self.previous_price, self.current_timestep)

        # 거래 내역 기록
        self.trade_history.append({
                'timestamp': self.current_timestep,
                'action': reversed_execution,
                'price': self.previous_price,
                'pnl': net_pnl,
                'cost': cost,
                'type': 'forced_liquidation'
            })
    
    def _get_market_features(self) -> Dict[str, float]:
        """현재 시장 상태 관련 주요 지표 반환"""
        return {
            'market_regime': self.market_regime.value,
            'volatility_regime': {'low': -1, 'normal': 0, 'high': 1}[self.volatility_regime],
            'sharpe_ratio': self.risk_metrics.get_sharpe_ratio(),
            'max_drawdown': self.risk_metrics.get_max_drawdown(),
            'volatility': self.risk_metrics.get_volatility(),
            'win_rate': self.winning_trades / max(self.account.total_trades, 1),
            'total_trades': self.account.total_trades,
            'transaction_cost_ratio': self.total_transaction_costs / self.account.initial_budget
        }

    def _is_dataset_reached_end(self, current_timestep):
        done = self.dataset.reach_end(current_timestep)
        info = 'end_of_data' if done else ''
        return done, info 

    def _is_maturity_data(self, next_timestep, current_timestep):
        # 만기
        is_maturity_date = self.current_timestep.date() in self.maturity_list
        day_changed = is_day_changed(next_timestep=next_timestep,
                                     current_timestep=current_timestep)
        
        done = is_maturity_date & day_changed

        info = 'maturity_data' if done else ''
        return done, info
    
    def _is_bankrupt(self):
        done = self.account.available_balance <= 0
        info = 'bankrupt' if done else ''
        return done, info
    
    def _check_near_margin_call(self):
        # 현재 딱 마진콜 기준 (7%)
        if (self.account.available_balance <= self.account.maintenance_margin):
            self.info = 'margin_call' 

    def _check_insufficient(self):
        # 새로운 계약을 체결할 수 없는 경우의 조건
        # 일 뿐 done=True가 아니다 
        if (self.account.available_balance <= self.previous_price * self.account.initial_margin_rate):
            self.info = 'insufficient'

    def _is_risk_limits(self):
        """최대 손실 한도, 최대 드로우다운 초과 여부 확인"""
        total_return = (self.account.available_balance + self.account.unrealized_pnl) / self.account.initial_budget - 1
        if total_return < -self.max_drawdown_limit:
            return True, 'risk_limits' 
        
        max_dd = self.risk_metrics.get_max_drawdown()
        if max_dd < -self.max_drawdown_limit:
            return True, 'risk_limits'
        
        return False, ''
    
    def step(self, action: int):
        """
        환경 한 스텝 진행
        1) 거래 비용 및 슬리피지 계산
        2) 포지션 및 평균 진입가 업데이트
        3) 실현 및 미실현 손익 계산
        4) 보상, 종료 여부 계산
        5) 강제 청산 처리 (필요시)
        6) 상태 및 기록 업데이트 후 반환
        """

        # 다음 상태 데이터, 종가, 타임스텝 받아오기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)
        current_price = close_price

        # 행동에 따른 계좌 업데이트
        net_realized_pnl, cost = self.account.step(action, current_price, next_timestep)

        if net_realized_pnl > 0:
            self.winning_trades += 1

        self.trade_history.append({
                'timestamp': self.current_timestep,
                'action': action,
                'price': current_price,
                'pnl': net_realized_pnl,
                'cost': cost,
                'type': 'regular'
            })

        # 7. 일일 수익률 계산 및 리스크 메트릭 업데이트
        # ==================== 디버깅 필요
        # (지민) 강제 청산 이후에 저장해야 할 것 같은데 이후 계산하는 거 좀 전체적으로 반영해야 할 듯 
        # ===================================

        daily_return = net_realized_pnl / self.account.initial_budget
        self.daily_returns.append(daily_return)
        self.risk_metrics.update(net_realized_pnl, daily_return)

        # 8. 시장 상태 업데이트 (필요시 주석 해제)
        # current_idx = self.df.index.get_loc(self.current_timestep)
        # start_idx = max(0, current_idx - self.window_size)
        # price_data = self.df['close'].iloc[start_idx:current_idx].values
        # self._update_market_regime(price_data)

        # info를 확인하기 
        self._check_insufficient()
        self._check_near_margin_call()

        # done, info를 동시에 확인하기 
        done, self.info = self.switch_done_info(next_timestep, self.current_timestep)
        
        # info를 확인하고 강제 청산 옵션 실행 
        if self.info in ['margin_call', 'maturity_data', 'bankrupt']:
            self._force_liquidate_all_positions()

        # 9. 다음 상태 생성 (여기에 시장 정보 포함)
        # market_features = self._get_market_features()
        next_state = self.state(
            next_fixed_state,  # 실제 데이터 기반 상태
            current_position=self.account.current_position,
            execution_strength=self.account.execution_strength,
            realized_pnl=self.account.realized_pnl,
            unrealized_pnl=self.account.unrealized_pnl,
            maintenance_margin=self.account.maintenance_margin,
            total_transaction_costs=self.account.total_transaction_costs
        )

        # 10. 보상 계산 
        reward = self.get_reward(
            unrealized_pnl=self.account.unrealized_pnl,
            prev_unrealized_pnl=self.account.prev_unrealized_pnl,
            current_budget=self.account.available_balance,
            transaction_cost=cost,
            risk_metrics=self.risk_metrics,  # Sharpe ratio를 위해 RiskMetrics 객체 전달
            market_regime=self.market_regime,
            daily_return=daily_return,
            net_realized_pnl=net_realized_pnl
        )

        # 12. action space에 대한 마스크 생성 
        self.mask = self.get_mask()

        # 
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        # 13. 선물 데이터에서 추가 done 상황 + update info 

        # # 14. 종료되면 남은 포지션 강제 청산
        # if done and self.execution_strength > 0:
        #     self._force_liquidate_all_positions()

        # 16. 다음 상태, 보상, 종료 플래그 반환
        return next_state, reward, done
    
    def switch_done_info(self, next_timestep, current_timestep):
        done = self.get_done(
            current_timestep=self.current_timestep,
            next_timestep=next_timestep,
            max_strength=self.position_cap,
            current_strength=self.account.execution_strength,
            intraday_only=self.intraday_only
        )
        if done:
            return done, 'done'
        # dataset end check
        done, info = self._is_dataset_reached_end(next_timestep)
        if done:
            return done, info 

        # maturity date check
        done, info = self._is_maturity_data(next_timestep, current_timestep)
        if done:
            return done, info 

        # bankruptcy check
        done, info = self._is_bankrupt()
        if done:
            return done, info 
        
        # 
        # done, info = self._is_risk_limits()
        # if done:
        #     return done, info
        
        return False, ''

    
    def get_performance_summary(self) -> Dict[str, Any]:
        """현재까지의 주요 성과 지표 요약 반환"""
        total_return = (self.account.available_balance + self.account.unrealized_pnl) / self.account.initial_budget - 1
        
        return {
            'total_return': total_return,
            'total_trades': self.account.total_trades,
            'win_rate': self.winning_trades / max(self.account.total_trades, 1),
            'sharpe_ratio': self.risk_metrics.get_sharpe_ratio(),
            'max_drawdown': self.risk_metrics.get_max_drawdown(),
            'total_transaction_costs': self.total_transaction_costs,
            'cost_ratio': self.total_transaction_costs / self.account.initial_budget,
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime,
            'current_budget': self.account.available_balance,
            'unrealized_pnl': self.account.unrealized_pnl
        }
    
    def reset(self):
        """환경 초기화 및 상태 리셋"""
        self.account.reset()
        self.total_transaction_costs = 0

        # info 상태 초기화
        self.info = ''
        
        self.risk_metrics = RiskMetrics(20)
        self.winning_trades = 0
        
        self.trade_history = []
        self.daily_returns = []
        
        self.market_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = 'normal'
        
        self.data_iterator = iter(self.dataset)
        fixed_state, close_price, timestep = next(self.data_iterator)
        
        self.mask = [1] *  self.n_actions
        self.previous_price = close_price
        self.current_timestep = timestep
        
        return self.state(
                fixed_state,  # 실제 데이터 기반 상태
                current_position=self.account.current_position,
                execution_strength=self.account.execution_strength,
                realized_pnl=self.account.realized_pnl,
                unrealized_pnl=self.account.unrealized_pnl,
                maintenance_margin=self.account.maintenance_margin,
                total_transaction_costs=self.account.total_transaction_costs
            )
    
    def conti(self):
        """done 후에도 다음 상태를 반환 (연속 거래용)"""
        return self.next_state
    
    def render(self, state, action, next_state):
        """기존 코드 호환성을 위한 render 메서드"""
        close_idx = self.dataset.indices.index('close')
        # memory : 제대로 예측이 되는지 보여줄 수 있는 지표여야 한다. 
        pass 
    
    def __str__(self):
        """환경 상태 및 주요 성과 출력용 문자열 생성"""
        perf = self.get_performance_summary()
        return (
            f"=== Improved Futures Trading Environment ===\n"
            f"⏱️  Current Timestep   : {self.current_timestep}\n"
            f"📈  Previous Close     : {self.previous_price:.2f}\n"
            f"💼  Current Position   : {self.position_dict[self.account.current_position]} ({self.account.current_position})\n"
            f"📊  Execution Strength : {self.account.execution_strength}/{self.position_cap}\n"
            f"📉  Unrealized PnL     : {self.account.unrealized_pnl:.2f} KRW\n"
            f"💰  Current Budget     : {self.account.available_balance:.2f} KRW\n"
            f"💵  Total Return       : {perf['total_return']*100:.2f}%\n"
            f"⚖️  Avg Entry Price    : {self.account.average_entry:.2f}\n"
            f"🎯  Win Rate          : {perf['win_rate']*100:.1f}%\n"
            f"📊  Sharpe Ratio      : {perf['sharpe_ratio']:.3f}\n"
            f"📉  Max Drawdown      : {perf['max_drawdown']*100:.1f}%\n"
            f"💸  Transaction Costs : {self.account.total_transaction_costs:.2f} KRW\n"
            f"🌍  Market Regime     : {self.market_regime.name}\n"
            f"📈  Volatility Regime : {self.volatility_regime}\n"
            f"🔢  Total Trades      : {self.account.total_trades}\n"
            f"ℹ️  Info Status       : {self.info}\n"
            f"===============================================\n"
        )