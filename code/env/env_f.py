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
        self.equity_curve = []  # 누적 자산 가치 추적
        self.initial_budget = None
        
        # 거래별 성과 추적 (개선)
        self.trade_pnls = []  # 실제 거래별 손익
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
    def update(self, pnl: float, returns: float, current_equity: float = None):
        """
        개선된 업데이트 함수
        - pnl: 실현 손익
        - returns: 수익률 
        - current_equity: 현재 총 자산 가치 (가용잔고 + 미실현손익)
        """
        # NaN이나 inf 값 체크
        if not np.isfinite(pnl):
            pnl = 0.0
        if not np.isfinite(returns):
            returns = 0.0
            
        # 최근 손익 및 수익률 기록 업데이트
        self.pnl_history.append(pnl)
        self.returns_history.append(returns)
        
        # 자산 가치 기록 (드로우다운 계산용)
        if current_equity is not None:
            self.equity_curve.append(current_equity)
            if self.initial_budget is None:
                self.initial_budget = current_equity
        
        # lookback 기간 초과된 데이터 제거
        if len(self.pnl_history) > self.lookback_period:
            self.pnl_history.pop(0)
            self.returns_history.pop(0)
    
    def update_trade_result(self, trade_pnl: float):
        """개별 거래 결과 업데이트 (실시간 승률 계산용)"""
        if not np.isfinite(trade_pnl):
            return
            
        self.trade_pnls.append(trade_pnl)
        
        if trade_pnl > 0:
            self.winning_trades += 1
            self.total_profit += trade_pnl
        elif trade_pnl < 0:
            self.losing_trades += 1
            self.total_loss += abs(trade_pnl)
        
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """개선된 샤프비율 계산"""
        if len(self.returns_history) < 2:
            return 0.0
            
        returns = np.array(self.returns_history)
        
        # NaN이나 inf 값 제거
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        mean_excess_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)  # 표본 표준편차 사용
        
        # 표준편차가 0에 가까우면 0 반환
        if std_return < 1e-8:
            return 0.0
        
        sharpe = mean_excess_return / std_return
        # 극값 제한 (연간화 고려시 일반적으로 -3~5 범위)
        return np.clip(sharpe, -10.0, 10.0)
    
    def get_max_drawdown(self) -> float:
        """개선된 최대 낙폭 계산"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(self.equity_curve)
        
        # NaN이나 inf 값 체크
        equity_array = equity_array[np.isfinite(equity_array)]
        if len(equity_array) < 2:
            return 0.0
        
        # 음수나 0인 자산 가치가 있으면 문제가 있음
        if np.any(equity_array <= 0):
            # 0 이하 값들을 최소 양수값으로 대체
            min_positive = np.min(equity_array[equity_array > 0]) if np.any(equity_array > 0) else 1.0
            equity_array = np.where(equity_array <= 0, min_positive, equity_array)
        
        # 누적 최고점 계산
        cummax = np.maximum.accumulate(equity_array)
        
        # 드로우다운 계산 (비율)
        drawdowns = (equity_array - cummax) / cummax
        max_dd = np.min(drawdowns)
        
        # 극값 제한 (-1.0 ~ 0.0 범위)
        return np.clip(max_dd, -1.0, 0.0)
    
    def get_volatility(self) -> float:
        """개선된 변동성 계산"""
        if len(self.returns_history) < 2:
            return 0.0
            
        returns = np.array(self.returns_history)
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns, ddof=1)
        # 극값 제한
        return np.clip(volatility, 0.0, 1.0)
    
    def get_calmar_ratio(self) -> float:
        """칼마 비율 계산 (연간 수익률 / 최대 드로우다운)"""
        if len(self.returns_history) < 2:
            return 0.0
        
        # 연간 수익률 계산 (252 거래일 기준)
        annual_return = np.mean(self.returns_history) * 252
        
        # 최대 드로우다운 (음수값이므로 절댓값 사용)
        max_dd = self.get_max_drawdown()  # 이미 음수로 반환됨
        max_dd_abs = abs(max_dd)  # 절댓값으로 변환
        
        # 드로우다운이 0에 가까우면 계산 불가
        if max_dd_abs < 1e-8:
            return 0.0 if annual_return <= 0 else float('inf')
            
        calmar = annual_return / max_dd_abs
        return np.clip(calmar, -10.0, 10.0)
    
    def get_win_rate(self) -> float:
        """실시간 승률 계산"""
        total_trades = self.winning_trades + self.losing_trades
        if total_trades == 0:
            return 0.0
        return self.winning_trades / total_trades
    
    def get_profit_factor(self) -> float:
        """이익 팩터 계산 (총 이익 / 총 손실)"""
        if self.total_loss < 1e-8:
            return float('inf') if self.total_profit > 0 else 1.0
        return self.total_profit / self.total_loss
    
    def get_average_trade(self) -> Dict[str, float]:
        """평균 거래 정보"""
        if len(self.trade_pnls) == 0:
            return {'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
        
        profits = [pnl for pnl in self.trade_pnls if pnl > 0]
        losses = [pnl for pnl in self.trade_pnls if pnl < 0]
        
        return {
            'avg_pnl': np.mean(self.trade_pnls),
            'avg_win': np.mean(profits) if profits else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0
        }
    
    def reset(self):
        """리스크 메트릭 초기화"""
        self.returns_history = []
        self.pnl_history = []
        self.equity_curve = []
        self.initial_budget = None
        
        # 거래별 성과 추적 초기화
        self.trade_pnls = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0


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
        # 'maturity_data' : 만기일, 'max_contract' : 최대 계약수 도달 
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
        
        # 성과 추적 개선
        self.equity_history = []  # 자산 가치 추적
        self.trade_count = 0      # 실제 거래 횟수 (action != 0)
        
        # 에피소드 단위 성과 추적 (EpisodicTrainer용)
        self.episode_pnl = 0          # 현재 에피소드의 총 손익
        self.episode_trades = 0       # 현재 에피소드의 거래 수
        self.episode_start_equity = start_budget  # 에피소드 시작 자산
        self.winning_episodes = 0     # 수익 에피소드 수 
        self.total_episodes = 0       # 총 에피소드 수
        
        # 실시간 거래 추적
        self.last_trade_pnl = 0.0     # 마지막 거래 손익
        self.cumulative_trade_pnl = 0.0  # 누적 거래 손익
        
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

        if len(mask) != self.n_actions:
            print(f"❗️[Warning] mask length mismatch: {len(mask)} != {self.n_actions}")
            print(f"remaining_strength: {remaining_strength}, position_cap: {self.position_cap}")
            mask = [1] * self.n_actions  # 안전장치

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
            return 0.0

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
        
        return net_pnl
    
    def _get_market_features(self) -> Dict[str, float]:
        """현재 시장 상태 관련 주요 지표 반환"""
        return {
            'market_regime': self.market_regime.value,
            'volatility_regime': {'low': -1, 'normal': 0, 'high': 1}[self.volatility_regime],
            'sharpe_ratio': self.risk_metrics.get_sharpe_ratio(),
            'max_drawdown': self.risk_metrics.get_max_drawdown(),
            'volatility': self.risk_metrics.get_volatility(),
            'win_rate': self.risk_metrics.get_win_rate(),
            'total_trades': self.risk_metrics.winning_trades + self.risk_metrics.losing_trades,
            'transaction_cost_ratio': self.account.total_transaction_costs / self.account.initial_budget
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
        개선된 환경 한 스텝 진행 (EpisodicTrainer 최적화)
        """
        # print(self.account)

        # 1. 다음 상태 데이터, 종가, 타임스텝 받아오기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)
        current_price = close_price

        # 2. 행동에 따른 계좌 업데이트
        net_realized_pnl, cost = self.account.step(action, current_price, next_timestep)

        # 3. 거래 카운트 및 성과 추적 개선
        if action != 0:
            self.trade_count += 1
            self.episode_trades += 1
            
            # 실시간 거래 손익 추적 (매 거래마다 즉시 평가)
            if net_realized_pnl != 0:  # 실현손익이 발생한 경우
                self.risk_metrics.update_trade_result(net_realized_pnl)
                self.last_trade_pnl = net_realized_pnl
                self.cumulative_trade_pnl += net_realized_pnl
            else:
                # 실현손익이 없더라도 미실현손익 변화를 평가
                prev_unrealized = getattr(self.account, 'prev_unrealized_pnl', 0)
                current_unrealized = self.account.unrealized_pnl
                unrealized_change = current_unrealized - prev_unrealized
                
                # 포지션 방향과 가격 변화를 고려한 손익 평가
                if self.account.current_position != 0 and abs(unrealized_change) > 1000:  # 임계값 설정
                    self.risk_metrics.update_trade_result(unrealized_change)

        # 4. 현재 총 자산 가치 계산 (가용잔고 + 미실현손익)
        current_equity = self.account.available_balance + self.account.unrealized_pnl
        current_equity = max(current_equity, 1.0)  # 음수 방지
        
        # 자산 가치 기록
        self.equity_history.append(current_equity)

        # 5. 일일 수익률 계산
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]
            daily_return = (current_equity - prev_equity) / max(prev_equity, 1.0)
        else:
            daily_return = 0.0

        # 6. 에피소드 손익 누적
        self.episode_pnl += net_realized_pnl

        # 7. 리스크 메트릭 업데이트 (개선된 방식)
        self.risk_metrics.update(
            pnl=net_realized_pnl,
            returns=daily_return,
            current_equity=current_equity
        )

        # 8. 거래 내역 기록
        self.trade_history.append({
            'timestamp': self.current_timestep,
            'action': action,
            'price': current_price,
            'pnl': net_realized_pnl,
            'cost': cost,
            'type': 'regular',
            'equity': current_equity
        })

        # 9. 시장 상태 업데이트
        current_idx = self.df.index.get_loc(self.current_timestep)
        start_idx = max(0, current_idx - self.window_size)
        price_data = self.df['close'].iloc[start_idx:current_idx].values
        if len(price_data) > 0:
            self._update_market_regime(price_data)

        # 10. 계좌 상태 확인 (info 설정)
        self.info = ''  # 초기화
        self._check_insufficient()
        self._check_near_margin_call()

        # 11. 종료 조건 확인
        done, self.info = self.switch_done_info(next_timestep, self.current_timestep)
        
        # 12. 강제 청산 처리 (보상 계산 전에 실행)
        forced_liquidation_pnl = 0.0
        if self.info in ['margin_call', 'maturity_data', 'bankrupt']:
            forced_liquidation_pnl = self._force_liquidate_all_positions()
            self.episode_pnl += forced_liquidation_pnl
            
            # 강제 청산 후 자산 가치 재계산
            final_equity = self.account.available_balance + self.account.unrealized_pnl
            final_equity = max(final_equity, 1.0)
            
            # 최종 수익률 계산
            if len(self.equity_history) > 1:
                final_return = (final_equity - self.equity_history[-2]) / max(self.equity_history[-2], 1.0)
            else:
                final_return = 0.0
                
            # 리스크 메트릭 최종 업데이트
            self.risk_metrics.update(
                pnl=forced_liquidation_pnl,
                returns=final_return,
                current_equity=final_equity
            )

        # 13. 에피소드 종료 시 성과 업데이트
        if done:
            self.total_episodes += 1
            episode_return = (current_equity - self.episode_start_equity) / self.episode_start_equity
            
            if episode_return > 0 or self.episode_pnl > 0:
                self.winning_episodes += 1

        # 14. 보상 계산 (모든 거래 완료 후 실행)
        reward = self.get_reward(
            unrealized_pnl=self.account.unrealized_pnl,
            prev_unrealized_pnl=getattr(self.account, 'prev_unrealized_pnl', 0),
            current_budget=self.account.available_balance,
            transaction_cost=cost,
            risk_metrics=self.risk_metrics,
            market_regime=self.market_regime,
            daily_return=daily_return,
            net_realized_pnl=net_realized_pnl
        )

        # 15. 다음 상태 생성
        next_state = self.state(
            next_fixed_state,  # 실제 데이터 기반 상태
            current_position=self.account.current_position,
            execution_strength=self.account.execution_strength,
            realized_pnl=self.account.realized_pnl,
            unrealized_pnl=self.account.unrealized_pnl,
            maintenance_margin=self.account.maintenance_margin,
            total_transaction_costs=self.account.total_transaction_costs
        )

        # 16. action space에 대한 마스크 생성 
        self.mask = self.get_mask()

        # 17. 상태 업데이트
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        # 18. 다음 상태, 보상, 종료 플래그 반환
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
        
        # 리스크 한도 체크 (필요시 활성화)
        # done, info = self._is_risk_limits()
        # if done:
        #     return done, info
        
        return False, ''

    def get_performance_summary(self) -> Dict[str, Any]:
        """개선된 성과 요약 (EpisodicTrainer 최적화)"""
        current_equity = self.account.available_balance + self.account.unrealized_pnl
        total_return = (current_equity / self.account.initial_budget) - 1
        
        # 실시간 승률 
        trade_win_rate = self.risk_metrics.get_win_rate()
        
        # 에피소드 승률
        episode_win_rate = self.winning_episodes / max(self.total_episodes, 1) if self.total_episodes > 0 else 0
        
        # 거래 기반 통계
        total_completed_trades = self.risk_metrics.winning_trades + self.risk_metrics.losing_trades
        
        return {
            # 기본 성과 지표
            'total_return': total_return,
            'current_equity': current_equity,
            'episode_return': (current_equity - self.episode_start_equity) / self.episode_start_equity,
            
            # 거래 통계 
            'total_actions': self.trade_count,           # 실제 action 횟수  
            'completed_trades': total_completed_trades,   # 완료된 거래 횟수
            'episode_trades': self.episode_trades,        # 현재 에피소드 거래 수
            
            # 승률 (두 가지 방식)
            'trade_win_rate': trade_win_rate,             # 거래별 승률
            'episode_win_rate': episode_win_rate,         # 에피소드별 승률
            'winning_trades': self.risk_metrics.winning_trades,
            'losing_trades': self.risk_metrics.losing_trades,
            
            # 리스크 지표
            'sharpe_ratio': self.risk_metrics.get_sharpe_ratio(),
            'max_drawdown': self.risk_metrics.get_max_drawdown(),
            'volatility': self.risk_metrics.get_volatility(),
            'calmar_ratio': self.risk_metrics.get_calmar_ratio(),
            'profit_factor': self.risk_metrics.get_profit_factor(),
            
            # 거래 분석
            'avg_trade_info': self.risk_metrics.get_average_trade(),
            'episode_pnl': self.episode_pnl,
            'cumulative_trade_pnl': self.cumulative_trade_pnl,
            'last_trade_pnl': self.last_trade_pnl,
            
            # 비용 분석
            'total_transaction_costs': self.account.total_transaction_costs,
            'cost_ratio': self.account.total_transaction_costs / self.account.initial_budget,
            
            # 시장 상태
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime,
            'unrealized_pnl': self.account.unrealized_pnl,
            
            # 에피소드 통계
            'total_episodes': self.total_episodes,
            'winning_episodes': self.winning_episodes
        }
    
    def reset(self):
        """
        개선된 환경 초기화 (EpisodicTrainer 최적화)
        """
        
        # 1. 계좌 초기화
        self.account.reset()
        
        # 2. 에피소드 단위 성과 추적 초기화
        self.episode_pnl = 0
        self.episode_trades = 0
        self.episode_start_equity = self.account.initial_budget
        
        # 3. 거래 추적 변수 초기화
        self.trade_count = 0                    # 실제 action 횟수
        self.last_trade_pnl = 0.0
        self.cumulative_trade_pnl = 0.0
        
        # 4. 거래 및 자산 기록 초기화
        self.trade_history = []                 # 거래 내역
        self.daily_returns = []                 # 일일 수익률
        self.equity_history = []                # 자산 가치 변화
        
        # 5. 리스크 메트릭 리셋
        self.risk_metrics.reset()
        
        # 6. 시장 상태 초기화
        self.market_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = 'normal'
        
        # 7. 환경 상태 초기화
        self.info = ''                          # 상태 정보 초기화
        self.mask = [1] * self.n_actions        # 액션 마스크 초기화
        
        # 8. 데이터 이터레이터 재설정
        self.data_iterator = iter(self.dataset)
        fixed_state, close_price, timestep = next(self.data_iterator)
        
        # 9. 초기 시장 정보 설정
        self.previous_price = close_price
        self.current_timestep = timestep
        
        # 10. 초기 자산 가치 기록 (드로우다운 계산을 위해)
        initial_equity = self.account.available_balance + self.account.unrealized_pnl
        initial_equity = max(initial_equity, 1.0)  # 음수 방지
        self.equity_history.append(initial_equity)
        self.episode_start_equity = initial_equity
        
        # 11. 리스크 메트릭에 초기 자산 가치 설정
        self.risk_metrics.update(
            pnl=0.0,
            returns=0.0,
            current_equity=initial_equity
        )
        
        # 12. 초기 상태 생성 및 반환
        initial_state = self.state(
            fixed_state,                        # 실제 데이터 기반 상태
            current_position=self.account.current_position,      # 0
            execution_strength=self.account.execution_strength,  # 0
            realized_pnl=self.account.realized_pnl,             # 0.0
            unrealized_pnl=self.account.unrealized_pnl,         # 0.0
            maintenance_margin=self.account.maintenance_margin, # 0.0
            total_transaction_costs=self.account.total_transaction_costs  # 0.0
        )
        
        return initial_state
    
    def conti(self):
        """done 후에도 다음 상태를 반환 (연속 거래용)"""
        return self.next_state
    
    def render(self, state, action, next_state):
        """기존 코드 호환성을 위한 render 메서드"""
        close_idx = self.dataset.indices.index('close')
        # memory : 제대로 예측이 되는지 보여줄 수 있는 지표여야 한다. 
        pass 
    
    def __str__(self):
        """환경 상태 및 주요 성과를 섹션별로 나누어 출력"""
        perf = self.get_performance_summary()
        
        # 계좌 상태는 account 객체에서 가져오기
        account_status = str(self.account)
        
        # 성과 지표 섹션
        performance_section = (
            f"📁 2. Performance Metrics (성과 지표)\n"
            f"💰  Current Equity     : {perf['current_equity']:,.0f} KRW\n"
            f"💵  Total Return       : {perf['total_return']*100:.2f}%\n"
            f"📈  Episode Return     : {perf['episode_return']*100:.2f}%\n"
            f"🏆  Episode Win Rate   : {perf['episode_win_rate']*100:.1f}% ({perf['winning_episodes']}/{perf['total_episodes']})\n"
            f"🎯  Trade Win Rate     : {perf['trade_win_rate']*100:.1f}% ({perf['winning_trades']}/{perf['completed_trades']})\n"
            f"📊  Sharpe Ratio       : {perf['sharpe_ratio']:.3f}\n"
            f"📉  Max Drawdown       : {perf['max_drawdown']*100:.1f}%\n"
            f"📈  Volatility         : {perf['volatility']*100:.1f}%\n"
            f"🔄  Calmar Ratio       : {perf['calmar_ratio']:.3f}\n"
            f"💎  Profit Factor      : {perf['profit_factor']:.2f}\n"
            f"💸  Cost Ratio         : {perf['cost_ratio']*100:.2f}%\n"
            f"===============================================\n"
        )
        
        # 거래 기록 섹션
        trade_history_section = (
            f"📁 3. Trade History (거래 기록)\n"
            f"✅  Completed Trades   : {perf['completed_trades']}\n"
            f"📋  Episode Trades     : {perf['episode_trades']}\n"
            f"💰  Episode PnL        : {perf['episode_pnl']:,.0f} KRW\n"
            f"💹  Last Trade PnL     : {perf['last_trade_pnl']:,.0f} KRW\n"
            f"💹  Cumulative PnL     : {perf['cumulative_trade_pnl']:,.0f} KRW\n"
            f"🔢  Total Actions      : {perf['total_actions']}\n"
            f"📊  Avg Trade Info     : Win={perf['avg_trade_info']['avg_win']:,.0f}, Loss={perf['avg_trade_info']['avg_loss']:,.0f}\n"
            f"===============================================\n"
        )
        
        # 시장 상태 섹션
        market_conditions_section = (
            f"📁 4. Market Conditions (시장 상태)\n"
            f"📈  Previous Close     : {self.previous_price:.2f}\n"
            f"🌍  Market Regime      : {self.market_regime.name}\n"
            f"📈  Volatility Regime  : {self.volatility_regime}\n"
            f"ℹ️  Info Status        : {self.info}\n"
            f"🎭  Action Mask        : {sum(self.mask)}/{len(self.mask)} valid actions\n"
            f"===============================================\n"
        )
        
        return account_status + performance_section + trade_history_section + market_conditions_section