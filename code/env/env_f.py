import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from datahandler.dataset import *
from env.done_ftn import *
from env.reward_ftn import *
from env.account import *
from env.maturity_ftn import *

# 선물 트레이딩 환경 클래스
from env.risk import RiskMetrics, MarketStateManager, PerformanceTracker, MarketRegime

class FuturesEnvironment:
    """선물 거래 환경 클래스"""
    
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
                 # 거래 비용 및 리스크 파라미터
                 transaction_cost: float = 0.0005,
                 slippage_factor: float = 0.0001,
                 margin_requirement: float = 0.1,
                 max_drawdown_limit: float = 0.2,
                 intraday_only: bool = False,
                 risk_lookback: int = 20):
        
        # === 기본 환경 설정 ===
        self._full_df = full_df
        self._date_range = date_range
        self.n_actions = n_actions
        self.df = self._slice_by_date(full_df, date_range)
        
        self.scaler = scaler
        self.window_size = window_size
        
        # 데이터셋 및 이터레이터
        self.dataset = FuturesDataset(self.df, window_size, self.scaler)
        self.data_iterator = iter(self.dataset)
        
        # 상태 관리
        self.state = state_type
        self.state.get_dataset_indices(self.dataset.indices)
        self.next_state = None
        
        # === 포지션 및 거래 설정 ===
        self.position_dict = {-1: 'short', 0: 'hold', 1: 'long'}
        self.position_cap = position_cap
        self.single_execution_cap = self.n_actions // 2
        
        # === 시장 정보 ===
        self.previous_price = None
        self.contract_unit = 50000  # 미니 선물 계약 단위
        self.current_timestep = date_range[0]
        
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
        self.performance_tracker = PerformanceTracker(start_budget)
        self.risk_metrics = RiskMetrics(risk_lookback)
        # self.total_trades = 0
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

    def _get_n_days_before_maturity(self, current_timestep):
        current_day = current_timestep.date()

        # 만약 만기일을 넘었다면, 다음 만기일로 이동 
        if (self.latest_maturity_day - current_day).days <= -1:
            self.latest_maturity_day = next(self.maturity_iter)
        
        return (self.latest_maturity_day - current_day).days


    def get_mask(self):
        position = self.account.current_position                                    # 
        remaining_strength = self.position_cap - self.account.execution_strength    # 가용 계약수 
        half = self.single_execution_cap
        n = self.n_actions

        # 기본 마스크 생성
        mask = np.ones(n, dtype=np.int32)

        if (self.position_cap == remaining_strength) or (self.info == 'insufficient'):
            # 최대 체결 가능 계약수에 도달했을 때 
            # 자본금 부족으로 새로운 포지션을 체결할 수 없을 때 
            if position == -1: # short 
                mask[:half] = 0
    
            elif position == 1: # long 
                mask[-half:] = 0 
            

        elif (remaining_strength) < self.single_execution_cap:
            # 최대 체결 가능 계약수에 근접하여 일부 행동에 제약이 있다. 
            restriction = half - remaining_strength 

            if self.account.current_position == -1: # short 
                mask[:restriction] = 0
            elif self.account.current_position == 1: # long 
                mask[-restriction:] = 0

        return mask.tolist()

    def _slice_by_date(self, full_df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
        """날짜 범위로 데이터프레임 슬라이싱"""
        full_df = full_df.copy()
        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.sort_index()
        
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return full_df[(full_df.index >= start) & (full_df.index <= end)]
    
    def _force_liquidate_all_positions(self, current_price):
        """리스크 제한 초과 시 모든 포지션 강제 청산"""
        if self.account.execution_strength == 0:
            return

        # 현재 체결된 계약에서 반대 포지션을 취함 
        reversed_execution = -self.account.execution_strength * self.account.current_position
        net_pnl, cost = self.account.settle_total_contract(market_pt=current_price) # self.account.step(reversed_execution, self.previous_price, self.current_timestep)
        # prev 맞는 지 고민하기 
        return net_pnl, cost, reversed_execution
    
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

    def _update_market_conditions(self):
        """시장 상태 업데이트"""
        current_idx = self.df.index.get_loc(self.current_timestep)
        start_idx = max(0, current_idx - self.window_size)
        price_data = self.df['close'].iloc[start_idx:current_idx].values
        
        if len(price_data) > 0:
            self.market_state_manager.update_market_regime(price_data)
    
    def _check_account_status(self):
        """계좌 상태 확인"""
        self.info = ''  # 초기화
        
        # 자본금 부족 확인
        if self.account.available_balance <= self.previous_price * self.account.initial_margin_rate:
            self.info = 'insufficient'
        
        # 마진콜 확인
        if self.account.available_balance <= self.account.maintenance_margin:
            self.info = 'margin_call'
    
    def _check_termination_conditions(self, next_timestep) -> Tuple[bool, str]:
        """종료 조건 확인"""
        # 사용자 정의 done 함수 확인
        if self.get_done(
            current_timestep=self.current_timestep,
            next_timestep=next_timestep,
            max_strength=self.position_cap,
            current_strength=self.account.execution_strength,
            intraday_only=self.intraday_only
        ):
            return True, 'done'
        
        # 데이터셋 종료 확인
        if self.dataset.reach_end(next_timestep):
            return True, 'end_of_data'
        
        # 만기일 확인
        is_maturity_date = self.current_timestep.date() in self.maturity_list
        day_changed = is_day_changed(next_timestep=next_timestep, current_timestep=self.current_timestep)
        if is_maturity_date and day_changed:
            return True, 'maturity_data'
        
        # 파산 확인
        if self.account.available_balance <= 0:
            return True, 'bankrupt'
        
        # 리스크 한도 확인 (옵션)
        # total_return = (self.account.available_balance + self.account.unrealized_pnl) / self.account.initial_budget - 1
        # if total_return < -self.max_drawdown_limit:
        #     return True, 'risk_limits'
        
        return False, ''
    
    def step(self, action: int) -> Tuple[Any, float, bool]:
        """환경 스텝 실행"""
        # 1. 다음 데이터 가져오기
        next_fixed_state, close_price, next_timestep = next(self.data_iterator)
        current_price = close_price
        
        # 2. 계좌 업데이트 (거래 실행)
        net_realized_pnl, cost = self.account.step(action, current_price, next_timestep)
        
        # ===========================================
        # 
        self._check_insufficient()
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
        if self.info in ['margin_call', 'maturity_data']:
            forced_liquidation_pnl, _cost = self._force_liquidate_all_positions()
            
            # 강제 청산 후 자산 계산
            current_equity = self.account.available_balance + self.account.unrealized_pnl
            current_equity = max(current_equity, 1.0)
            
            daily_return = self.performance_tracker.update_equity(current_equity)
             
            # 최종 수익률 계산 및 업데이트
            # ============ ㅙ 있는지 ㅘ긴 =============
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

        if self.info == 'done':
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
            execution_strength=self.account.execution_strength
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
            market_regime=perf['market_regime'],
            volatility_regime=perf['volatility_regime']
        )

        # 12. action space에 대한 마스크 생성 
        self.mask = self.get_mask()

        # 업데이트 
        self.next_state = next_state
        self.previous_price = current_price
        self.current_timestep = next_timestep

        return next_state, reward, done
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약 반환"""
        current_equity = self.account.available_balance + self.account.unrealized_pnl
        total_return = (current_equity / self.account.initial_budget) - 1
        
        # 리스크 지표 요약
        risk_summary = self.risk_metrics.get_summary()
        
        return {
            # 기본 성과 지표
            'total_return': total_return,
            'current_equity': current_equity,
            'episode_return': (current_equity - self.performance_tracker.episode_start_equity) / 
                             self.performance_tracker.episode_start_equity,
            
            # 거래 통계
            'total_actions': self.performance_tracker.trade_count,
            'completed_trades': risk_summary['total_trades'],
            'episode_trades': self.performance_tracker.episode_trades,
            
            # 승률 (두 가지 방식)
            'trade_win_rate': risk_summary['win_rate'],
            'episode_win_rate': self.performance_tracker.get_episode_win_rate(),
            'winning_trades': risk_summary['winning_trades'],
            'losing_trades': risk_summary['losing_trades'],
            
            # 리스크 지표
            'sharpe_ratio': risk_summary['sharpe_ratio'],
            'max_drawdown': risk_summary['max_drawdown'],
            'volatility': risk_summary['volatility'],
            'calmar_ratio': risk_summary['calmar_ratio'],
            'profit_factor': risk_summary['profit_factor'],
            
            # 거래 분석
            'avg_trade_info': self.risk_metrics.get_average_trade(),
            'episode_pnl': self.performance_tracker.episode_pnl,
            'cumulative_trade_pnl': self.performance_tracker.cumulative_trade_pnl,
            'last_trade_pnl': self.performance_tracker.last_trade_pnl,
            
            # 비용 분석
            'total_transaction_costs': self.account.total_transaction_costs,
            'cost_ratio': self.account.total_transaction_costs / self.account.initial_budget,
            
            # 시장 상태
            'market_regime': self.market_state_manager.get_regime_value(),
            'volatility_regime': self.market_state_manager.get_volatility_value(),
            'unrealized_pnl': self.account.unrealized_pnl,
            
            # 에피소드 통계
            'total_episodes': self.performance_tracker.total_episodes,
            'winning_episodes': self.performance_tracker.winning_episodes
        }
    
    def reset(self):
        """환경 초기화"""
        # 1. 계좌 초기화
        self.account.reset()
        
        # 2. 관리 객체들 초기화
        self.performance_tracker.reset()
        self.risk_metrics.reset()
        
        # 3. 환경 상태 초기화
        self.info = ''
        self.mask = [1] * self.n_actions
        
        # 4. 데이터 이터레이터 재설정
        self.data_iterator = iter(self.dataset)
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
            market_regime=0,
            volatility_regime=0
        )
        
        return initial_state
    
    def conti(self):
        """done 후에도 다음 상태를 반환 (연속 거래용)"""
        return self.next_state
    
    def render(self, state, action, next_state):
        """기존 코드 호환성을 위한 render 메서드"""
        # 필요시 시각화 로직 구현
        pass
    
    def __str__(self):
        """환경 상태 및 주요 성과 출력"""
        perf = self.get_performance_summary()
        
        # 계좌 상태
        account_status = str(self.account)
        
        # 성과 지표 섹션
        performance_section = (
            f"📁 2. Performance Metrics (성과 지표)\n"
            f"💵  Total Return       : {perf['total_return']*100:.2f}%\n"
            f"🏆  Episode Win Rate   : {perf['episode_win_rate']*100:.1f}% "
            f"({perf['winning_episodes']}/{perf['total_episodes']})\n"
            f"🎯  Trade Win Rate     : {perf['trade_win_rate']*100:.1f}% "
            f"({perf['winning_trades']}/{perf['completed_trades']})\n"
            f"📊  Sharpe Ratio       : {perf['sharpe_ratio']:.3f}\n"
            f"📉  Max Drawdown       : {perf['max_drawdown']*100:.1f}%\n"
            f"===============================================\n"
        )
        
        # 거래 기록 섹션
        trade_history_section = (
            f"📁 3. Trade History (거래 기록)\n"
            f"✅  Completed Trades   : {perf['completed_trades']}\n"
            f"💰  Episode PnL        : {perf['episode_pnl']:,.0f} KRW\n"
            f"💹  Last Trade PnL     : {perf['last_trade_pnl']:,.0f} KRW\n"
            f"🔢  Total Actions      : {perf['total_actions']}\n"
            f"===============================================\n"
        )
        
        # 시장 상태 섹션
        market_regime_name = {1: 'BULL', -1: 'BEAR', 0: 'SIDEWAYS'}[perf['market_regime']]
        
        market_conditions_section = (
            f"📁 4. Market Conditions (시장 상태)\n"
            f"📈  Previous Close     : {self.previous_price:.2f}\n"
            f"🌍  Market Regime      : {market_regime_name}\n"
            f"ℹ️  Info Status        : {self.info}\n"
            f"===============================================\n"
        )
        
        return account_status + performance_section + trade_history_section + market_conditions_section
    
    def get_detailed_status(self):
        """상세한 환경 상태 출력"""
        perf = self.get_performance_summary()
        
        # 계좌 상태
        total_equity = self.account.available_balance + self.account.unrealized_pnl
        detailed_account = (
            f"===============================================\n"
            f"📁 1. Account Status (계좌 상태)\n"
            f"⏱️  Current Timestep   : {self.current_timestep}\n"
            f"💰  Available Balance  : {self.account.available_balance:,.0f} KRW\n"
            f"💼  Margin Deposit     : {self.account.margin_deposit:,.0f} KRW\n"
            f"💸  Transaction Costs  : {self.account.total_transaction_costs:,.0f} KRW\n"
            f"📉  Unrealized PnL     : {self.account.unrealized_pnl:,.0f} KRW\n"
            f"💵  Realized PnL       : {self.account.realized_pnl:,.0f} KRW\n"
            f"💰  Total Equity       : {total_equity:,.0f} KRW\n"
            f"⚖️  Avg Entry Price    : {self.account.average_entry:.2f}\n"
            f"💼  Current Position   : {self.account.position_dict[self.account.current_position]} ({self.account.current_position})\n"
            f"📊  Execution Strength : {self.account.execution_strength}/{self.account.position_cap}\n"
            f"🔢  Total Trades       : {self.account.total_trades}\n"
            f"===============================================\n"
        )
        
        # 성과 지표
        detailed_performance = (
            f"📁 2. Performance Metrics (성과 지표)\n"
            f"💰  Current Equity     : {perf['current_equity']:,.0f} KRW\n"
            f"💵  Total Return       : {perf['total_return']*100:.2f}%\n"
            f"📈  Episode Return     : {perf['episode_return']*100:.2f}%\n"
            f"🏆  Episode Win Rate   : {perf['episode_win_rate']*100:.1f}% "
            f"({perf['winning_episodes']}/{perf['total_episodes']})\n"
            f"🎯  Trade Win Rate     : {perf['trade_win_rate']*100:.1f}% "
            f"({perf['winning_trades']}/{perf['completed_trades']})\n"
            f"📊  Sharpe Ratio       : {perf['sharpe_ratio']:.3f}\n"
            f"📉  Max Drawdown       : {perf['max_drawdown']*100:.1f}%\n"
            f"📈  Volatility         : {perf['volatility']*100:.1f}%\n"
            f"🔄  Calmar Ratio       : {perf['calmar_ratio']:.3f}\n"
            f"💎  Profit Factor      : {perf['profit_factor']:.2f}\n"
            f"💸  Cost Ratio         : {perf['cost_ratio']*100:.2f}%\n"
            f"===============================================\n"
        )
        
        # 거래 기록
        detailed_trades = (
            f"📁 3. Trade History (거래 기록)\n"
            f"✅  Completed Trades   : {perf['completed_trades']}\n"
            f"📋  Episode Trades     : {perf['episode_trades']}\n"
            f"💰  Episode PnL        : {perf['episode_pnl']:,.0f} KRW\n"
            f"💹  Last Trade PnL     : {perf['last_trade_pnl']:,.0f} KRW\n"
            f"💹  Cumulative PnL     : {perf['cumulative_trade_pnl']:,.0f} KRW\n"
            f"🔢  Total Actions      : {perf['total_actions']}\n"
            f"📊  Avg Trade Info     : Win={perf['avg_trade_info']['avg_win']:,.0f}, "
            f"Loss={perf['avg_trade_info']['avg_loss']:,.0f}\n"
            f"===============================================\n"
        )
        
        # 시장 상태
        market_regime_name = {1: 'BULL', -1: 'BEAR', 0: 'SIDEWAYS'}[perf['market_regime']]
        volatility_regime_name = {-1: 'LOW', 0: 'NORMAL', 1: 'HIGH'}[perf['volatility_regime']]
        
        detailed_market = (
            f"📁 4. Market Conditions (시장 상태)\n"
            f"📈  Previous Close     : {self.previous_price:.2f}\n"
            f"🌍  Market Regime      : {market_regime_name}\n"
            f"📈  Volatility Regime  : {volatility_regime_name}\n"
            f"ℹ️  Info Status        : {self.info}\n"
            f"🎭  Action Mask        : {sum(self.mask)}/{len(self.mask)} valid actions\n"
            f"===============================================\n"
        )
        
        return detailed_account + detailed_performance + detailed_trades + detailed_market