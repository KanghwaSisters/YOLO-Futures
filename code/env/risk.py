import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# 시장 상태 구분 Enum (강세장, 약세장, 횡보장)
class MarketRegime(Enum):
    BULL = 1        # 강세장 : 최근 저점 대비 20% 이상 상승 
    BEAR = -1       # 약세장 : 최근 고점 대비 20% 이상 하락 
    SIDEWAYS = 0    # 횡보장 : etc

@dataclass
class TradeRecord:
    """거래 기록 데이터 클래스"""
    timestamp: str
    action: int
    price: float
    pnl: float
    cost: float
    trade_type: str  # 'regular', 'forced_liquidation'
    equity: float
    position: int
    execution_strength: int

class MarketStateManager:
    """시장 상태 관리 클래스"""
    
    def __init__(self):
        self.market_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = 'normal'  # 'low', 'normal', 'high'
        self._price_history = []
        
    def update_market_regime(self, price_data: np.ndarray):
        """가격 데이터를 바탕으로 시장 상태 갱신"""
        if len(price_data) < 80:
            return
        
        short_ma = np.mean(price_data[-5:])
        long_ma = np.mean(price_data[-80:])
        
        # 트렌드 방향 판단 (2% 임계값)
        if short_ma > long_ma * 1.003:
            self.market_regime = MarketRegime.BULL
        elif short_ma < long_ma * 0.997:
            self.market_regime = MarketRegime.BEAR
        else:
            self.market_regime = MarketRegime.SIDEWAYS
        
        # 변동성 구간 판단
        volatility = np.std(price_data[-10:]) / np.mean(price_data[-10:])
        if volatility > 0.02:
            self.volatility_regime = 'high'
        elif volatility < 0.01:
            self.volatility_regime = 'low'
        else:
            self.volatility_regime = 'normal'
    
    def get_regime_value(self) -> int:
        """시장 상태를 수치로 반환"""
        return self.market_regime.value
    
    def get_volatility_value(self) -> int:
        """변동성 상태를 수치로 반환"""
        return {'low': -1, 'normal': 0, 'high': 1}[self.volatility_regime]

class PerformanceTracker:
    """성과 추적 관리 클래스"""
    
    def __init__(self, initial_budget: float):
        self.initial_budget = initial_budget
        
        # 에피소드 단위 추적
        self.episode_pnl = 0.0
        self.episode_trades = 0
        self.episode_start_equity = initial_budget
        self.total_episodes = 0
        self.winning_episodes = 0
        
        # 거래 단위 추적
        self.trade_count = 0  # 실제 action 횟수 (action != 0)
        self.last_trade_pnl = 0.0
        self.cumulative_trade_pnl = 0.0
        
        # 기록 저장
        self.equity_history: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_history: List[TradeRecord] = []
    
    def update_trade(self, action: int, net_pnl: float, cost: float, 
                    current_price: float, current_timestep: str, 
                    current_equity: float, position: int, execution_strength: int,
                    trade_type: str = 'regular'):
        """거래 정보 업데이트"""
        if action != 0:
            self.trade_count += 1
            self.episode_trades += 1
            
            if net_pnl != 0:
                self.last_trade_pnl = net_pnl
                self.cumulative_trade_pnl += net_pnl
        
        # 에피소드 손익 누적
        self.episode_pnl += net_pnl
        
        # 거래 기록 저장
        trade_record = TradeRecord(
            timestamp=current_timestep,
            action=action,
            price=current_price,
            pnl=net_pnl,
            cost=cost,
            trade_type=trade_type,
            equity=current_equity,
            position=position,
            execution_strength=execution_strength
        )
        self.trade_history.append(trade_record)
    
    def update_equity(self, current_equity: float):
        """자산 가치 업데이트"""
        current_equity = max(current_equity, 1.0)  # 음수 방지
        self.equity_history.append(current_equity)
        
        # 일일 수익률 계산
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]
            daily_return = (current_equity - prev_equity) / max(prev_equity, 1.0)
            self.daily_returns.append(daily_return)
            return daily_return
        
        self.daily_returns.append(0.0)
        return 0.0
    
    def complete_episode(self, final_equity: float):
        """에피소드 완료 처리"""
        self.total_episodes += 1
        episode_return = (final_equity - self.episode_start_equity) / self.episode_start_equity
        
        if episode_return > 0 or self.episode_pnl > 0:
            self.winning_episodes += 1
    
    def get_episode_win_rate(self) -> float:
        """에피소드 승률"""
        if self.total_episodes == 0:
            return 0.0
        return self.winning_episodes / self.total_episodes
    
    def reset(self):
        """성과 추적 초기화"""
        self.episode_pnl = 0.0
        self.episode_trades = 0
        self.episode_start_equity = self.initial_budget
        self.trade_count = 0
        self.last_trade_pnl = 0.0
        self.cumulative_trade_pnl = 0.0
        self.equity_history = []
        self.daily_returns = []
        self.trade_history = []

class RiskMetrics:
    """리스크 지표 계산 클래스"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.returns_history = []
        self.pnl_history = []
        self.equity_curve = []
        self.initial_budget = None
        
        # 거래별 성과 추적
        self.trade_pnls = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
    
    def update(self, pnl: float, returns: float, current_equity: Optional[float] = None):
        """리스크 메트릭 업데이트"""
        # NaN/inf 값 검증
        pnl = 0.0 if not np.isfinite(pnl) else pnl
        returns = 0.0 if not np.isfinite(returns) else returns
        
        # 기록 업데이트
        self.pnl_history.append(pnl)
        self.returns_history.append(returns)
        
        # 자산 곡선 업데이트
        if current_equity is not None:
            self.equity_curve.append(current_equity)
            if self.initial_budget is None:
                self.initial_budget = current_equity
        
        # lookback 기간 유지
        if len(self.pnl_history) > self.lookback_period:
            self.pnl_history.pop(0)
            self.returns_history.pop(0)
    
    def update_trade_result(self, trade_pnl: float):
        """개별 거래 결과 업데이트"""
        if not np.isfinite(trade_pnl) or trade_pnl == 0:
            return
        
        self.trade_pnls.append(trade_pnl)
        
        if trade_pnl > 0:
            self.winning_trades += 1
            self.total_profit += trade_pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(trade_pnl)
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """샤프 비율 계산"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) < 2:
            return 0.0
        
        excess_returns = returns_array - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return < 1e-8:
            return 0.0
        
        sharpe = mean_return / std_return
        return np.clip(sharpe, -10.0, 10.0)
    
    def get_max_drawdown(self) -> float:
        """최대 드로우다운 계산"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(self.equity_curve)
        equity_array = equity_array[np.isfinite(equity_array)]
        
        if len(equity_array) < 2:
            return 0.0
        
        # 음수/0 처리
        if np.any(equity_array <= 0):
            min_positive = np.min(equity_array[equity_array > 0]) if np.any(equity_array > 0) else 1.0
            equity_array = np.where(equity_array <= 0, min_positive, equity_array)
        
        # 드로우다운 계산
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax
        max_dd = np.min(drawdowns)
        
        return np.clip(max_dd, -1.0, 0.0)
    
    def get_volatility(self) -> float:
        """변동성 계산"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) < 2:
            return 0.0
        
        volatility = np.std(returns_array, ddof=1)
        return np.clip(volatility, 0.0, 1.0)
    
    def get_calmar_ratio(self) -> float:
        """칼마 비율 계산"""
        if len(self.returns_history) < 2:
            return 0.0
        
        annual_return = np.mean(self.returns_history) * 252
        max_dd = self.get_max_drawdown()
        max_dd_abs = abs(max_dd)
        
        if max_dd_abs < 1e-8:
            return 0.0 if annual_return <= 0 else 10.0
        
        calmar = annual_return / max_dd_abs
        return np.clip(calmar, -10.0, 10.0)
    
    def get_win_rate(self) -> float:
        """거래 승률"""
        total_trades = self.winning_trades + self.losing_trades
        return self.winning_trades / total_trades if total_trades > 0 else 0.0
    
    def get_profit_factor(self) -> float:
        """수익 팩터"""
        if self.total_loss < 1e-8:
            return 10.0 if self.total_profit > 0 else 1.0
        return min(self.total_profit / self.total_loss, 100.0)  # 상한 제한
    
    def get_average_trade(self) -> Dict[str, float]:
        """평균 거래 정보"""
        if not self.trade_pnls:
            return {'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
        
        profits = [pnl for pnl in self.trade_pnls if pnl > 0]
        losses = [pnl for pnl in self.trade_pnls if pnl < 0]
        
        return {
            'avg_pnl': np.mean(self.trade_pnls),
            'avg_win': np.mean(profits) if profits else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0
        }
    
    def get_summary(self) -> Dict[str, float]:
        """리스크 지표 요약"""
        return {
            'sharpe_ratio': self.get_sharpe_ratio(),
            'max_drawdown': self.get_max_drawdown(),
            'volatility': self.get_volatility(),
            'calmar_ratio': self.get_calmar_ratio(),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'total_trades': self.winning_trades + self.losing_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
    
    def reset(self):
        """리스크 메트릭 초기화"""
        self.returns_history = []
        self.pnl_history = []
        self.equity_curve = []
        self.initial_budget = None
        self.trade_pnls = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0