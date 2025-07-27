from enum import Enum
import numpy as np

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