# state.py (수정된 부분만)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class State:
    def __init__(self, target_values: List[str], indices_list: Optional[List[str]] = None, 
                 fixed_agent_dim: Optional[int] = None):
        """
        강화학습 상태를 처리하는 클래스
        
        Args:
            target_values: 시계열 데이터에서 사용할 특성 이름들
            indices_list: 데이터셋의 전체 컬럼 인덱스 리스트
            fixed_agent_dim: 고정된 agent 차원수 (None이면 동적 처리)
        """
        self.target_values = target_values
        self._indices_list = indices_list
        self.fixed_agent_dim = fixed_agent_dim
        
        # 에이전트 특성의 기본 순서 정의 (일관성을 위해)
        self.default_agent_features = [
            'current_position', 'execution_strength', 'market_regime', 
            'volatility_regime', 'sharpe_ratio', 'max_drawdown', 
            'volatility', 'win_rate', 'total_trades', 'transaction_cost_ratio'
        ]
        
        # 최소 필수 특성들 (reset시에만 사용)
        self.minimal_features = ['current_position', 'execution_strength']
        
        if indices_list is None:
            self.flag = False
        else:
            self.get_dataset_indices(indices_list)
    
    def get_dataset_indices(self, indices_list: List[str]):
        """데이터셋의 인덱스 리스트를 받아 타겟 인덱스를 설정"""
        self.flag = True
        self._indices_list = indices_list
        
        # Safe indexing - target_values가 indices_list에 있는지 확인
        try:
            self.target_indices = [self._indices_list.index(val) for val in self.target_values]
        except ValueError as e:
            missing_values = [val for val in self.target_values if val not in self._indices_list]
            raise ValueError(f"target_values {missing_values} must exist in indices_list. Available: {self._indices_list}")
    
    def _get_target_data(self, data: np.ndarray, target_indices: List[int]) -> np.ndarray:
        """시계열 데이터에서 타겟 특성만 추출"""
        return data[:, target_indices]
    
    def _process_agent_features(self, **kwargs) -> np.ndarray:
        """에이전트 특성들을 일관된 순서로 정렬하여 배열로 변환"""
        # fixed_agent_dim이 설정된 경우, 해당 크기에 맞춰 처리
        if self.fixed_agent_dim is not None:
            # 최소 필수 특성만 사용 (reset 시)
            if len(kwargs) <= len(self.minimal_features):
                agent_features = []
                for key in self.minimal_features:
                    if key in kwargs:
                        agent_features.append(float(kwargs[key]))
                    else:
                        agent_features.append(0.0)  # 기본값
                
                # 나머지는 0으로 패딩
                while len(agent_features) < self.fixed_agent_dim:
                    agent_features.append(0.0)
                
                return np.array(agent_features[:self.fixed_agent_dim], dtype=np.float32)
        
        # 동적 처리 (기존 방식)
        # 기본 특성들을 우선 처리
        agent_features = []
        processed_keys = set()
        
        for key in self.default_agent_features:
            if key in kwargs:
                agent_features.append(float(kwargs[key]))
                processed_keys.add(key)
        
        # 추가로 전달된 특성들을 알파벳 순으로 추가
        remaining_keys = sorted([k for k in kwargs.keys() if k not in processed_keys])
        for key in remaining_keys:
            agent_features.append(float(kwargs[key]))
            processed_keys.add(key)
        
        return np.array(agent_features, dtype=np.float32)
    
    def _validate_inputs(self, timeseries_data: np.ndarray, **kwargs):
        """입력 데이터 검증"""
        if not self.flag:
            raise ValueError("Must call get_dataset_indices() before using the state")
        
        if timeseries_data.ndim != 2:
            raise ValueError(f"Expected timeseries_data to be 2D array (T, D), got shape {timeseries_data.shape}")
        
        # 필수 에이전트 특성 확인
        required_features = ['current_position', 'execution_strength']
        missing_features = [f for f in required_features if f not in kwargs]
        if missing_features:
            raise ValueError(f"Missing required agent features: {missing_features}")
    
    def __call__(self, timeseries_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        상태 데이터 처리 및 반환
        
        Args:
            timeseries_data: 시계열 데이터 (T, D) 형태
            **kwargs: 에이전트 관련 특성들 (포지션, 실행강도, 시장상태 등)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (시계열 특성, 에이전트 특성)
        """
        # 입력 검증
        self._validate_inputs(timeseries_data, **kwargs)
        
        # 시계열 데이터에서 타겟 특성 추출
        self.timeseries_data = timeseries_data
        self.target_timeseries_data = self._get_target_data(timeseries_data, self.target_indices)
        
        # 에이전트 특성 처리
        self.agent_data = self._process_agent_features(**kwargs)
        
        return self.target_timeseries_data, self.agent_data
    
    def get_state_info(self) -> Dict[str, Any]:
        """현재 상태 정보 반환 (디버깅용)"""
        if not hasattr(self, 'timeseries_data'):
            return {"status": "No state data processed yet"}
        
        return {
            "timeseries_shape": self.timeseries_data.shape,
            "target_timeseries_shape": self.target_timeseries_data.shape,
            "agent_features_count": len(self.agent_data),
            "target_values": self.target_values,
            "target_indices": self.target_indices,
            "agent_data": self.agent_data.tolist(),
            "fixed_agent_dim": self.fixed_agent_dim
        }
    
    def normalize_agent_features(self, features: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        에이전트 특성들을 정규화 (선택사항)
        
        Args:
            features: 특성별 (평균, 표준편차) 딕셔너리
        
        Returns:
            정규화된 에이전트 특성 배열
        """
        if not hasattr(self, 'agent_data'):
            raise ValueError("No agent data to normalize. Call state first.")
        
        if features is None:
            # 기본 정규화 값들 (경험적으로 설정)
            default_normalization = {
                'current_position': (0.0, 1.0),      # -1, 0, 1 값이므로 표준화 불필요
                'execution_strength': (25.0, 15.0),  # 대략적인 평균과 표준편차
                'market_regime': (0.0, 1.0),         # -1, 0, 1
                'volatility_regime': (0.0, 1.0),     # -1, 0, 1  
                'sharpe_ratio': (0.0, 2.0),          # 샤프비율 정규화
                'max_drawdown': (-0.1, 0.05),        # 드로우다운 정규화
                'volatility': (0.02, 0.01),          # 변동성 정규화
                'win_rate': (0.5, 0.2),              # 승률 정규화
                'total_trades': (50.0, 30.0),        # 거래횟수 정규화
                'transaction_cost_ratio': (0.01, 0.005) # 비용비율 정규화
            }
            features = default_normalization
        
        normalized_features = self.agent_data.copy()
        
        # 각 특성별로 정규화 적용 (사용 가능한 것만)
        for i, key in enumerate(self.default_agent_features[:len(self.agent_data)]):
            if key in features:
                mean, std = features[key]
                normalized_features[i] = (normalized_features[i] - mean) / (std + 1e-8)
        
        return normalized_features
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def timeseries_dim(self) -> int:
        """시계열 특성 차원수 반환"""
        return len(self.target_values)
    
    @property 
    def agent_dim(self) -> int:
        """에이전트 특성 차원수 반환"""
        if self.fixed_agent_dim is not None:
            return self.fixed_agent_dim
        if hasattr(self, 'agent_data'):
            return len(self.agent_data)
        return len(self.default_agent_features)  # 기본값
    
    def __repr__(self) -> str:
        status = "initialized" if self.flag else "not initialized"
        return (f"State(target_features={len(self.target_values)}, "
                f"agent_features={self.agent_dim}, status={status})")
    
    def __str__(self) -> str:
        if not self.flag:
            return f"State: Not initialized. Target values: {self.target_values}"
        
        info = self.get_state_info()
        return (f"State Information:\n"
                f"  - Timeseries Shape: {info.get('timeseries_shape', 'N/A')}\n"
                f"  - Target Timeseries Shape: {info.get('target_timeseries_shape', 'N/A')}\n" 
                f"  - Agent Features Count: {info.get('agent_features_count', 'N/A')}\n"
                f"  - Fixed Agent Dim: {info.get('fixed_agent_dim', 'N/A')}\n"
                f"  - Target Values: {self.target_values}\n"
                f"  - Status: Ready")