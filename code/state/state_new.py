import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

class MarketRegime(Enum):
    BULL = 1
    BEAR = -1
    SIDEWAYS = 0

class State:
    def __init__(self, target_values: List[str], indices_list: Optional[List[str]] = None):
        """
        Initialize State class for handling timeseries and agent data.
        
        Args:
            target_values: List of feature names to extract from timeseries data
            indices_list: List of all available feature names in the dataset
        """
        self.target_values = target_values
        self._indices_list = indices_list
        
        # State tracking for enhanced functionality
        self._previous_states = []
        self._state_history_limit = 100
        
        if indices_list is None:
            self.flag = False
        else:
            self.get_dataset_indices(indices_list)

    def get_dataset_indices(self, indices_list: List[str]) -> None:
        """
        Map target_values to their indices in the dataset.
        
        Args:
            indices_list: List of all available feature names
        """
        self.flag = True
        self._indices_list = indices_list

        # Safe indexing
        try:
            self.target_indices = [self._indices_list.index(val) for val in self.target_values]
            
        except ValueError as e:
            raise ValueError(f"target_values must exist in indices_list. {e}")

    def _get_target_data(self, data: np.ndarray, target_indices: List[int]) -> np.ndarray:
        """Extract target features from timeseries data."""
        return data[:, target_indices]

    def _detect_market_regime(self, timeseries_data: np.ndarray) -> MarketRegime:
        """
        Detect current market regime based on price action.
        
        Args:
            timeseries_data: Timeseries data with shape (T, D)
            
        Returns:
            MarketRegime enum value
        """
        if 'close' in self.target_values:
            close_idx = self.target_values.index('close')
            close_prices = timeseries_data[:, close_idx]
            
            if len(close_prices) >= 20:
                # Calculate short and long term trends
                short_ma = np.mean(close_prices[-5:])
                long_ma = np.mean(close_prices[-20:])
                
                # Calculate price momentum
                price_change = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
                
                # Determine regime
                if short_ma > long_ma * 1.02 and price_change > 0.03:
                    return MarketRegime.BULL
                elif short_ma < long_ma * 0.98 and price_change < -0.03:
                    return MarketRegime.BEAR
                else:
                    return MarketRegime.SIDEWAYS
        
        return MarketRegime.SIDEWAYS

    def _calculate_volatility_regime(self, timeseries_data: np.ndarray) -> str:
        """
        Calculate current volatility regime.
        
        Args:
            timeseries_data: Timeseries data with shape (T, D)
            
        Returns:
            Volatility regime: 'low', 'normal', or 'high'
        """
        if 'close' in self.target_values and len(timeseries_data) >= 20:
            close_idx = self.target_values.index('close')
            close_prices = timeseries_data[:, close_idx]
            
            # Calculate returns
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Calculate rolling volatility
            if len(returns) >= 10:
                volatility = np.std(returns[-10:])
                
                # Define volatility thresholds
                if volatility < 0.01:
                    return 'low'
                elif volatility > 0.03:
                    return 'high'
                else:
                    return 'normal'
        
        return 'normal'

    def _get_enhanced_agent_features(self, **kwargs) -> Dict[str, Any]:
        """
        Create enhanced agent features including market context.
        
        Args:
            **kwargs: Agent state variables
            
        Returns:
            Dictionary of enhanced agent features
        """
        enhanced_features = dict(kwargs)
        
        # Add market regime if timeseries data is available
        if hasattr(self, 'timeseries_data') and self.timeseries_data is not None:
            enhanced_features['market_regime'] = self._detect_market_regime(self.timeseries_data)
            enhanced_features['volatility_regime'] = self._calculate_volatility_regime(self.timeseries_data)
        
        # Add position concentration risk
        if 'position' in kwargs and 'budget' in kwargs:
            position_ratio = abs(kwargs['position']) / max(kwargs['budget'], 1)
            enhanced_features['position_concentration'] = position_ratio
        
        # Add momentum indicators
        if hasattr(self, '_previous_states') and len(self._previous_states) > 0:
            # Calculate agent state momentum
            if len(self._previous_states) >= 3:
                prev_agent_data = self._previous_states[-1][1]  # Previous agent data
                current_agent_data = np.array([kwargs[k] for k in sorted(kwargs.keys())], dtype=np.float32)
                
                if len(prev_agent_data) == len(current_agent_data):
                    momentum = np.mean(current_agent_data - prev_agent_data)
                    enhanced_features['agent_momentum'] = momentum
        
        return enhanced_features

    def __call__(self, timeseries_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process timeseries and agent data to return state representation.
        
        Args:
            timeseries_data: Timeseries data with shape (T, D)
            **kwargs: Agent state variables
            
        Returns:
            Tuple of (timeseries_features, agent_features)
        """
        if self.flag == False:
            raise ValueError("Must get target indices before calling")
        
        if timeseries_data.ndim != 2:
            raise ValueError("Expected timeseries_data to be 2D array (T, D)")

        self.timeseries_data = timeseries_data
        self.target_timeseries_data = self._get_target_data(timeseries_data, self.target_indices)

        # Get enhanced agent features
        enhanced_kwargs = self._get_enhanced_agent_features(**kwargs)
        
        # Ordered agent data (excluding non-numeric features)
        numeric_kwargs = {k: v for k, v in enhanced_kwargs.items() 
                         if isinstance(v, (int, float, np.number))}
        
        self.agent_keys = sorted(numeric_kwargs.keys())
        self.agent_data = np.array([numeric_kwargs[k] for k in self.agent_keys], dtype=np.float32)
        
        # Store non-numeric features separately for potential use in reward functions
        self.market_context = {k: v for k, v in enhanced_kwargs.items() 
                              if k not in numeric_kwargs}

        # Update state history
        current_state = (self.target_timeseries_data.copy(), self.agent_data.copy())
        self._previous_states.append(current_state)
        
        # Limit history size
        if len(self._previous_states) > self._state_history_limit:
            self._previous_states.pop(0)

        return self.target_timeseries_data, self.agent_data

    def get_market_context(self) -> Dict[str, Any]:
        """
        Get current market context for use in reward functions.
        
        Returns:
            Dictionary containing market regime and other contextual information
        """
        return getattr(self, 'market_context', {})

    def get_state_history(self, n_steps: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get recent state history.
        
        Args:
            n_steps: Number of recent steps to return
            
        Returns:
            List of (timeseries_features, agent_features) tuples
        """
        return self._previous_states[-n_steps:] if len(self._previous_states) >= n_steps else self._previous_states

    def reset_history(self) -> None:
        """Reset state history."""
        self._previous_states = []

    @property
    def name(self) -> str:
        return self.__class__.__name__
        
    @property
    def feature_names(self) -> List[str]:
        """Get names of target features."""
        return self.target_values
        
    @property
    def agent_feature_names(self) -> List[str]:
        """Get names of agent features."""
        return getattr(self, 'agent_keys', [])
        
    def __repr__(self) -> str:
        return f"State(target_values={self.target_values}, initialized={self.flag})"