import numpy as np

class DifferentialSharpeRatio:
    def __init__(self, span=300, initial_returns=None, epsilon=1e-8):
        if span < 1:
            raise ValueError("Span must be a positive integer.")
        
        self.eta = 2.0 / (span + 1)
        self.A = 0.0
        self.B = 1.0 # B의 초기값을 1.0으로 설정하여 초기 분산이 0이 되는 것을 방지
        self.epsilon = epsilon
        
        if initial_returns is not None and len(initial_returns) > 1:
            self.A = np.mean(initial_returns)
            self.B = np.mean(np.square(initial_returns))
            # 초기 분산이 너무 작으면 B를 약간 조정
            if (self.B - self.A**2) < self.epsilon:
                self.B = self.A**2 + self.epsilon
        
        print(f"DSR Initialized. A_0: {self.A:.6f}, B_0: {self.B:.6f}, eta: {self.eta:.6f}")

    def __call__(self, current_return):
        prev_A = self.A
        prev_B = self.B
        
        # 1. 분자 공식 수정: prev_B -> (prev_B - prev_A**2)
        variance = prev_B - prev_A**2
        
        # 2. 수치 안정성 강화
        if variance < self.epsilon:
            dsr = 0.0
        else:
            delta_A = current_return - prev_A
            delta_B = current_return**2 - prev_B
            numerator = variance * delta_A - 0.5 * prev_A * delta_B
            denominator = variance**(3/2)
            dsr = numerator / denominator
            
        self.A = prev_A + self.eta * (current_return - prev_A)
        self.B = prev_B + self.eta * (current_return**2 - prev_B)
        
        return dsr if not np.isnan(dsr) else 0.0