import numpy as np

def reward_unrealized_pnl(**kwargs):
    return kwargs['unrealized_pnl']

def reward_unrealized_pnl_diff_log(**kwargs):
    curr = kwargs['unrealized_pnl']
    prev = kwargs['prev_unrealized_pnl']
    
    # 분모가 너무 작거나 0이면 계산하지 않음
    if abs(prev) < 1e-8:
        return 0.0
    
    # 로그 계산값이 실수이며 안전할 때만 수행
    ratio = curr / prev
    
    # log(음수) 방지
    if ratio <= 0:
        return 0.0
    
    # 이전보다 악화된 경우에만 보상 부여
    if abs(curr) < abs(prev):
        return float(np.log(ratio))
    
    return 0.0

def reward_sharpe_ratio(**kwargs):
    """
    Sharpe ratio를 reward로 사용하는 함수
    risk_metrics 객체에서 sharpe ratio를 가져와서 반환
    """
    risk_metrics = kwargs.get('risk_metrics', None)
    if risk_metrics is None:
        return 0.0
    
    sharpe = risk_metrics.get_sharpe_ratio()
    
    # Sharpe ratio가 음수이거나 너무 큰 값일 때 클리핑
    sharpe = np.clip(sharpe, -5.0, 5.0)
    
    return float(sharpe)

def reward_sharpe_ratio_scaled(**kwargs):
    """
    스케일링된 Sharpe ratio를 reward로 사용하는 함수
    더 안정적인 학습을 위해 스케일링 적용
    """
    risk_metrics = kwargs.get('risk_metrics', None)
    if risk_metrics is None:
        return 0.0
    
    sharpe = risk_metrics.get_sharpe_ratio()
    
    # Sharpe ratio 정규화 (tanh 함수 사용으로 -1 ~ 1 사이로 스케일링)
    scaled_sharpe = np.tanh(sharpe / 2.0)
    
    return float(scaled_sharpe)

def reward_combined_pnl_sharpe(**kwargs):
    """
    미실현 손익과 Sharpe ratio를 결합한 reward 함수
    단기 수익성과 장기 안정성을 모두 고려
    """
    unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
    risk_metrics = kwargs.get('risk_metrics', None)
    
    # PnL 기반 보상 (정규화)
    pnl_reward = unrealized_pnl / 1000000  # 100만원 기준으로 정규화
    
    # Sharpe ratio 기반 보상
    sharpe_reward = 0.0
    if risk_metrics is not None:
        sharpe = risk_metrics.get_sharpe_ratio()
        sharpe_reward = np.tanh(sharpe / 2.0)
    
    # 가중 결합 (PnL 70%, Sharpe 30%)
    combined_reward = 0.7 * pnl_reward + 0.3 * sharpe_reward
    
    return float(combined_reward)

def risk_adjusted_pnl_reward(
                            hold_over_penalty=-0.01, 
                            margin_call_penalty=-1.0, 
                            maturity_date_penalty=-0.5,
                            bankrupt_penalty=-2.0, 
                            insufficient_penalty=-0.5,
                            risk_penalty=-1.0,
                            initial_budget=1_000_000,
                            env_info='',
                            **kwargs
                        ):
    
    # 1. 미실현 손익의 변화량 
    delta_unrealized_pnl = (kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']) 

    # 2. 실현 손익의 변화량 
    realized_pnl = kwargs['realized_pnl'] 

    # 3. 실현 손익과 미실현 손익을 더한 reward
    reward = (delta_unrealized_pnl + realized_pnl) / initial_budget

    # 4. 장기 보유 시 패널티 부여
    # if kwargs['unrealized_pnl'] != 0:
    #     reward += hold_over_penalty

    # 5. 마진콜일 때 패널티 부여 
    if env_info == 'margin_call':
        reward += margin_call_penalty

    # 6. 파산일 때 패널티 부여 
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty

    # 7. 만기일일 때 패널티 부여 
    elif env_info == 'maturity_data':
        reward += maturity_date_penalty

    elif env_info == 'insufficient':
        reward += insufficient_penalty

    elif env_info == 'risk_limits':
        reward += risk_penalty

    return reward 


def pnl_change_based_reward(margin_call_penalty=-10.0, 
                            bankrupt_penalty=-10.0,
                            maturity_date_penalty=-10.0,
                            env_info='',
                            initial_budget=1_000_000, 
                            **kwargs):

    pnl_change = (kwargs['current_price'] - kwargs['pev_price']) * kwargs['execution_strength']
    reward = pnl_change + (kwargs['realized_pnl'] / initial_budget)

    # 5. 마진콜일 때 패널티 부여 
    if env_info == 'margin_call':
        reward += margin_call_penalty

    # 6. 파산일 때 패널티 부여 
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty

    # 7. 만기일일 때 패널티 부여 
    elif env_info == 'maturity_data':
        reward += maturity_date_penalty

    return reward