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

def risk_adjusted_pnl_reward(hold_over_penalty=-0.05, 
                             margin_call_penalty=-10.0, 
                             maturity_date_penalty=-10.0,
                             bankrupt_penalty=-10.0, 
                             initial_budget=1_000_000, 
                             env_info='',
                             **kwargs):
    
    # 1. 미실현 손익의 변화량 
    delta_unrealized_pnl = (kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']) 

    # 2. 실현 손익의 변화량 
    realized_pnl = kwargs['realized_pnl'] 

    # 3. 실현 손익과 미실현 손익을 더한 reward
    reward = (delta_unrealized_pnl + realized_pnl) / initial_budget

    # 4. 장기 보유 시 패널티 부여
    if kwargs['unrealized_pnl'] != 0:
        reward += hold_over_penalty

    # 5. 마진콜일 때 패널티 부여 
    if env_info == 'margin_call':
        reward += margin_call_penalty

    # 6. 파산일 때 패널티 부여 
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty

    # 7. 만기일일 때 패널티 부여 
    elif env_info == 'maturity_date':
        reward += maturity_date_penalty

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
    elif env_info == 'maturity_date':
        reward += maturity_date_penalty

    return reward