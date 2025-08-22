import numpy as np
from env.DifferentialSharpeRatio import *

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

def risk_adjusted_pnl_reward(alpha=1.0,
                             beta=0.1,
                             gamma=0.3,
                             bonus_scale=0.5,
                             position_change_penalty=-0.01,
                             margin_call_penalty=-1.5,
                             maturity_date_penalty=-0.5,
                             bankrupt_penalty=-3.0,
                             insufficient_penalty=-1.0,
                             risk_penalty=-1.0,
                             scaling_factor=10_000,
                             env_info='',
                             **kwargs):
    
    # 1. 미실현 손익의 변화량 
    delta_unrealized_pnl = (kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']) 

    # 2. 순실현 손익
    net_realized_pnl = kwargs['net_realized_pnl'] 
    
    realized_bonus = gamma * net_realized_pnl if net_realized_pnl > 0 else 0

    # 3. 실현 손익과 미실현 손익을 더한 reward
    reward = (beta*delta_unrealized_pnl + alpha*net_realized_pnl + realized_bonus ) / scaling_factor # + realized_bonus

    # 포지션 청산 시점일 경우
    if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
        reward += (kwargs['equity'] - kwargs['initial_budget']) / kwargs['initial_budget'] * bonus_scale

    # # 4. 포지션 변경 시 패널티 부여 
    # delta_position = kwargs['current_position'] * kwargs['prev_position']
    # if np.sign(delta_position) == -1:
    #     reward += position_change_penalty

    # 마진콜일 때 패널티 부여 
    if env_info == 'margin_call':
        reward += margin_call_penalty

    #  파산일 때 패널티 부여 
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty

    # 7. 만기일 마지막 시점에서 포지션을 들고 있으면 패널티 부여 
    elif (env_info == 'maturity_data') & (kwargs['execution_strength'] != 0):
        reward += maturity_date_penalty

    elif env_info == 'insufficient':
        reward += insufficient_penalty

    elif env_info == 'risk_limits':
        reward += risk_penalty

    return np.tanh(reward)


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



def GOT_pnl_reward(alpha=1.0,
                   beta=0.3,
                   gamma=0.3,
                   bonus_scale=0.5,
                   position_change_penalty=-0.01,
                   margin_call_penalty=-1.5,
                   maturity_date_penalty=-0.5,
                   bankrupt_penalty=-3.0,
                   max_steps_penalty=-1.0,
                   goal_reward_bonus=2.0,
                   scaling_factor=10_000,
                   env_info='',
                   **kwargs):

    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    realized_bonus = net_realized_pnl if net_realized_pnl > 0 else 0

    # 3. 실현 손익과 미실현 손익을 더한 reward
    # reward = (beta*delta_unrealized_pnl + alpha*net_realized_pnl + gamma*realized_bonus ) / scaling_factor # + realized_bonus

    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor

    # 청산 시점 보너스
    if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
        reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor ) * bonus_scale

    # 환경 정보 기반 보너스/패널티
    if env_info == 'margin_call':
        reward += margin_call_penalty
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
        reward += maturity_date_penalty
    # elif env_info == 'max_step':
    #     reward += max_steps_penalty
    elif env_info == 'goal_profit':
        reward += goal_reward_bonus

    return np.tanh(reward)


def GOT_pnl_reward_log(alpha=1.0,
                        beta=0.3,
                        bonus=0.5,
                        margin_call_penalty=-1.0,
                        maturity_date_penalty=-0.5,
                        bankrupt_penalty=-1.0,
                        goal_reward_bonus=1.0,
                        scaling_factor=10_000,
                        env_info='',
                        **kwargs):

    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor 

    # 청산 시점 보너스
    if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
        # reward += ((kwargs['equity'] - kwargs['initial_budget']) / kwargs['initial_budget']) * bonus
        reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor) * bonus

    # 환경 정보 기반 보너스/패널티
    if env_info == 'margin_call':
        reward += margin_call_penalty
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
        reward += maturity_date_penalty

    elif env_info == 'goal_profit':
        reward += goal_reward_bonus

    scaled_reward = np.sign(reward) * np.log1p(abs(reward))

    return np.clip(scaled_reward, -2, 2)

def GOT_tanh_reward_postpenalty(alpha=1.0,
                                beta=0.3,
                                bonus=0.5,
                                margin_call_penalty=-1.0,
                                maturity_date_penalty=-0.5,
                                bankrupt_penalty=-1.0,
                                goal_reward_bonus=1.0,
                                scaling_factor=10_000,
                                env_info='',
                                **kwargs):
    '''tanh + 패널티 적용 후 전체 스케일링'''
    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor 

    # 청산 시점 보너스
    if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
        # reward += ((kwargs['equity'] - kwargs['initial_budget']) / kwargs['initial_budget']) * bonus
        reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor) * bonus

    # 환경 정보 기반 보너스/패널티
    if env_info == 'margin_call':
        reward += margin_call_penalty
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
        reward += maturity_date_penalty
    elif env_info == 'goal_profit':
        reward += goal_reward_bonus

    return np.tanh(reward) 



def GOT_tanh_reward_prepenalty(alpha=1.0,
                                beta=0.3,
                                bonus=0.5,
                                margin_call_penalty=-1.0,
                                maturity_date_penalty=-0.5,
                                bankrupt_penalty=-1.0,
                                goal_reward_bonus=1.0,
                                scaling_factor=10_000,
                                env_info='',
                                **kwargs):
    '''tanh + 보상 스케일링 후 패널티 적용'''
    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor 

    # 청산 시점 보너스
    # if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
    #     reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor) * bonus

    # scaling 
    reward = np.tanh(reward)

    # 환경 정보 기반 보너스/패널티
    # if env_info == 'margin_call':
    #     reward += margin_call_penalty
    # elif env_info == 'bankrupt':
    #     reward += bankrupt_penalty
    # elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
    #     reward += maturity_date_penalty
    # elif env_info == 'goal_profit':
    #     reward += goal_reward_bonus

    return reward 


def GOT_log_reward_postpenalty(alpha=1.0,
                                beta=0.3,
                                bonus=0.5,
                                margin_call_penalty=-1.0,
                                maturity_date_penalty=-0.5,
                                bankrupt_penalty=-1.0,
                                goal_reward_bonus=1.0,
                                scaling_factor=10_000,
                                env_info='',
                                **kwargs):
    '''log1p + 패널티 적용 후 전체 스케일링'''
    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor 

    # 청산 시점 보너스
    # if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
    #     reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor) * bonus

    # 환경 정보 기반 보너스/패널티
    if env_info == 'margin_call':
        reward += margin_call_penalty
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
        reward += maturity_date_penalty
    elif env_info == 'goal_profit':
        reward += goal_reward_bonus

    scaled_reward = np.sign(reward) * np.log1p(abs(reward))

    return np.clip(scaled_reward, -2, 2)

def GOT_log_reward_prepenalty(alpha=1.0,
                                beta=0.3,
                                bonus=0.5,
                                margin_call_penalty=-1.0,
                                maturity_date_penalty=-0.5,
                                bankrupt_penalty=-1.0,
                                goal_reward_bonus=1.0,
                                scaling_factor=10_000,
                                env_info='',
                                **kwargs):
    '''log1p + 보상 스케일링 후 패널티 적용'''
    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl = kwargs['net_realized_pnl']

    # 얼마나 실현했는가? 보너스 [추가]
    # bonus_prop = (net_realized_pnl / (np.abs(delta_unrealized_pnl) + kwargs['eps']))
    
    reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor 
    
    # 청산 시점 보너스
    # if kwargs['current_position'] == 0 and kwargs['prev_position'] != 0:
    #     reward += ((kwargs['equity'] - kwargs['initial_budget']) / scaling_factor) * bonus

    # scaled 
    reward = np.sign(reward) * np.log1p(abs(reward))
    reward = np.clip(reward, -2, 2)

    # 환경 정보 기반 보너스/패널티 
    # if env_info == 'margin_call':
    #     reward += margin_call_penalty
    # elif env_info == 'bankrupt':
    #     reward += bankrupt_penalty
    # elif env_info == 'maturity_data' and kwargs['execution_strength'] != 0:
    #     reward += maturity_date_penalty
    # elif env_info == 'goal_profit':
    #     reward += goal_reward_bonus

    return reward



def GOT_log_reward_entryneutral(alpha=1.0,
                                beta=0.3,
                                bonus=0.5,
                                margin_call_penalty=-1.0,
                                maturity_date_penalty=-0.5,
                                bankrupt_penalty=-1.0,
                                goal_reward_bonus=1.0,
                                scaling_factor=10_000,
                                entry_L=30,          # 첫 진입 후 중립화할 스텝 수
                                entry_weight=1.0,    # 엔트리 구간 가중
                                use_vol_norm=True,   # 변동성 정규화 여부
                                eps=1e-8,
                                env_info='',
                                **kwargs):
    """
    - 첫 진입 후 L 스텝: drift/vol 중립 보상
    - 그 외: 기존 PnL 기반 보상
    """
    delta_unrealized_pnl = kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl']
    net_realized_pnl     = kwargs['net_realized_pnl']
    prev_pos             = kwargs.get('prev_position', 0)
    since_entry          = kwargs.get('since_entry', 10**9)

    # 1) 기본 PnL 보상
    pnl_reward = (beta * delta_unrealized_pnl + alpha * net_realized_pnl) / scaling_factor

    # 2) 엔트리-중립 구간 보상 치환
    if since_entry < entry_L and prev_pos != 0:
        score    = kwargs.get('score', None)
        entry_reward = entry_weight * prev_pos * score
        base_reward = entry_reward
    else:
        base_reward = pnl_reward

    # 3) 환경 보너스/패널티
    if env_info == 'margin_call':
        base_reward += margin_call_penalty
    elif env_info == 'bankrupt':
        base_reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs.get('execution_strength', 0) != 0:
        base_reward += maturity_date_penalty
    elif env_info == 'goal_profit':
        base_reward += goal_reward_bonus

    # 4) 스케일러(로그 압축 유지 or tanh로 대체 가능)
    scaled_reward = np.sign(base_reward) * np.log1p(abs(base_reward))
    return np.clip(scaled_reward, -2, 2)


def reward_per_equity(**kwargs):
    # 현재 잔고 기반  
    # 개똥쓰레기 성능 
    MAX_STEPS = kwargs['max_step']
    current_step = kwargs['current_step']
    equity = kwargs['equity']

    delay_modifier = (current_step / MAX_STEPS)

    reward = equity * delay_modifier + current_step
    reward = np.sign(reward) * np.log1p(abs(reward))

    return reward


DSR = DifferentialSharpeRatio()

def hybrid_reward(w_profit=.5,
                  w_risk=.5,
                  w_regret=.2, # 이건 4에서는 안 쓰고 5
                  scaling_factor=1, # 50_000,
                  margin_call_penalty=-2.0, # -1
                  maturity_date_penalty=-0.5,
                  bankrupt_penalty=-5.0,
                  goal_reward_bonus=2.0,
                  env_info='',
                  **kwargs):
    def log(value):
        # LOG without problems 
        return np.sign(value) * np.log1p(abs(value) + 1e-6)        # 로그 값이 튀는 걸 방지 

    current_pnl = kwargs['realized_pnl']
    previous_pnl = kwargs['realized_pnl'] - kwargs['net_realized_pnl']

    net_pnl = log(current_pnl) - log(previous_pnl)
    unrealized_pnl = log(kwargs['unrealized_pnl']) - log(kwargs['prev_unrealized_pnl'])
    # 종가를 기준으로, 포트폴리오 수익률 지표 
    # 현재 포트폴리오 가치 = 보유 현금 + 미실현 손익 
    portfolio_value = log(kwargs['current_balance']) - log(kwargs['previous_balance'])

    # 종합 수익 지표 
    # R_profit = net_pnl + unrealized_pnl # 3부터 미실현 손익 추가 2까지는 net_pnl만 썼고 그 전에는 log(kwargs['net_realized_pnl'])
    R_profit = log(kwargs['net_realized_pnl']) + log(kwargs['unrealized_pnl'] - kwargs['prev_unrealized_pnl'])
    R_risk = DSR(portfolio_value)                                # DifferentialSharpeRatio (DSR)
    
    # 포지션을 들어가지 않을 때의 후회 
    prev_position = np.sign(kwargs['prev_position'])
    current_position = np.sign(kwargs['current_position'])

    if (prev_position == 0) & (current_position == 0):
        regret = log(abs(kwargs['diff']))
    else:
        regret = 0

    # regret은 4부터 추가함 
    reward = w_profit*R_profit + w_risk*R_risk - w_regret*regret

    if not np.isfinite(reward):
        reward = 0.0

    if env_info == 'margin_call':
        reward += margin_call_penalty
    elif env_info == 'bankrupt':
        reward += bankrupt_penalty
    elif env_info == 'maturity_data' and kwargs.get('execution_strength', 0) != 0:
        reward += maturity_date_penalty
    elif env_info == 'goal_profit':
        reward += goal_reward_bonus

    return np.clip(reward, -1e2, 1e2) 