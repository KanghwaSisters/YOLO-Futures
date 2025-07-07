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