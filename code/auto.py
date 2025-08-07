import itertools
import copy
from config import CONFIG
from CTTSmain import main

# 실험하고 싶은 하이퍼파라미터
param_grid = {
    'lr': [1e-3, 1e-4],
    'gamma': [0.95, 0.99],
    'clip_eps': [0.1, 0.2],
    'entropy_coeff': [0.0, 0.01],
}

if '__name__' == __main__():
    # 조합 생성
    keys, values = zip(*param_grid.items())
    experiments = list(itertools.product(*values))

    for i, combination in enumerate(experiments):
        config = copy.deepcopy(CONFIG)
        
        # 하이퍼파라미터 설정
        config.lr = combination[keys.index('lr')]
        config.gamma = combination[keys.index('gamma')]
        config.clip_eps = combination[keys.index('clip_eps')]
        config.entropy_coeff = combination[keys.index('entropy_coeff')]
        
        print(f"\n===== Experiment {i+1}/{len(experiments)}: {dict(zip(keys, combination))} =====\n")

        main(config)