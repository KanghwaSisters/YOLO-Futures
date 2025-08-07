import itertools
import copy
from config import CONFIG
from CTTSmain import main
from env.reward_ftn import *

# 실험하고 싶은 하이퍼파라미터
# param_grid = {
#     'lr': [1e-3, 1e-4],
#     'gamma': [0.95, 0.99],
#     'clip_eps': [0.1, 0.2],
#     'entropy_coeff': [0.0, 0.01],
# }

name_value = {
    'tanh_reward_postpenalty': GOT_tanh_reward_postpenalty,
    'tanh_reward_prepenalty' : GOT_tanh_reward_prepenalty,
    'log_reward_postpenalty' : GOT_log_reward_postpenalty, 
    'log_reward_prepenalty' : GOT_log_reward_prepenalty
    
}

if __name__ == "__main__":
    for i, (name, ftn) in enumerate(name_value.items()):
        config = copy.deepcopy(CONFIG)

        # 경로 설정 
        base_dir = 'logs/GOT_NE/'
        directory = base_dir + name

        # 하이퍼파라미터 설정
        config.REWARD_FTN = ftn
        config.PATH = directory

        print(f"\n===== Experiment {i+1}/{len(name_value)}: {name} {ftn} =====\n")

        main(config)


# if '__name__' == __main__:
#     # 조합 생성
#     keys, values = zip(*param_grid.items())
#     experiments = list(itertools.product(*values))

#     for i, combination in enumerate(experiments):
#         config = copy.deepcopy(CONFIG)
        
#         # 하이퍼파라미터 설정
#         config.lr = combination[keys.index('lr')]
#         config.gamma = combination[keys.index('gamma')]
#         config.clip_eps = combination[keys.index('clip_eps')]
#         config.entropy_coeff = combination[keys.index('entropy_coeff')]
        
#         print(f"\n===== Experiment {i+1}/{len(experiments)}: {dict(zip(keys, combination))} =====\n")

#         main(config)