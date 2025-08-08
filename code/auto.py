import itertools
import copy
from config import CONFIG
from CTTSmain import main
from env.reward_ftn import *
from models.CTTS import *

import itertools
import copy
from config import CONFIG


# 실험하고 싶은 하이퍼파라미터
param_grid = {
    'model': [RegimeAwareMultiStatePV, DropRegimeNet],
    'lr': [3e-4, 0.00025],
    'reward_ftn': [GOT_log_reward_postpenalty, GOT_tanh_reward_prepenalty],
}

def format_run_name(model, reward_ftn, lr):
    model_name = model.__name__
    reward_name = reward_ftn.__name__
    lr_str = f"L{lr:.0e}" if lr < 1e-2 else f"L{lr:.4f}".replace(".", "")
    return f"{model_name}_{reward_name}_{lr_str}"

if __name__ == '__main__':
    keys, values = zip(*param_grid.items())
    experiments = list(itertools.product(*values))

    for i, combination in enumerate(experiments):
        config = copy.deepcopy(CONFIG)

        # 하이퍼파라미터 unpack
        lr = combination[keys.index('lr')]
        model = combination[keys.index('model')]
        reward_ftn = combination[keys.index('reward_ftn')]

        # 이름 생성
        run_name = format_run_name(model, reward_ftn, lr)

        # 경로 설정
        base_dir = 'logs/GOT_TEST/'
        log_dir = base_dir + run_name

        # 설정 적용
        config.LR = lr
        config.NETWORK = model
        config.REWARD_FTN = reward_ftn
        config.PATH = log_dir

        print(f"\n===== Experiment {i+1}/{len(experiments)}: {run_name} =====\n")

        main(config)

# name_value = {
#     # 'tanh_reward_postpenalty': GOT_tanh_reward_postpenalty,
#     # 'tanh_reward_prepenalty' : GOT_tanh_reward_prepenalty,
#     # 'log_reward_postpenalty' : GOT_log_reward_postpenalty, 
#     'log_reward_prepenalty' : GOT_log_reward_prepenalty
    
# }

# if __name__ == "__main__":
#     for i, (name, ftn) in enumerate(name_value.items()):
#         config = copy.deepcopy(CONFIG)

#         # 경로 설정 
#         base_dir = 'logs/GOT_NE/'
#         directory = base_dir + name

#         # 하이퍼파라미터 설정
#         config.REWARD_FTN = ftn
#         config.PATH = directory

#         print(f"\n===== Experiment {i+1}/{len(name_value)}: {name} {ftn} =====\n")

#         main(config)
