from easydict import EasyDict
import torch
import torch.optim as optim

from datahandler.scaler import *
from agent.PPOAgent_ms import *
from models.CTTS import *

from trainer.Episodic import *
from trainer.GoalOrTimeoutTrainer import *

from env.reward_ftn import *
from env.done_ftn import *

from env.BasicEnv import *
from env.HorizonBoundEnv import *
from env.GoalOrTimeoutEnv import *

from utils.setDevice import *

# ===================================================================================
# [ 00 ] Select State 
# ===================================================================================
# target_values = ['open', 'high', 'low', 'close', 'vol',
#                 'log_return','return_5', 'return_10', 'volume_change', 'ema_5', 
#                 'ema_20', 'ema_cross', 'cci', 'sar', '%K', 
#                 '%D', 'roc', 'rsi', 'obv', 'ad_line', 
#                 'bb_upper', 'bb_lower', 'bb_width', 'atr', 'gap_size']
# ===================================================================================
# [ 01 ] Scaler 
# ===================================================================================
# 1. RobustScaler()
# 2. MinMaxScaler()
# 3. StandardScaler()
# ===================================================================================
# [ 02 ] Match Trainer : Env  
# ===================================================================================
# 1. NonEpisodicTrainer : FuturesEnvironment 
# - 지속되는 에피소드에서 파산하지 않고 수익을 내며 계약을 유지하자. 
# - 파산을 해도 리셋하지 않고 에피소드를 이어나간다. 
# 2. EpisodicTrainer : FuturesEnvironment
# - 지속되는 에피소드에서 파산하지 않고 수익을 내며 계약을 유지하자. 
# - 파산을 하면 리셋해 시작 타임스텝으로 이동한다. 
# 3. HorizonBoundNonEpisodicTrainer : HorizonBoundEnv
# - 제한된 타임스텝 동안 수익을 내보자. (추가 보상은 없음)
# - 파산을 해도 리셋하지 않고 에피소드를 이어나간다. 
# 4. HorizonBoundEpisodicTrainer : HorizonBoundEnv
# - 제한된 타임스텝 동안 수익을 내보자. (추가 보상은 없음)
# - 파산을 하면 리셋해 시작 타임스텝으로 이동한다. 
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼SWING▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 5. GOTNonEpisodicTrainer : GoalOrTimeoutEnv
# - 제한된 타임스텝 내에서 목표 수익에 도달하면 추가 보상을 준다. ( 스윙 전략 )
# - 파산을 해도 리셋하지 않고 에피소드를 이어나간다. 
# 6. GOTRandomTrainer : GOTRandomEnv
# - 제한된 타임스텝 내에서 목표 수익에 도달하면 추가 보상을 준다. ( 스윙 전략 )
# - 에피소드 별로 데이터를 나누어 시간 종속성이 깨진 랜덤 에피소드를 경험한다. 

# 지민 COMMENT : 3,4는 안 쓰는 거 추천, 스윙으로 넘어가던 과도기 시절에 
# 고안한 방법론이라 보상 체계나 이런게 별로임 
# ===================================================================================
# [ 04 ] Reward Ftn  
# ===================================================================================
# 1. GOT_tanh_reward_postpenalty
# - tanh + 패널티 적용 후 전체 스케일링 
# 2. GOT_tanh_reward_prepenalty
# - tanh + 보상 스케일링 후 패널티 적용 
# 3. GOT_log_reward_postpenalty
# - log1p + 패널티 적용 후 전체 스케일링
# 4. GOT_log_reward_prepenalty
# - log1p + 보상 스케일링 후 패널티 적용
# ===================================================================================
# [ 05 ] Agent  
# ===================================================================================
# 1. PPOAgent 
# - 일반 PPO 에이전트로, 단일 옵티마이저가 Actor와 Critic을 전부 학습한다. 
# 2. DecoupledPPOAgent
# - Actor와 Critic을 구분해 별도의 옵티마이저를 사용한다. 
# 단, 역전파는 통합 loss를 이용한다. 
# ===================================================================================
# [ 07 ] Model   
# ===================================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ CTTS ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 1. MultiStatePV 
# - 가장 baseline이 되는 모델로, Timeseries Data와 Agent info를 
# 별도로 처리해 fusion한다. fusion한 이후 Actor와 Critic으로 분기한다. 
# 2. RegimeAwareMultiStatePV 
# - 단기적인 장 정보를 추가적으로 임베딩한다. 
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ INFORMER ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 1.
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ DLINEAR ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 1.
# ===================================================================================

position_cap = 10

target_values = ['close', 'high', 'low', 'volume_change',
                'ema_5', 'ema_20', 'ema_cross',
                'rsi', '%K', '%D', 'cci',
                'atr', 'bb_width',
                'obv']

# target_values = ['close', 'high', 'low',
#                 'ema_5', 'ema_20', 
#                 'rsi', 'volume_change']     
                 
scaler = RobustScaler() # RobustScaler

device = get_device()

CONFIG = EasyDict({
    # main component. 
    'TRAINER': GOTNonEpisodicTrainer, 
    'ENV': GoalOrTimeoutEnv, 
    'AGENT': DecoupledPPOAgent,
    'NETWORK': RegimeAwareMultiStatePV,
    'REWARD_FTN': GOT_pnl_reward_log, 
    'DONE_FTN': reach_max_step,
    'SCALER': scaler,
    'PATH': 'logs/GOT/log_reward',  # '../logs/RobustDivertedNonepi'
    'DATASET_PATH': 'data/processed/kospi200_ffill_clean_version.pkl', # ../data/processed/kospi200_ffill_clean_version.pkl

    # 기본 설정
    'DEVICE': device,
    'START_BUDGET': 30_000_000,
    'WINDOW_SIZE': 80,
    'N_GROUP': 15,
    'POSITION_CAP': position_cap,
    'TARGET_VALUES': target_values,
    'TRAIN_VALID_TIMESTEP': None, 

    # PPO Agent 설정
    'SINGLE_EXECUTION_CAP' : position_cap,
    'N_ACTIONS': 1+2*position_cap,
    'ACTION_SPACE': list(range(-position_cap, position_cap+1)),
    'GAMMA': 0.99,
    'LR': 3e-4,
    'VALUE_COEFF': 0.5,
    'ENTROPY_COEFF': 0.05,
    'CLIP_EPS': 0.2,
    'BATCH_SIZE': 256,
    'EPOCH': 32,

    # 모델 설정
    'INPUT_DIM': len(target_values),
    'AGENT_INPUT_DIM': 7,
    'EMBED_DIM': 32,
    'KERNEL_SIZE': 4,
    'STRIDE': 1,
    'AGENT_HIDDEN_DIM': 32,
    'AGENT_OUT_DIM': 32,
    'FUSION_HIDDEN_DIM': 64,
    'NUM_LAYERS': 3,
    'NUM_HEADS': 4,
    'D_FF': 64,
    'DROPOUT': 0.1,

    # 학습 관련
    'N_ITERATION' : 5_000,
    'N_STEPS': 2048,
    'MA_INTERVAL': 50,
    'SAVE_INTERVAL': 10,
    'PRINT_LOG_INTERVAL': 1,
    'PRINT_ENV_LOG_INTERVAL': 500
})