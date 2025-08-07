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

from env.env import *
from env.GoalOrTimeoutEnv import *

from utils.setDevice import *


# ===================================================================================
# target_values = ['open', 'high', 'low', 'close', 'vol',
#                 'log_return','return_5', 'return_10', 'volume_change', 'ema_5', 
#                 'ema_20', 'ema_cross', 'cci', 'sar', '%K', 
#                 '%D', 'roc', 'rsi', 'obv', 'ad_line', 
#                 'bb_upper', 'bb_lower', 'bb_width', 'atr', 'gap_size']
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
    'TRAINER': GOTRandomTrainer, # GOTRandomTrainer,  # HorizonBoundNonEpisodicTrainer, GOTNonEpisodicTrainer
    'ENV': GOTRandomEnv, # GOTRandomEnv,    # FuturesEnvironment, GoalOrTimeoutEnv
    'AGENT': DecoupledPPOAgent, # DecoupledPPOAgent
    'NETWORK': RegimeAwareMultiStatePV,
    'REWARD_FTN': GOT_pnl_reward_log, # GOT_pnl_reward,
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