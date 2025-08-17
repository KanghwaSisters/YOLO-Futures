from easydict import EasyDict
import torch
import torch.optim as optim

from datahandler.scaler import *
from agent.PPOAgent_ms import *
from models.CTTS import *
from models.MultiInformer import *

from trainer.Episodic import *
from trainer.GoalOrTimeoutTrainer import *

from env.reward_ftn import *
from env.done_ftn import *

from env.BasicEnv import *
from env.HorizonBoundEnv import *
from env.GoalOrTimeoutEnv import *

from utils.setDevice import *

position_cap = 10

target_values = ['close', 'high', 'low', 'volume_change',
                'ema_5', 'ema_20', 'ema_cross',
                'rsi', '%K', '%D', 'cci',
                'atr', 'bb_width',
                'obv']
                 
scaler = RobustScaler()
device = get_device()

CONFIG = EasyDict({
    # main component. 
    'TRAINER': GOTNonEpisodicTrainer, 
    'ENV': GoalOrTimeoutEnv, 
    'AGENT': DecoupledPPOAgent,
    'NETWORK': MultiInformerPV,
    'REWARD_FTN': GOT_pnl_reward_log, 
    'DONE_FTN': reach_max_step,
    'SCALER': scaler,
    'PATH': 'logs/GOT/112',
    'DATASET_PATH': 'data/processed/kospi200_ffill_clean_version.pkl',

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

    # MultiInformer 모델 설정
    'ENC_IN': len(target_values),
    'DEC_IN': len(target_values),
    'C_OUT': 1,
    'SEQ_LEN': 80,  # WINDOW_SIZE와 일치
    'LABEL_LEN': 40,  # SEQ_LEN의 절반
    'D_MODEL': 32,  # 작은 값으로 시작 (메모리 절약)
    'N_HEADS': 4,
    'E_LAYERS': 2,  # 레이어 수 줄임
    'D_LAYERS': 1,
    'D_FF': 64,  # 작은 값으로 시작
    'DROPOUT': 0.1,
    'FACTOR': 5,
    'ATTN': 'prob',
    'EMBED': 'timeF',
    'ACTIVATION': 'gelu',
    
    # Agent와 Fusion 파라미터
    'AGENT_INPUT_DIM': 8,  # agent state 차원
    'AGENT_HIDDEN_DIM': 32,
    'AGENT_OUT_DIM': 32,
    'FUSION_HIDDEN_DIM': 64,

    # 학습 관련
    'N_ITERATION': 5_000,
    'N_STEPS': 2048,
    'MA_INTERVAL': 50,
    'SAVE_INTERVAL': 10,
    'PRINT_LOG_INTERVAL': 1,
    'PRINT_ENV_LOG_INTERVAL': 500
})