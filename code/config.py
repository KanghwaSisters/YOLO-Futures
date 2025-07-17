from easydict import EasyDict
import torch
import torch.optim as optim

CONFIG = EasyDict({
    # 기본 설정
    'MAX_CONTRACT': 3,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'START_BUDGET': 20_000_000,
    'WINDOW_SIZE': 30,
    'POSITION_CAP': 50,
    'TRAIN_VALID_TIMESTEP': None,  # 외부에서 정의해야 함

    # PPO Agent 설정
    'N_ACTIONS': 9,
    'ACTION_SPACE': list(range(-4, 5)),
    'GAMMA': 0.99,
    'LR': 1e-3,
    'VALUE_COEFF': 0.5,
    'ENTROPY_COEFF': 0.01,
    'CLIP_EPS': 0.2,
    'BATCH_SIZE': 32,
    'EPOCH': 10,

    # 모델 설정
    'INPUT_DIM': None,  # 외부에서 target_values로부터 정의
    'AGENT_INPUT_DIM': 2,
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
    'N_STEPS': 100,
    'MA_INTERVAL': 50,
    'SAVE_INTERVAL': 10,

    # 기타
    'REWARD_FTN': None,  # 외부에서 정의
    'DONE_FTN': None,    # 외부에서 정의
    'SCALER': None,      # 외부에서 정의
    'PATH': '/Users/ijimin/Documents/GitHub/YOLO-Futures/result',
})