import pickle
import warnings
import torch.optim as optim

from Informer_config import CONFIG
from easydict import EasyDict

from env.BasicEnv import *
from state.state import *
from agent.PPOAgent_ms import *
from models.MultiInformer import *
from trainer.nonEpisodic import *
from trainer.Episodic import *
from datahandler.scaler import *
from utils.setDevice import *
from utils.timestepRelated import *
from visualization.methods import *

warnings.filterwarnings("ignore", category=FutureWarning)

def main(CONFIG):
    """
    MultiInformer를 사용한 강화학습 트레이딩 메인 함수
    """
    print("=== MultiInformer Trading System 시작 ===")
    print(f"Device: {CONFIG.DEVICE}")
    print(f"Target values: {CONFIG.TARGET_VALUES}")
    print(f"Window size: {CONFIG.WINDOW_SIZE}")

    # 데이터셋 로드
    print("데이터셋 로딩 중...")
    with open(CONFIG.DATASET_PATH, 'rb') as f:
        df = pickle.load(f)
    
    print(f"데이터 크기: {df.shape}")
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")

    # timestep 분리 
    train_valid_timestep = split_date_ranges_by_group(df.index, n_group=CONFIG.N_GROUP, train_ratio=0.9) 
    CONFIG.TRAIN_VALID_TIMESTEP = train_valid_timestep 

    # State 객체 생성
    state = State(CONFIG.TARGET_VALUES)

    # MultiInformerPV 모델 생성
    print("MultiInformerPV 모델 생성 중...")
    model = CONFIG.NETWORK(
        # MultiInformer 파라미터
        enc_in=CONFIG.ENC_IN,
        dec_in=CONFIG.DEC_IN,
        c_out=CONFIG.C_OUT,
        seq_len=CONFIG.SEQ_LEN,
        label_len=CONFIG.LABEL_LEN,
        d_model=CONFIG.D_MODEL,
        n_heads=CONFIG.N_HEADS,
        e_layers=CONFIG.E_LAYERS,
        d_layers=CONFIG.D_LAYERS,
        d_ff=CONFIG.D_FF,
        factor=CONFIG.FACTOR,
        dropout=CONFIG.DROPOUT,
        attn=CONFIG.ATTN,
        embed=CONFIG.EMBED,
        activation=CONFIG.ACTIVATION,
        device=CONFIG.DEVICE,
        # Fusion 파라미터
        agent_input_dim=CONFIG.AGENT_INPUT_DIM,
        agent_hidden_dim=CONFIG.AGENT_HIDDEN_DIM,
        agent_out_dim=CONFIG.AGENT_OUT_DIM,
        fusion_hidden_dim=CONFIG.FUSION_HIDDEN_DIM,
        action_size=CONFIG.N_ACTIONS
    ).to(CONFIG.DEVICE)

    # 모델 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")

    # Agent 생성
    print("Agent 생성 중...")
    agent = CONFIG.AGENT(
        action_space=CONFIG.ACTION_SPACE,
        n_actions=CONFIG.N_ACTIONS,
        model=model,
        value_coeff=CONFIG.VALUE_COEFF,
        entropy_coeff=CONFIG.ENTROPY_COEFF,
        clip_eps=CONFIG.CLIP_EPS,
        gamma=CONFIG.GAMMA,
        lr=CONFIG.LR,
        batch_size=CONFIG.BATCH_SIZE,
        epoch=CONFIG.EPOCH,
        device=CONFIG.DEVICE
    )

    # Trainer 생성
    print("Trainer 생성 중...")
    trainer = CONFIG.TRAINER( 
        df=df,
        env=CONFIG.ENV,
        train_valid_timestep=CONFIG.TRAIN_VALID_TIMESTEP,
        window_size=CONFIG.WINDOW_SIZE,
        state=state,
        reward_ftn=CONFIG.REWARD_FTN,
        done_ftn=CONFIG.DONE_FTN,
        start_budget=CONFIG.START_BUDGET,
        scaler=CONFIG.SCALER,
        position_cap=CONFIG.POSITION_CAP,
        agent=agent,
        model=model,
        optimizer=optim.Adam,
        device=CONFIG.DEVICE,
        n_steps=CONFIG.N_STEPS,
        ma_interval=CONFIG.MA_INTERVAL,
        save_interval=CONFIG.SAVE_INTERVAL,
        path=CONFIG.PATH,
        print_log_interval=CONFIG.PRINT_LOG_INTERVAL,
        print_env_log_interval=CONFIG.PRINT_ENV_LOG_INTERVAL,
        n_iteration=CONFIG.N_ITERATION
    )

    # 설정 저장
    print("설정 저장 중...")
    trainer.save(CONFIG)
    
    # 학습 시작
    print("=== 학습 시작 ===")
    try:
        trainer()
        print("=== 학습 완료 ===")
    except Exception as e:
        print(f"학습 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    

if __name__ == '__main__':
    main(CONFIG)