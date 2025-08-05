import pickle
import warnings

from config import CONFIG
from easydict import EasyDict

from env.env import *
from state.state import *
from agent.PPOAgent_ms import *
from models.CTTS import *
from trainer.nonEpisodic import *
from trainer.Episodic import *
from datahandler.scaler import *
from utils.setDevice import *
from utils.timestepRelated import *
from visualization.methods import *

warnings.filterwarnings("ignore", category=FutureWarning)

def main(CONFIG):

    # load dataset 
    with open(CONFIG.DATASET_PATH, 'rb') as f:
        df = pickle.load(f)

    # timestep 분리 
    train_valid_timestep = split_date_ranges_by_group(df.index, n_group=CONFIG.N_GROUP, train_ratio=0.9) 
    CONFIG.TRAIN_VALID_TIMESTEP = train_valid_timestep 

    state =  State(CONFIG.TARGET_VALUES)

    model = CONFIG.NETWORK(
        input_dim=CONFIG.INPUT_DIM,
        agent_input_dim=CONFIG.AGENT_INPUT_DIM,
        embed_dim=CONFIG.EMBED_DIM,
        kernel_size=CONFIG.KERNEL_SIZE,
        stride=CONFIG.STRIDE,
        action_size=CONFIG.N_ACTIONS,
        device=CONFIG.DEVICE,
        agent_hidden_dim=CONFIG.AGENT_HIDDEN_DIM,
        agent_out_dim=CONFIG.AGENT_OUT_DIM,
        fusion_hidden_dim=CONFIG.FUSION_HIDDEN_DIM,
        num_layers=CONFIG.NUM_LAYERS,
        num_heads=CONFIG.NUM_HEADS,
        d_ff=CONFIG.D_FF,
        dropout=CONFIG.DROPOUT
    )

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
        print_env_log_interval=CONFIG.PRINT_ENV_LOG_INTERVAL
    )

    trainer.save(CONFIG)
    trainer()
    

if __name__ == '__main__':
    main(CONFIG)