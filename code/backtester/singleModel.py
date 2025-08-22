from backtester.backTester import *

def _load_config_from_setting(setting_path: str, path_dir_override: Optional[str]=None) -> EasyDict:
    parser = BackTester.__new__(BackTester)  # 인스턴스 없이 파서 메서드 사용
    cfg_text = open(setting_path, "r", encoding="utf-8").read()
    cfg_dict = BackTester.parse_config_text(parser, cfg_text, resolver=_build_resolver())
    cfg = EasyDict(cfg_dict)

    # 상하관계 정합성: setting.txt가 포함된 디렉터리를 PATH로 강제 보정 (요청 반영)
    setting_dir = os.path.dirname(os.path.abspath(setting_path))
    cfg.PATH = path_dir_override or setting_dir
    return cfg

def _build_resolver() -> Dict[str, Any]:
    resolver = {}
    import env.reward_ftn as reward_mod
    import env.done_ftn as done_mod
    for name, obj in reward_mod.__dict__.items():
        if callable(obj): resolver[name] = obj
    for name, obj in done_mod.__dict__.items():
        if callable(obj): resolver[name] = obj
    return resolver


# ===== SingleModel: net_pnl.csv(총 PnL 최대) 또는 지정 경로 모델로 valid별 N회 평균 측정 =====
class SingleModel(BackTester):
    def _pick_from_netpnl_csv(self, root_dir: str) -> str:
        """
        root_dir/net_pnl.csv 를 읽어 'total' 행이 가장 큰 컬럼을 선택하고,
        그 컬럼명(=가중치 파일명)을 models/ 아래 경로로 변환해서 반환.
        """
        csv_path = os.path.join(root_dir, "net_pnl.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"net_pnl.csv not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0)
        # 우선 'total' 행 사용, 없으면 'mean' 사용
        key_row = "total" if "total" in df.index else ("mean" if "mean" in df.index else None)
        if key_row is None:
            raise ValueError("net_pnl.csv needs a 'total' or 'mean' row to select best model.")

        # 가장 큰 값을 갖는 컬럼 = 가중치 파일명
        col = df.loc[key_row].astype(float).idxmax()
        model_path = os.path.join(root_dir, "models", col)
        if not os.path.isfile(model_path):
            # 혹시 csv 컬럼명이 경로 또는 확장자가 다른 경우를 대비한 보조 탐색
            candidates = glob.glob(os.path.join(root_dir, "models", "*"))
            # 파일명만 비교
            base = os.path.basename(col)
            for c in candidates:
                if os.path.basename(c) == base:
                    model_path = c
                    break
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file inferred from net_pnl.csv not found: {model_path}")
        print(f"[Best from net_pnl.csv] {os.path.basename(model_path)}")
        return model_path

    def run(self, root_dir: str, model_path: Optional[str] = None, n_runs: int = 30):
        """
        (기본) net_pnl.csv 기준 best 모델 → 각 valid env마다 n_runs회 평가 평균을 CSV로 저장.
        (옵션) model_path 제공 시 해당 모델 사용.
        CSV 파일명 = 선택된 가중치 파일명(.csv로 확장자 변경)
        열 = [avg_pnl, avg_sharpe, avg_mdd, avg_winrate], 행 = valid index
        """
        # 1) 모델 경로 결정
        if model_path is None:
            model_path = self._pick_from_netpnl_csv(root_dir)
        else:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")

        device = getattr(self, "device", "cpu")
        state_dict = torch.load(model_path, map_location=device)

        # 2) valid list
        env_labels = list(range(len(self.valid_timestep)))

        # 3) 결과 컨테이너
        avg_pnls, avg_sharpes, avg_mdds, avg_wr_totals = [], [], [], []

        # 4) valid별 n_runs 반복
        for i, time_interval in enumerate(self.valid_timestep):
            pnl_list, sharpe_list, mdd_list, wr_total_list = [], [], [], []
            print(f"[Run] valid {i} — {n_runs} runs")

            env = self.set_env(time_interval_valid=time_interval)

            for r in range(n_runs):
                net_pnl, wr_long, wr_short, wr_total, sharpe_ratio, max_drawdown = \
                    self.valid(env=env, agent=self.agent, model_name=os.path.basename(model_path), state_dict=state_dict)

                pnl_list.append(net_pnl)
                sharpe_list.append(sharpe_ratio)
                mdd_list.append(max_drawdown)            # 음수 비율
                wr_total_list.append(wr_total)           # 0~1

            avg_pnls.append(int(np.nanmean(pnl_list)))
            avg_sharpes.append(round(float(np.nanmean(sharpe_list)),2))
            avg_mdds.append(round(float(np.nanmean(mdd_list)),2))
            avg_wr_totals.append(round(float(np.nanmean(wr_total_list)),2))

            print(f"  -> avg_pnl={avg_pnls[-1]:,.0f} ₩ | avg_sharpe={avg_sharpes[-1]:.2f} | "
                  f"avg_mdd={avg_mdds[-1]*100:.2f}% | avg_winrate={avg_wr_totals[-1]*100:.1f}%")

        # 5) 저장
        out_df = pd.DataFrame({
            "avg_pnl": avg_pnls,
            "avg_sharpe": avg_sharpes,
            "avg_mdd": avg_mdds,
            "avg_winrate": avg_wr_totals,
        }, index=env_labels)

        base_name = os.path.splitext(os.path.basename(model_path))[0] + ".csv"
        out_path = os.path.join(root_dir, base_name)
        out_df.to_csv(out_path, encoding="utf-8-sig")
        print(f"[Saved SingleModel] {out_path}")


# ---------- 메인 ----------
def main_single_backTester(entry: Optional[str]=None, model_path=None, n_runs=30):
    """
    entry가 디렉터리면 그 안의 setting.txt 사용.
    entry가 파일이면 그 파일을 setting.txt로 사용.
    entry가 없으면 ./setting.txt 사용.
    """
    if entry is None:
        entry = "setting.txt"

    if os.path.isdir(entry):
        setting_path = os.path.join(entry, "setting.txt")
    else:
        setting_path = entry

    if not os.path.isfile(setting_path):
        raise FileNotFoundError(f"setting.txt not found: {setting_path}")

    # 1) CONFIG 로드(경로는 setting.txt 부모 디렉토리로 고정)
    CONFIG = _load_config_from_setting(setting_path)

    # 2) 데이터셋 로드
    with open(CONFIG.DATASET_PATH, "rb") as f:
        df = pickle.load(f)

    # 3) TRAIN_VALID_TIMESTEP 확보(없으면 백업 분할)
    if "TRAIN_VALID_TIMESTEP" not in CONFIG or not CONFIG.TRAIN_VALID_TIMESTEP:
        print("[Info] TRAIN_VALID_TIMESTEP not in setting.txt — creating fallback splits.")
        CONFIG.TRAIN_VALID_TIMESTEP = _even_split_timesteps(df.index, n_group=CONFIG.N_GROUP, train_ratio=0.9)

    # 4) state / model / agent 구성
    state = State(CONFIG.TARGET_VALUES)
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
        dropout=CONFIG.DROPOUT,
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
        device=CONFIG.DEVICE,
    )

    # 5) SingleModel 생성 후 실행
    sm = SingleModel(
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
        path=CONFIG.PATH,      # ← setting.txt가 들어있는 디렉터리(예: 28_scaling)
    )

    sm.run(root_dir=CONFIG.PATH, model_path=model_path, n_runs=n_runs)
