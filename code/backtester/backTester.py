# run_all_valid.py
import os, re, ast, glob, copy, pickle, importlib, sys
from typing import Any, Dict, Optional, List, Tuple
import inspect
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from easydict import EasyDict

# ===== 프로젝트 모듈 =====
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
from env.BasicEnv import *
from state.state import *

from utils.setDevice import *
from utils.timestepRelated import *
from visualization.methods import *

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- 유틸: 시간 구간 자동 분할(백업용) ----------
def _even_split_timesteps(index: pd.DatetimeIndex, n_group: int, train_ratio: float=0.9):
    """
    setting.txt에 TRAIN_VALID_TIMESTEP가 없을 때 쓸 간단한 백업 분할기.
    연속성 고려 없이 균등한 인덱스 분할로 (train_range, valid_range) 튜플의 리스트를 만든다.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    # 고유한 날짜 단위로 먼저 나눔
    dates = pd.Index(sorted(index.normalize().unique()))
    chunks = np.array_split(dates, n_group)
    pairs = []
    for ch in chunks:
        if len(ch) < 2:
            continue
        n_train = max(1, int(len(ch) * train_ratio))
        train_days = ch[:n_train]
        valid_days = ch[n_train:]
        if len(valid_days) == 0:
            continue
        train_range = (train_days[0].strftime("%Y-%m-%d"), train_days[-1].strftime("%Y-%m-%d"))
        valid_range = (valid_days[0].strftime("%Y-%m-%d"), valid_days[-1].strftime("%Y-%m-%d"))
        pairs.append((train_range, valid_range))
    return pairs


# =========================
# BackTester
# =========================
class BackTester:
    # 정규표현식 (setting 파서용)
    LINE_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*:\s*(.+?)\s*$")
    CLASS_RE = re.compile(r"^<\s*class\s*'([^']+)'\s*>$")
    FUNC_RE  = re.compile(r"^<\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s+at\s+0x[0-9A-Fa-f]+>\s*$")
    OBJ_RE   = re.compile(r"^<\s*([A-Za-z_][A-Za-z0-9_\.]*)\s+object\s+at\s+0x[0-9A-Fa-f]+>\s*$")

    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap,
                 agent, model, optimizer, device, n_steps, path):
        # dataframe & env 파라미터
        self.df = df
        self.env = env
        self.train_valid_timestep = train_valid_timestep
        _, self.valid_timestep = zip(*self.train_valid_timestep)
        self.window_size = window_size
        self.state = state
        self.reward_ftn = reward_ftn
        self.done_ftn = done_ftn
        self.start_budget = start_budget

        if scaler and inspect.isclass(scaler):
            self.scaler = scaler()
        else:
            self.scaler = scaler

        self.position_cap = position_cap

        # agent/model
        self.agent = agent
        self.agent.set_optimizer(optimizer)
        self.valid_agent = copy.deepcopy(self.agent)
        self.model = model

        # etc
        self.device = device
        self.n_steps = n_steps
        self.path = path

    @staticmethod
    def _sharpe_from_equity(
        equity_curve,
        risk_free_annual: float = 0.0,
        steps_per_year: Optional[float] = None,
        index_like: Optional["pd.DatetimeIndex"] = None,
        mode: str = "daily",  # "daily" | "bar"
    ):
        """
        mode="daily": equity를 날짜별 마지막 값으로 리샘플 → 일간 수익률 Sharpe(연율화, *sqrt(252))
        mode="bar":   기존 바 단위 Sharpe (연율화는 steps_per_year로 *sqrt)
        """
        import numpy as np
        import pandas as pd

        eq = np.asarray(equity_curve, dtype=float)
        if eq.size < 2:
            return 0.0

        # ===== DAILY MODE =====
        if mode == "daily" and index_like is not None and len(index_like) >= len(eq):
            idx = pd.DatetimeIndex(index_like[: len(eq)])
            ser = pd.Series(eq, index=idx)
            daily_eq = ser.resample("1D").last().dropna()
            if daily_eq.size < 2:
                return 0.0

            prev = np.clip(daily_eq.shift(1), 1e-9, None)
            rets = (daily_eq - prev) / prev
            rets = rets.dropna().replace([np.inf, -np.inf], np.nan).dropna()
            if rets.size < 2:
                return 0.0

            rf_daily = risk_free_annual / 252.0
            excess = rets - rf_daily
            mu = float(excess.mean())
            sd = float(excess.std(ddof=1))
            if not np.isfinite(sd) or sd < 1e-10:
                return 0.0

            sharpe = mu / sd
            sharpe *= np.sqrt(252.0)  # 연율화
            return float(np.clip(sharpe, -10.0, 10.0))

        # ===== BAR MODE (기존) =====
        prev = np.clip(eq[:-1], 1e-9, None)
        rets = (eq[1:] - prev) / prev
        rets = rets[np.isfinite(rets)]
        if rets.size < 2:
            return 0.0

        rf_per_step = (risk_free_annual / steps_per_year) if (steps_per_year and steps_per_year > 0) else 0.0
        excess = rets - rf_per_step
        mu = float(np.mean(excess))
        sd = float(np.std(excess, ddof=1))
        if not np.isfinite(sd) or sd < 1e-10:
            return 0.0

        sharpe = mu / sd
        if steps_per_year and steps_per_year > 0:
            sharpe *= np.sqrt(steps_per_year)
        return float(np.clip(sharpe, -10.0, 10.0))

    @staticmethod
    def _max_drawdown_from_equity(equity_curve):
        """
        Max Drawdown = min( (E_t - cummax(E_t)) / cummax(E_t) )
        음수 비율(-0.x) 반환
        """
        import numpy as np
        eq = np.asarray(equity_curve, dtype=float)
        if eq.size < 2:
            return 0.0
        eq = np.where(eq <= 0, np.min(eq[eq > 0]) if np.any(eq > 0) else 1.0, eq)
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / cummax
        return float(np.clip(dd.min(), -1.0, 0.0))

    @staticmethod
    def _estimate_steps_per_year(dt_index: pd.DatetimeIndex, trading_days_per_year: int = 252) -> float:
        """
        인덱스에서 '하루 평균 바 수'를 추정해 steps/year를 근사.
        - 날짜별 카운트를 내서 평균 bars/day 계산
        - steps_per_year = bars_per_day * trading_days_per_year
        """
        if not isinstance(dt_index, pd.DatetimeIndex):
            dt_index = pd.DatetimeIndex(dt_index)

        days = dt_index.normalize()
        counts = pd.Series(1, index=days).groupby(level=0).sum()
        if len(counts) == 0:
            return float(trading_days_per_year)
        bars_per_day = counts.mean()
        return float(bars_per_day * trading_days_per_year)

    # ----- helper -----
    @staticmethod
    def split_position_strength(action: int) -> Tuple[int, int]:
        if action > 0:   return 1, int(action)
        if action < 0:   return -1, int(-action)
        return 0, 0

    @staticmethod
    def get_equity(account):
        return float(account.available_balance + account.margin_deposit + account.unrealized_pnl)

    def set_env(self, time_interval_valid: tuple):
        return self.env(
            full_df=self.df,
            date_range=time_interval_valid,
            window_size=self.window_size,
            state_type=self.state,
            reward_ftn=self.reward_ftn,
            done_ftn=self.done_ftn,
            start_budget=self.start_budget,
            n_actions=self.agent.n_actions,
            scaler=self.scaler,
            position_cap=self.position_cap,
        )

    def get_valid_env_list(self, valid_timestep):
        return [self.set_env(t) for t in valid_timestep]

    # ----- core valid -----
    def valid(self, env, agent, model_name, state_dict):
        def cal_wr(lst): return round(sum(lst) / len(lst), 2) if len(lst) else 0.0

        agent.load_model(state_dict)
        n_win_long, n_win_short, n_win_total = [], [], []
        net_pnl = 0.0
        sr, mdd = 0.0, 0.0
        n = 0

        state = env.reset()

        idx = getattr(getattr(env, "dataset", None), "cleaned_df", None)
        if idx is None:
            idx = getattr(getattr(env, "base_dataset", None), "cleaned_df", None)
        ep_index = idx.index if idx is not None else pd.DatetimeIndex([])
        steps_per_year = self._estimate_steps_per_year(ep_index)

        # 텐서화 (tuple 상태 지원)
        if isinstance(state, tuple):
            ts_state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
            agent_state = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
            state = (ts_state, agent_state)

        while not env.dataset.reach_end(env.current_timestep):
            done = False
            current_position = 0

            # 에피소드 단위 equity/시간 기록 시작
            episode_equity = [self.get_equity(env.account)]
            episode_times  = [env.current_timestep]

            while not done:
                previous_position = current_position

                mask = getattr(env, "mask", None)
                action, _, _ = agent.get_action(state, mask)  # stochastic=True는 선택사항
                next_state, _, done = env.step(action)
                current_position, _ = self.split_position_strength(int(action))

                if isinstance(next_state, tuple):
                    ts_state = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                    agent_state = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
                    next_state = (ts_state, agent_state)

                state = next_state

                # 실현손익 발생 시 승/패 집계: '이전 포지션' 기준으로 기록 (정확)
                if getattr(env.account, "net_realized_pnl", 0) != 0:
                    win = int(env.account.net_realized_pnl > 0)
                    if previous_position == 1:    n_win_long.append(win)
                    elif previous_position == -1: n_win_short.append(win)
                    n_win_total.append(win)

                # equity/시간 기록
                episode_equity.append(self.get_equity(env.account))
                episode_times.append(env.current_timestep)

            net_pnl += int(getattr(env.account, "realized_pnl", 0.0))

            # Sharpe/MDD 계산: 일간 리샘플 기반 Sharpe
            ep_sr  = self._sharpe_from_equity(
                episode_equity,
                risk_free_annual=0.0,
                steps_per_year=steps_per_year,
                index_like=pd.DatetimeIndex(episode_times),
                mode="daily",
            )
            # MDD는 환경이 주는 값 대신 로컬 equity로 계산해도 됨 (가독성↑)
            ep_mdd = self._max_drawdown_from_equity(episode_equity)

            # 길이 가중 평균(에피소드 길이 = 포인트-1)
            m = max(len(episode_equity) - 1, 0)
            if m > 0:
                sr  += (m / (n + m)) * (ep_sr  - sr)
                mdd += (m / (n + m)) * (ep_mdd - mdd)
                n   += m

            # 다음 에피소드 대비 초기화 
            env.account.reset()
            env.performance_tracker.reset()
            env.risk_metrics.reset()

        return int(net_pnl), cal_wr(n_win_long), cal_wr(n_win_short), cal_wr(n_win_total), sr, mdd

    # ----- weights loop & save -----
    def _list_weight_files(self, models_dir: str) -> List[str]:
        files = []
        for ext in ("*.pt", "*.pth", "*.bin"):
            files.extend(glob.glob(os.path.join(models_dir, ext)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No weight files found in: {models_dir}")
        return files

    def run_all_models_and_save_csvs(self, root_dir: str):
        models_dir = os.path.join(root_dir, "models")
        weight_files = self._list_weight_files(models_dir)

        valid_env_list = self.get_valid_env_list(self.valid_timestep)
        env_labels = list(range(len(self.valid_timestep)))

        pnl_index   = env_labels + ["mean", "total"]
        wr_index    = env_labels + ["mean"]

        df_pnl      = pd.DataFrame(index=pnl_index)
        df_wr_long  = pd.DataFrame(index=wr_index)
        df_wr_short = pd.DataFrame(index=wr_index)
        df_wr_total = pd.DataFrame(index=wr_index)
        df_md       = pd.DataFrame(index=wr_index)
        df_sr       = pd.DataFrame(index=wr_index)

        device = getattr(self, "device", "cpu")

        for wpath in weight_files:
            print(f"Start Validation {wpath}")
            col = os.path.basename(wpath)
            state_dict = torch.load(wpath, map_location=device)

            pnl_vals, wrL_vals, wrS_vals, wrT_vals, sharpe_ratios, max_drawdowns = [], [], [], [], [], []

            for idx, env in enumerate(valid_env_list):
                net_pnl, wr_long, wr_short, wr_total, sharpe_ratio, max_drawdown = self.valid(env=env, agent=self.agent, model_name=col, state_dict=state_dict)
                print(f"[{idx:2}] PnL : {net_pnl:12,.0f} ₩ | total win rate : {wr_total*100:3.0f}% | long win rate : {wr_long*100:3.0f}% | short win rate : {wr_short*100:3.0f}% | mdd :{max_drawdown*100:4.0f}% | sharpe ratio :{sharpe_ratio:3.2f}")

                pnl_vals.append(int(net_pnl))
                wrL_vals.append(float(wr_long))
                wrS_vals.append(float(wr_short))
                wrT_vals.append(float(wr_total))
                sharpe_ratios.append(float(sharpe_ratio))
                max_drawdowns.append(float(max_drawdown))

            df_pnl.loc[env_labels, col]      = pnl_vals
            df_wr_long.loc[env_labels, col]  = wrL_vals
            df_wr_short.loc[env_labels, col] = wrS_vals
            df_wr_total.loc[env_labels, col] = wrT_vals
            df_md.loc[env_labels, col]       = max_drawdowns
            df_sr.loc[env_labels, col]       = sharpe_ratios

            df_pnl.loc["mean",  col] = np.nanmean(pnl_vals)
            df_pnl.loc["total", col] = np.nansum(pnl_vals)
            df_wr_long.loc["mean",  col] = np.nanmean(wrL_vals)
            df_wr_short.loc["mean", col] = np.nanmean(wrS_vals)
            df_wr_total.loc["mean", col] = np.nanmean(wrT_vals)
            df_md.loc["mean", col]       = np.nanmean(max_drawdowns)
            df_sr.loc["mean", col]       = np.nanmean(sharpe_ratios)

            print(f"Cumulated PnL : {sum(pnl_vals):12,.0f} ₩")

        out_pnl      = os.path.join(root_dir, "net_pnl.csv")
        out_wr_long  = os.path.join(root_dir, "win_rate_long.csv")
        out_wr_short = os.path.join(root_dir, "win_rate_short.csv")
        out_wr_total = os.path.join(root_dir, "win_rate_total.csv")
        out_md       = os.path.join(root_dir, "max_drawdown.csv")
        out_sr       = os.path.join(root_dir, "sharpe_ratio.csv")

        df_pnl.to_csv(out_pnl, encoding="utf-8-sig")
        df_wr_long.to_csv(out_wr_long, encoding="utf-8-sig")
        df_wr_short.to_csv(out_wr_short, encoding="utf-8-sig")
        df_wr_total.to_csv(out_wr_total, encoding="utf-8-sig")
        df_md.to_csv(out_md, encoding="utf-8-sig")
        df_sr.to_csv(out_sr, encoding="utf-8-sig")

        print(f"[Saved]\n  {out_pnl}\n  {out_wr_long}\n  {out_wr_short}\n  {out_wr_total}\n {out_sr}\n {out_md}")

    # ----- visuals -----
    def generate_valid_visuals(self, root_dir: str, init_capital=30_000_000):
        os.makedirs(os.path.join(root_dir, "valid_vis"), exist_ok=True)
        vis_dir = os.path.join(root_dir, "valid_vis")

        paths = {
            "pnl": os.path.join(root_dir, "net_pnl.csv"),
            "wr_long": os.path.join(root_dir, "win_rate_long.csv"),
            "wr_short": os.path.join(root_dir, "win_rate_short.csv"),
            "wr_total": os.path.join(root_dir, "win_rate_total.csv"),
        }

        # 최고의 mean 모델 텍스트로 저장
        tops = {}
        for key, p in paths.items():
            df = pd.read_csv(p, index_col=0)
            mean_row = df.loc["mean"]
            best_model = mean_row.astype(float).idxmax()
            best_value = float(mean_row[best_model])
            tops[key] = (best_model, best_value)

        txt_path = os.path.join(vis_dir, "top_models.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# Top models by highest mean (column name = weight file)\n")
            f.write(f"PNL (mean): {tops['pnl'][0]} | value={tops['pnl'][1]:.6f}\n")
            f.write(f"WinRate-Long (mean):  {tops['wr_long'][0]} | value={tops['wr_long'][1]:.6f}\n")
            f.write(f"WinRate-Short (mean): {tops['wr_short'][0]} | value={tops['wr_short'][1]:.6f}\n")
            f.write(f"WinRate-Total (mean): {tops['wr_total'][0]} | value={tops['wr_total'][1]:.6f}\n")
        print(f"[Saved] {txt_path}")

        def save_heatmap(data: pd.DataFrame, title: str, out_path: str, value_fmt="%.2f"):
            fig, ax = plt.subplots(figsize=(max(8, data.shape[1]*0.6), max(6, data.shape[0]*0.4)))
            im = ax.imshow(data.values, aspect="auto")
            ax.set_title(title)
            ax.set_xticks(np.arange(data.shape[1])); ax.set_xticklabels(data.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(data.shape[0])); ax.set_yticklabels(data.index)

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data.iat[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, value_fmt % val, ha="center", va="center", fontsize=8)

            cbar = fig.colorbar(im, ax=ax, cmap="bwr"); cbar.ax.set_ylabel("value", rotation=270, labelpad=14)
            plt.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
            print(f"[Saved] {out_path}")

        # PnL 퍼센트 히트맵
        pnl_df = pd.read_csv(paths["pnl"], index_col=0)
        env_rows = [r for r in pnl_df.index if r not in ("mean", "total")]
        pnl_env = pnl_df.loc[env_rows].astype(float)
        pnl_pct = pnl_env / float(init_capital) * 100.0
        save_heatmap(pnl_pct, f"Net PnL (% of {init_capital:,} KRW)", os.path.join(vis_dir, "pnl_percent_heatmap.png"))

        # Winrate 히트맵
        for key, title, fname in [
            ("wr_long",  "Win Rate (Long)",  "winrate_long_heatmap.png"),
            ("wr_short", "Win Rate (Short)", "winrate_short_heatmap.png"),
            ("wr_total", "Win Rate (Total)", "winrate_total_heatmap.png"),
        ]:
            wr_df = pd.read_csv(paths[key], index_col=0)
            env_rows = [r for r in wr_df.index if r != "mean"]
            wr_env = wr_df.loc[env_rows].astype(float)
            save_heatmap(wr_env, title, os.path.join(vis_dir, fname))

    # ======================
    # setting.txt 파서
    # ======================
    @staticmethod
    def _import_by_qualname(qualname: str):
        parts = qualname.split('.')
        for i in range(len(parts)-1, 0, -1):
            mod_name = '.'.join(parts[:i]); attr_path = parts[i:]
            try:
                mod = importlib.import_module(mod_name)
                obj = mod
                for a in attr_path:
                    obj = getattr(obj, a)
                return obj
            except Exception:
                continue
        return None

    def _parse_value(self, raw: str, resolver: Optional[Dict[str, Any]] = None) -> Any:
        raw = raw.strip()
        m = self.CLASS_RE.match(raw)
        if m:
            qual = m.group(1)
            cls = self._import_by_qualname(qual)
            return cls if cls is not None else qual

        m = self.FUNC_RE.match(raw)
        if m:
            fname = m.group(1)
            if resolver and fname in resolver:
                return resolver[fname]
            return fname

        m = self.OBJ_RE.match(raw)
        if m:
            qual = m.group(1)
            cls = self._import_by_qualname(qual)
            return cls if cls is not None else qual

        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw

    def parse_config_text(self, text: str, resolver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('=====') or line.startswith('#'):
                continue
            m = self.LINE_RE.match(line)
            if not m: 
                continue
            key, raw_val = m.group(1), m.group(2)
            cfg[key] = self._parse_value(raw_val, resolver=resolver)
        return cfg


# ---------- setting.txt -> CONFIG ----------
def _build_resolver() -> Dict[str, Any]:
    resolver = {}
    import env.reward_ftn as reward_mod
    import env.done_ftn as done_mod
    for name, obj in reward_mod.__dict__.items():
        if callable(obj): resolver[name] = obj
    for name, obj in done_mod.__dict__.items():
        if callable(obj): resolver[name] = obj
    return resolver

def _load_config_from_setting(setting_path: str, path_dir_override: Optional[str]=None) -> EasyDict:
    parser = BackTester.__new__(BackTester)  # 인스턴스 없이 파서 메서드 사용
    cfg_text = open(setting_path, "r", encoding="utf-8").read()
    cfg_dict = BackTester.parse_config_text(parser, cfg_text, resolver=_build_resolver())
    cfg = EasyDict(cfg_dict)

    # 상하관계 정합성: setting.txt가 포함된 디렉터리를 PATH로 강제 보정 (요청 반영)
    setting_dir = os.path.dirname(os.path.abspath(setting_path))
    cfg.PATH = path_dir_override or setting_dir
    return cfg


# ---------- 메인 ----------
def main_backTester(entry: Optional[str]=None):
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

    # 5) BackTester 생성 후 실행
    bt = BackTester(
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

    bt.run_all_models_and_save_csvs(root_dir=CONFIG.PATH)
    bt.generate_valid_visuals(root_dir=CONFIG.PATH)
    print("[Done] Artifacts saved under:", CONFIG.PATH)

def change_cwd(new_directory):
    import os

    # 1. 현재 작업 디렉토리 확인 (변경 전)
    original_directory = os.getcwd()
    print(f"변경 전 CWD: {original_directory}")

    try:
        # 3. 작업 디렉토리 변경
        os.chdir(new_directory)

        # 4. 현재 작업 디렉토리 확인 (변경 후)
        current_directory = os.getcwd()
        print(f"🎉 변경 후 CWD: {current_directory}")

    except FileNotFoundError:
        print(f"🚫 오류: '{new_directory}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"🚫 오류 발생: {e}")

if __name__ == "__main__":

    base_directory = '/home/tonnonssi/YOLO-Futures'
    # change_cwd(base_directory)

    target_directory = base_directory + '/logs/GOT_KL_hybrid/55_scaling_MultiState_6'

    main_backTester(target_directory)