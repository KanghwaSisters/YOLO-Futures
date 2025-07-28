import numpy as np
import pandas as pd
import matplotlib.dates as mdates

def apply_string_xticks(ax, timesteps, n_ticks=6):
    idxs = np.linspace(0, len(timesteps) - 1, n_ticks, dtype=int)
    ticklabels = pd.to_datetime(timesteps).strftime('%Y-%m').values
    ax.set_xticks(idxs)
    ax.set_xticklabels([ticklabels[i] for i in idxs], rotation=45)

def plot_market_with_actions(ax, name, timesteps, market, actions, reset_point):
    ax.plot(range(len(market)), market, label='Market Price', color='black')
    colors = np.where(actions > 0, 'red', np.where(actions < 0, 'blue', 'gray'))
    labels = { 'red': 'Long', 'blue': 'Short', 'gray': 'Hold' }
    plotted_labels = set()
    for t, a in enumerate(actions):
        color = colors[t]
        label = labels[color] if color not in plotted_labels else None
        ax.scatter(t, market[t], color=color, alpha=min(abs(a) / 10, 1.0), s=30, label=label)
        plotted_labels.add(color)
    ax.axvline(reset_point, linestyle='--', color='black', alpha=0.3, label='Init Recharge Point')
    ax.set_title(f'Market Flow with Actions : {name}')
    ax.set_ylabel('Market')
    ax.set_xlabel('Date')
    apply_string_xticks(ax, timesteps)
    ax.legend()

def plot_model_returns(ax, timesteps, model_returns_all, reset_point):
    for name, series in model_returns_all.items():
        ax.plot(timesteps, series, label=name)
    ax.axvline(timesteps[reset_point], linestyle='--', color='black', alpha=0.3)
    ax.set_title('Model-wise Realized Returns')
    ax.set_ylabel('Return')
    ax.set_xlabel('Date')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

def plot_pnls(ax, timesteps, model_pnls_all, reset_point):
    for name, series in model_pnls_all.items():
        ax.plot(range(len(series)), series, label=name)
    ax.axvline(reset_point, linestyle='--', color='black', alpha=0.3, label='Init Recharge Point')
    ax.set_title('Cumulated Realized Pnl Over Time')
    ax.set_ylabel('Value')
    ax.set_xlabel('Date')
    apply_string_xticks(ax, timesteps)  
    ax.legend()

def plot_volumes(ax, model_volumes_all):
    styles = {
        'latest model': {'color': '#4E79A7', 'marker': 'o'},
        'highest reward': {'color': '#F28E2B', 'marker': 's'},
        'per steps': {'color': '#59A14F', 'marker': '^'},
    }

    for name, series in model_volumes_all.items():
        x = range(len(series))
        style = styles.get(name, {'marker': 'x'})  # 기본 스타일
        ax.plot(x, series, label=name, linewidth=2, **style)

    ax.set_title('Trade Volume Over Time', fontsize=13, weight='bold')
    ax.set_ylabel('Volume')
    ax.set_xlabel('Episodes')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

def plot_rewards(ax, model_rewards_all):
    for name, series in model_rewards_all.items():
        ax.plot(range(len(series)), series, label=name)
    ax.set_title('Cumulative Rewards Over Time')
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Episodes')
    ax.legend()

def plot_durations(ax, model_durations_all):
    for name, series in model_durations_all.items():
        ax.plot(range(len(series)), series, label=name)
    ax.set_title('Maintained Duration Over Time')
    ax.set_ylabel('Cumulative Steps')
    ax.set_xlabel('Episodes')
    ax.legend()

def plot_maintained_length(ax, len_lst):
    x = list(range(len(len_lst)))
    ax.bar(x, len_lst)
    ax.set_xlabel("Index")
    ax.set_ylabel("Minutes")
    ax.set_title("Maintained Length")
    ax.grid(axis='y')

def plot_both_pnl_ticks(ax, timesteps, pnls):
    unrealized_pnl, realized_pnl = zip(*pnls)
    width = 0.4  # 막대 폭

    # 양수면 빨강, 음수면 파랑
    unrealized_colors = ['red' if v >= 0 else 'blue' for v in realized_pnl]
    realized_colors   = ['lightcoral' if v >= 0 else 'lightblue' for v in unrealized_pnl]

    # 막대 위치 조정 (좌우로 나란히)
    x = list(range(len(timesteps)))
    x1 = [i - width/2 for i in x]
    x2 = [i + width/2 for i in x]

    # 막대그래프 그리기
    ax.bar(x1, unrealized_pnl, width=width, color=unrealized_colors, label='Unrealized PnL')
    ax.bar(x2, realized_pnl, width=width, color=realized_colors, label='Realized PnL')

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("PnL")
    ax.set_title("Unrealized vs Realized PnL")
    ax.legend()
    ax.grid(axis='y')


def plot_rewards_with_ma(ax, model_rewards_all, ma_window=10):
    for name, series in model_rewards_all.items():
        x = range(len(series))
        ax.plot(x, series, label=f"{name} (raw)", alpha=0.3, linewidth=1)  # 원래 시계열 (얇게)
        
        # 이동 평균 계산
        ma = np.convolve(series, np.ones(ma_window)/ma_window, mode='valid')
        ax.plot(range(ma_window - 1, len(series)), ma, label=f"{name} (MA{ma_window})", linewidth=2.5)

    ax.set_title('Cumulative Rewards Over Time')
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Episodes')
    ax.legend()