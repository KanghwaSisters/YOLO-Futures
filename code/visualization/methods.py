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
    ax.set_title('Pnl Over Time')
    ax.set_ylabel('Value')
    ax.set_xlabel('Date')
    apply_string_xticks(ax, timesteps)  # ✅ 문자열 x축 적용
    ax.legend()

def plot_volumes(ax, model_volumes_all):
    for name, series in model_volumes_all.items():
        ax.plot(range(len(series)), series, label=name)
    ax.set_title('Trade Volume Over Time')
    ax.set_ylabel('Volume')
    ax.set_xlabel('Episodes')
    ax.legend()

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