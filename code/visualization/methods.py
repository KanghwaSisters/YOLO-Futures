import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def apply_string_xticks(ax, timesteps, n_ticks=6):
    idxs = np.linspace(0, len(timesteps) - 1, n_ticks, dtype=int)
    ticklabels = pd.to_datetime(timesteps).strftime('%Y-%m').values
    ax.set_xticks(idxs)
    ax.set_xticklabels([ticklabels[i] for i in idxs], rotation=45)

def plot_market_with_actions(ax, name, timesteps, market, actions, reset_point, cumulative_actions):
    x = np.arange(len(market))
    
    # 배경 색상 (누적 행동 기반)
    for i in range(len(cumulative_actions)):
        alpha = min(abs(cumulative_actions[i]) / 10, 1.0)
        color = 'red' if cumulative_actions[i] > 0 else 'blue' if cumulative_actions[i] < 0 else None
        if color:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=alpha * 0.1, zorder=0)

    # 시장 가격 라인
    ax.plot(x, market, label='Market Price', color='black', linewidth=1.2)

    # 행동 점 시각화
    colors = np.where(actions > 0, 'red', np.where(actions < 0, 'blue', 'gray'))
    labels = {'red': 'Long', 'blue': 'Short', 'gray': 'Hold'}
    plotted_labels = set()
    for t, a in enumerate(actions):
        color = colors[t]
        label = labels[color] if color not in plotted_labels else None
        edgecolor = 'black'
        ax.scatter(t, market[t], color=color, edgecolors=edgecolor, marker='^', 
                   alpha=min(abs(a) / 10, 1.0), s=40, label=label, zorder=3)
        plotted_labels.add(color)

    # 리셋 지점 표시
    # ax.axvline(reset_point, linestyle='--', color='black', alpha=0.3, label='Init Recharge Point', zorder=2)

    # 기타 설정
    ax.set_title(f'Market Flow with Actions : {name}')
    ax.set_ylabel('Market')
    ax.set_xlabel('Date')
    ax.set_xticks(np.linspace(0, len(timesteps)-1, min(10, len(timesteps))))
    ax.set_xticklabels([str(timesteps[int(i)])[:10] for i in np.linspace(0, len(timesteps)-1, min(10, len(timesteps)))], rotation=45)
    ax.legend(loc='best')
    ax.grid(True)

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

    # 색상: Realized는 진하게, Unrealized는 연하게
    unrealized_colors = ['lightcoral' if v >= 0 else 'lightblue' for v in unrealized_pnl]
    realized_colors   = ['red' if v >= 0 else 'blue' for v in realized_pnl]

    # 막대 위치 조정
    x = list(range(len(timesteps)))
    x1 = [i - width/2 for i in x]  # Unrealized
    x2 = [i + width/2 for i in x]  # Realized

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
def plot_equity_curves(ax, timesteps, model_equities_all, reset_point, start_budget):
    """자산 변화 곡선 - 실제 거래 성과를 가장 명확히 보여줌"""
    for name, equity_series in model_equities_all.items():
        # 수익률로 변환 (백분율)
        returns = [(eq / start_budget - 1) * 100 for eq in equity_series]
        ax.plot(range(len(returns)), returns, label=f'{name}', linewidth=2)
    
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, label='Break-even')
    ax.axvline(reset_point, linestyle='--', color='red', alpha=0.5, label='Reset Point')
    ax.set_title('Portfolio Equity Curves (Total Return %)')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Time Steps')
    apply_string_xticks(ax, timesteps)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_position_tracking(ax, timesteps, actions, reset_point):
    """포지션 상태 추적 - Long/Short/Hold 상태 변화"""
    # 액션을 포지션으로 변환 (-1: Short, 0: Hold, 1: Long)
    positions = np.sign(actions)
    
    # 색상 매핑
    colors = ['blue' if p == -1 else 'gray' if p == 0 else 'red' for p in positions]
    
    # 포지션별로 영역 채우기
    for i in range(len(positions)):
        if positions[i] == 1:  # Long
            ax.fill_between([i, i+1], 0, 1, color='red', alpha=0.3)
        elif positions[i] == -1:  # Short  
            ax.fill_between([i, i+1], -1, 0, color='blue', alpha=0.3)
        # Hold은 0 라인 주변 (자동으로 비어있음)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(reset_point, linestyle='--', color='black', alpha=0.5, label='Reset Point')
    ax.set_title('Position Tracking Over Time')
    ax.set_ylabel('Position (Red: Long, Blue: Short)')
    ax.set_xlabel('Time Steps')
    ax.set_ylim(-1.2, 1.2)
    apply_string_xticks(ax, timesteps)
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.3, label='Long Position'),
                      Patch(facecolor='gray', alpha=0.3, label='Hold'),
                      Patch(facecolor='blue', alpha=0.3, label='Short Position')]
    ax.legend(handles=legend_elements)

def plot_drawdown_analysis(ax, timesteps, model_equities_all, reset_point, start_budget):
    """드로우다운 분석 - 손실 구간과 회복 과정 시각화"""
    for name, equity_series in model_equities_all.items():
        # 누적 최대값 계산
        running_max = np.maximum.accumulate(equity_series)
        # 드로우다운 계산 (백분율)
        drawdown = [(eq - peak) / peak * 100 for eq, peak in zip(equity_series, running_max)]
        
        ax.fill_between(range(len(drawdown)), drawdown, 0, 
                       alpha=0.3, label=f'{name} Drawdown')
        ax.plot(range(len(drawdown)), drawdown, label=f'{name}', linewidth=1.5)
    
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(reset_point, linestyle='--', color='red', alpha=0.5, label='Reset Point')
    ax.set_title('Drawdown Analysis (Risk Visualization)')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Time Steps')
    apply_string_xticks(ax, timesteps)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 최대 드로우다운 표시
    for name, equity_series in model_equities_all.items():
        running_max = np.maximum.accumulate(equity_series)
        drawdown = [(eq - peak) / peak * 100 for eq, peak in zip(equity_series, running_max)]
        max_dd_idx = np.argmin(drawdown)
        max_dd_val = min(drawdown)
        ax.annotate(f'Max DD: {max_dd_val:.1f}%', 
                   xy=(max_dd_idx, max_dd_val), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

def plot_training_curves(ax, train_rewards, train_losses):
    """학습 곡선 - 훈련 중 보상과 손실 변화"""
    ax2 = ax.twinx()  # 두 번째 y축 생성
    
    # 보상 곡선 (왼쪽 y축)
    episodes = range(len(train_rewards))
    line1 = ax.plot(episodes, train_rewards, 'b-', alpha=0.3, label='Episode Rewards')
    
    # 이동평균으로 트렌드 표시
    if len(train_rewards) > 10:
        window = min(50, len(train_rewards) // 10)
        moving_avg = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(train_rewards)), moving_avg, 'b-', 
               linewidth=2, label=f'Reward MA({window})')
    
    # 손실 곡선 (오른쪽 y축, None 값 제거)
    valid_losses = [(i, loss) for i, loss in enumerate(train_losses) if loss is not None]
    if valid_losses:
        loss_episodes, loss_values = zip(*valid_losses)
        line2 = ax2.plot(loss_episodes, loss_values, 'r-', alpha=0.6, label='Training Loss')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Training Progress: Rewards vs Loss')
    
    # 범례 합치기
    lines1, labels1 = ax.get_legend_handles_labels()
    if valid_losses:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend(loc='upper left')
    
    ax.grid(True, alpha=0.3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_action_distribution_heatmap(ax, timesteps, actions, n_actions=21):
    """액션 분포 히트맵 - 시간에 따른 액션 선택 패턴 (실제 액션 값 기준)"""
    # 액션 범위 계산: n_actions=21이면 -10~+10
    action_center = n_actions // 2  # 10 (hold 액션)
    min_action = -action_center     # -10
    max_action = action_center      # +10

    # 액션을 시간 구간별로 그룹화
    n_time_bins = min(50, len(actions) // 10)  # 최대 50개 구간
    if n_time_bins < 5:
        n_time_bins = min(10, len(actions))

    time_bins = np.array_split(range(len(actions)), n_time_bins)

    # 실제 액션 값별 카운트 (인덱스가 아닌 실제 값)
    action_counts = np.zeros((n_time_bins, n_actions))

    for i, time_bin in enumerate(time_bins):
        if len(time_bin) > 0:
            bin_actions = [actions[j] for j in time_bin]
            for raw_action in bin_actions:
                # 실제 액션 값을 인덱스로 변환 (예: -10 → 0, 0 → 10, +10 → 20)
                action_idx = raw_action + action_center
                if 0 <= action_idx < n_actions:
                    action_counts[i, action_idx] += 1

    # 정규화 (각 시간 구간의 합이 1이 되도록)
    row_sums = action_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    action_probs = action_counts / row_sums

    extent = [0, action_probs.shape[0], min_action - 0.5, max_action + 0.5]

    # 히트맵 그리기
    im = ax.imshow(
        action_probs.T,
        aspect='auto',
        cmap='Greys',
        interpolation='nearest',
        origin='lower',
        extent=extent
    )

    # 축 레이블 설정
    ax.set_xlabel('Time Periods')
    ax.set_ylabel('Action Value')
    ax.set_title('Action Distribution Heatmap Over Time (Action Values)')

    # x축: 시간 구간별 대표 타임스탬프
    if len(timesteps) > 0:
        time_labels = []
        for time_bin in time_bins[::max(1, len(time_bins)//6)]:  # 최대 6개 레이블
            if len(time_bin) > 0:
                mid_idx = time_bin[len(time_bin)//2]
                if mid_idx < len(timesteps):
                    time_labels.append(pd.to_datetime(timesteps[mid_idx]).strftime('%m-%d'))
                else:
                    time_labels.append('')
        
        tick_positions = np.linspace(0, len(time_bins), len(time_labels))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(time_labels, rotation=45)

    # y축: 실제 액션 값으로 표시
    important_action_values = [min_action, -5, 0, 5, max_action]
    ax.set_yticks(important_action_values)
    ax.set_yticklabels([f'{val:+d}' if val != 0 else '0(Hold)' for val in important_action_values])

    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Action Probability', rotation=270, labelpad=20)

    # 중요한 액션 영역 표시
    # Hold 영역 (0 근처)
    ax.axhline(0 - 0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0 + 0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(time_bins)*0.02, 0, 'HOLD(0)', 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7),
           fontsize=8)

    # Short 영역 (음수)
    ax.axhspan(min_action - 0.5, -0.5, alpha=0.1, color='blue', label='Short Zone')
    ax.text(len(time_bins)*0.02, min_action + 2.5, 'SHORT(-)', 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7),
           fontsize=8)

    # Long 영역 (양수)
    ax.axhspan(0.5, max_action + 0.5, alpha=0.1, color='red', label='Long Zone')
    ax.text(len(time_bins)*0.02, max_action - 2.5, 'LONG(+)', 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7),
           fontsize=8)
