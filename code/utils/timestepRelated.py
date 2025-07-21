import pandas as pd

def split_date_ranges_by_group(index, n_group=5, train_ratio=0.9):
    """
    날짜 단위로 DatetimeIndex를 그룹으로 나눈 뒤, 각 그룹 내에서
    train/valid의 (start_date, end_date)를 문자열로 반환합니다.

    Parameters:
    - index (pd.Index): DatetimeIndex
    - n_group (int): 그룹 개수
    - train_ratio (float): 각 그룹 내에서 train 비율

    Returns:
    - list[tuple]: [(train_start, train_end), (valid_start, valid_end)] per group
    """
    date_index = pd.to_datetime(index).normalize().unique()
    date_index = pd.Series(sorted(date_index))

    total_len = len(date_index)
    group_size = total_len // n_group

    result = []

    for i in range(n_group):
        start = i * group_size
        end = (i + 1) * group_size if i < n_group - 1 else total_len

        group = date_index[start:end]
        split = int(len(group) * train_ratio)

        train_range = (str(group.iloc[0].date()), str(group.iloc[split - 1].date())) if split > 0 else None
        valid_range = (str(group.iloc[split].date()), str(group.iloc[-1].date())) if split < len(group) else None

        result.append((train_range, valid_range))

    return result