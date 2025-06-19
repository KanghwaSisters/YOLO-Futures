import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import compress

class TimeSeriesSegmenter:
    def __init__(self, window_size: int, target_step: int, with_datetime: bool = True):
        '''
        __init__(window_size: int, target_step: int, with_datetime: bool) 
            -> None : 생성자

        -------
        시계열 분할 클래스의 초기화 함수
        - window_size: 입력 시계열의 길이 (과거 관찰 구간의 크기)
        - target_step: 예측 대상 시점 (미래 몇 분 뒤를 예측할 것인지)
        - with_datetime: datetime 정보 포함 여부

        예:
        segmenter = TimeSeriesSegmenter(80, 10, 'close')
        segmented_df = segmenter(df_dict)
        '''
        self.window_size = window_size
        self.target_step = target_step
        self.with_datetime = with_datetime

    def __call__(self, dataset: dict, data_type: str) -> pd.DataFrame:
        '''
        __call__(dataset: dict, data_type: str) -> return segmented_df : pd.DataFrame

        -------
        여러 개의 시계열 DataFrame이 담긴 dict에서 데이터를 분할한다.
        - 서킷브레이커, 수능, 일반적이지 않은 장을 쉽게 넣고 빼기 위해 dict로 나누어 사용한다. 
        - 각 DataFrame에 대해 window / target_step만큼 슬라이딩 윈도우 방식으로 입력-타겟 시퀀스를 생성한다.
        - 생성된 시퀀스를 하나의 DataFrame으로 반환한다.

        정보:
        - data_type: 사용할 컬럼명 (예: 'close', 'open')

        Return:
        df.DataFrame
        - 입력 시퀀스 (X)
        - 타겟 값 (y)
        - (optional) 해당 시점의 타임스탬프
        '''

        self.data_type = data_type

        xs, ys, target_times = [], [], []

        for df in dataset.values():
            X, y, t_time = self._segment_dataset(df)
            xs.extend(X)
            ys.extend(y)
            target_times.extend(t_time)
        
        df = pd.DataFrame(np.array(xs))
        df['target'] = ys
        
        if self.with_datetime:
            df['target_time'] = target_times  

        return df

    def _segment_dataset(self, df: pd.DataFrame) -> tuple[list, list, list]:
        '''
        _segment_dataset(df: pd.DataFrame) -> return (X: list, y: list, target_times: list)

        -------
        단일 DataFrame에 대해 시계열 분할을 수행한다.
        - 날짜(date)별로 데이터를 그룹화하여 각 그룹마다 슬라이딩 윈도우 분할을 수행한다. 
        - target_step 분 만큼 미래 시점을 타겟으로 사용한다. 
        - 모든 타임스텝이 연속적이지 않은 경우를 처리하기 위해 time을 지표로 이용한다. 
          (예시: 장마감 때는 10분 동안 가격 고지가 이뤄지지 X  15:05 -> 15:15 로 타임 스탬프가 뛴다.)

        예:
        X, y, t = self._segment_dataset(df)
        '''
        X, y, target_times = [], [], []

        for _, group in df.groupby('date'):
            timeseries = group[self.data_type].values         # 예: 'close' 컬럼
            timestep_idx = group.index                        # 인덱스는 datetime index

            # 예측 대상 시점 리스트
            target_indices = timestep_idx[self.window_size:] + timedelta(minutes=self.target_step)
            target_mask = timestep_idx.isin(target_indices)

            # 윈도우 수만큼 반복
            total_iteration = len(timeseries) - self.target_step - self.window_size

            # X: 입력 시계열을 윈도우 단위로 쪼갠다. 
            X_window = [timeseries[i:i+self.window_size] for i in range(total_iteration)]

            # 마스킹된 X만 남기기 (유효한 타겟이 있는 위치만)
            X.extend(compress(X_window, target_mask[self.target_step + self.window_size:]))
            y.extend(timeseries[target_mask])                     # 타겟 y
            target_times.extend(timestep_idx[target_mask])        # 해당 시점의 시간 정보

        return X, y, target_times
    
