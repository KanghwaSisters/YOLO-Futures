import random
from datahandler.dataset import FuturesDataset
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    def __init__(self, base_dataset: FuturesDataset, window_len: int):
        """
        base_dataset: FuturesDataset에서 만들어진 단일 샘플 단위의 데이터셋
        window_len: 묶을 데이터 개수 (시퀀스 길이)
        """
        self.base_dataset = base_dataset
        self.window_len = window_len
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self):
        # 마지막 index에서 window_len - 1 만큼 이전까지만 가능
        return [i for i in range(len(self.base_dataset) - self.window_len + 1)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.window_len
        return [self.base_dataset[i] for i in range(start, end)]

class EpisodeDataloader:
    # 전체 에피소드를 섞고 나눠주는 코드 
    def __init__(self, dataset: EpisodeDataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

        if self.shuffle:
            self.shuffle_indices()

    def shuffle_indices(self):
        random.shuffle(self.indices)
        self.ptr = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration

        idx = self.indices[self.ptr]
        self.ptr += 1
        return self.dataset[idx]  # 반환: MiniFuturesDataset