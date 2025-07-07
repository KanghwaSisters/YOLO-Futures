import numpy as np

class State:
    def __init__(self, target_values, indices_list=None):
        # Check shape
        self.target_values = target_values
        self._indices_list = indices_list

        if indices_list is None:
            self.flag = False
        else:
            self.get_dataset_indices(indices_list)

    def get_dataset_indices(self, indices_list):

        self.flag = True
        self._indices_list = indices_list

         # Safe indexing
        try:
            self.target_indices = [self._indices_list.index(val) for val in self.target_values]
            
        except ValueError as e:
            raise ValueError(f"target_values must exist in indices_list. {e}")


    def _get_target_data(self, data, target_indices):
        return data[:, target_indices]

    def __call__(self, timeseries_data, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Returns (timeseries_features, agent_features)"""
        if self.flag == False:
            raise ValueError("Must get target indices before calling ")
        
        if timeseries_data.ndim != 2:
            raise ValueError("Expected timeseries_data to be 2D array (T, D)")

        self.timeseries_data = timeseries_data
        self.target_timeseries_data = self._get_target_data(timeseries_data, self.target_indices)

        # Ordered agent data
        self.agent_keys = sorted(kwargs.keys())
        self.agent_data = np.array([kwargs[k] for k in self.agent_keys], dtype=np.float32)

        return self.target_timeseries_data, self.agent_data

    @property
    def name(self):
        return self.__class__.__name__