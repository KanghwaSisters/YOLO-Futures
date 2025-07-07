import numpy as np

class MinMaxScaler:
    def fit(self, x):
        # x: shape = (Batch, T, D)
        self.min = np.min(x, axis=1, keepdims=True)
        self.max = np.max(x, axis=1, keepdims=True)

        self.transform_ftn = lambda x: (x - self.min) / (self.max - self.min + 1e-8) 
        self.recover_ftn = lambda x: self.min + x*(self.max - self.min)

    def transform(self, x):
        return self.transform_ftn(x)

    def recover(self, x):
        return self.recover_ftn(x)

    def fit_transform(self, x):
        self.fit(x)
        result = self.transform(x)
        print("Scaling Completed.")
        return result

class RobustScaler:
    def fit(self, x):
        # x: shape = (Batch, T, D)
        self.median = np.median(x, axis=1, keepdims=True)
        q1 = np.percentile(x, 25, axis=1, keepdims=True)
        q3 = np.percentile(x, 75, axis=1, keepdims=True)
        self.iqr = q3 - q1 + 1e-8  # numerical stability

        self.transform_ftn = lambda x: (x - self.median) / self.iqr
        self.recover_ftn = lambda x: self.median + x * self.iqr

    def transform(self, x):
        return self.transform_ftn(x)

    def recover(self, x):
        return self.recover_ftn(x)

    def fit_transform(self, x):
        self.fit(x)
        result = self.transform(x)
        print("Robust Scaling Completed.")
        return result

class StandardScaler:
    def fit(self, x):
        # x: shape = (Batch, T, D)
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.std = np.std(x, axis=1, keepdims=True) 

        self.transform_ftn = lambda x: (x - self.mean) / (self.std + 1e-8) 
        self.recover_ftn = lambda x: self.mean + x * self.std

    def transform(self, x):
        return self.transform_ftn(x)

    def recover(self, x):
        return self.recover_ftn(x)

    def fit_transform(self, x):
        self.fit(x)
        result = self.transform(x)
        print("Standard Scaling Completed.")
        return result