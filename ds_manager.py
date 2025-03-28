import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class DSManager:
    def __init__(self, name="lucas",folds=1,scale_y="robust",shuffle=True,train_size=0.75,split="normal"):
        self.name = name
        dataset_path = f"data/{self.name}.csv"
        df = pd.read_csv(dataset_path)
        self.folds = folds
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.data = df.to_numpy()

        self.scaler_X = MinMaxScaler()
        self.scaler_y = RobustScaler()
        if scale_y == "minmax":
            self.scaler_y = MinMaxScaler()

        self.data[:,0:-1] = self.scaler_X.fit_transform(self.data[:,0:-1])
        self.data[:,-1] = self.scaler_y.fit_transform(self.data[:,-1].reshape(-1,1)).ravel()
        self.train_size = train_size
        self.split = split

    def get_k_folds(self):
        for i in range(self.folds):
            train_data, test_data = train_test_split(self.data, train_size=self.train_size, random_state=42+i)
            yield train_data[:,0:-1], train_data[:,-1], test_data[:,0:-1], test_data[:,-1]


