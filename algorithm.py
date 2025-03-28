from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error
import torch
import importlib
import numpy as np
import time


class Algorithm(ABC):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        self.dataset = dataset
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        
        self.target_size = target_size
        self.fold = fold
        self.scaler_y = scaler_y
        self.mode = mode
        self.train_size = train_size
        self.reporter = reporter
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_fold(self):
        start = time.time()
        self._fit()
        end = time.time()
        execution_time = end - start
        train_y_hat = self.predict_train()
        test_y_hat = self.predict_test()
        train_r2, train_rmse, train_rpd, train_rpiq, train_r2_o, train_rmse_o, train_rpd_o, train_rpiq_o\
            = self.calculate_metrics(self.train_y, train_y_hat)
        r2, rmse, rpd, rpiq, r2_o, rmse_o, rpd_o, rpiq_o \
            = self.calculate_metrics(self.test_y, test_y_hat)
        self.reporter.write_details(self.get_name(),self.dataset.name,self.target_size,
                                    self.scaler_y, self.mode, self.train_size,
                                    r2, rmse, rpd, rpiq,
                                    r2_o,rmse_o, rpd_o, rpiq_o,
                                    train_r2, train_rmse,train_rpd, train_rpiq,
                                    train_r2_o, train_rmse_o, train_rpd_o, train_rpiq_o,
                                    execution_time,
                                    self.get_num_params(),
                                    self.fold, self.get_indices())

    def calculate_4_metrics(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        std_dev = np.std(y_test, ddof=1)
        rpd = std_dev/rmse
        iqr = np.percentile(y_test, 75) - np.percentile(y_test, 25)
        rpiq = iqr/rmse

        return r2, rmse, rpd, rpiq

    def calculate_metrics(self, y_test, y_pred):
        y_test = Algorithm.convert_to_numpy(y_test.detach().cpu().numpy())
        y_pred = Algorithm.convert_to_numpy(y_pred.detach().cpu().numpy())

        r2, rmse, rpd, rpiq = self.calculate_4_metrics(y_test, y_pred)
        y_test_original = self.dataset.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_original = self.dataset.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        r2_o, rmse_o, rpd_o, rpiq_o = self.calculate_4_metrics(y_test_original, y_pred_original)

        return r2, rmse, rpd, rpiq, r2_o, rmse_o, rpd_o, rpiq_o

    @staticmethod
    def convert_to_numpy(t):
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
        return t

    @abstractmethod
    def _fit(self):
        pass

    def get_name(self):
        class_name = self.__class__.__name__
        name_part = class_name[len("Algorithm_"):]
        return name_part

    @staticmethod
    def create(name, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        class_name = f"Algorithm_{name}"
        module = importlib.import_module(f"algorithms.algorithm_{name}")
        clazz = getattr(module, class_name)
        try:
            obj = clazz(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose)
            return obj
        except Exception as e:
            print(e)
            return None

    @abstractmethod
    def get_indices(self):
        pass

    @abstractmethod
    def predict_train(self):
        pass

    @abstractmethod
    def predict_test(self):
        pass

    @abstractmethod
    def get_num_params(self):
        pass