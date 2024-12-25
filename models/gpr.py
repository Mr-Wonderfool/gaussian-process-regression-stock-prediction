from utils.preprocess import Config, data_processing, company_data
from utils.plotter import Plotter
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

class GPR:
    _field = "GPR"

    def __init__(
        self, data_path, config_path
    ):
        configs = Config(config_path)
        raw_data = pd.read_csv(data_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        # Config section
        _field = "Config"
        self.company = configs.query(_field, "company")
        self.start = pd.to_datetime(configs.query(_field, "start"))
        self.end = pd.to_datetime(configs.query(_field, "end"))
        self.look_back = configs.query(_field, "look_back")
        # train test split and normalization
        _field = "Train"
        company_stock_price = company_data(raw_data, self.company, self.start, self.end)
        price = company_stock_price[self.company].values
        dev_size = configs.query(_field, "dev_size")
        split = int(len(price) * (1 - dev_size))
        orig_train_data, orig_dev_data = price[:split], price[split:]
        # change to first order difference
        train_data = np.diff(orig_train_data, n=1).reshape(-1, 1)
        dev_data = np.diff(orig_dev_data, n=1).reshape(-1, 1)
        X_train, y_train = data_processing(train_data, self.look_back)
        X_dev, _ = data_processing(dev_data, self.look_back)
        # register variables
        self.config = configs.config
        self.X_train, self.y_train, self.X_dev = X_train, y_train, X_dev
        self.orig_train_data, self.orig_dev_data = orig_train_data, orig_dev_data
        curr_field = 'kernel'
        constant = self.config[self._field][curr_field].pop("constant")
        self.kernel = constant * RBF(**self.config[self._field].pop(curr_field))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, **self.config[self._field])
    
    @classmethod
    def rescale_std(cls, delta_std):
        """ input std for first order difference, scale std back to original range """
        cov = 1. # cov heuristic for x_t, x_{t-1}
        train_std_reversed = np.sqrt((delta_std ** 2) * 0.5 + cov)
        return train_std_reversed
    def acquire_prediction(self, mean_delta_predict, orig_data):
        predict = np.empty_like(orig_data)
        predict[:self.look_back + 1] = np.NaN
        predict[self.look_back + 1:] = orig_data[self.look_back:-1] + mean_delta_predict
        return predict
    
    def fit(self, ):
        self.gp = self.gp.fit(self.X_train, self.y_train)
        delta_train_mean_predict, delta_train_std_predict = self.gp.predict(self.X_train, return_std=True)
        delta_dev_mean_predict, delta_dev_std_predict = self.gp.predict(self.X_dev, return_std=True)
        std_train_predict = self.rescale_std(delta_train_std_predict)
        std_dev_predict = self.rescale_std(delta_dev_std_predict)
        train_predict = self.acquire_prediction(mean_delta_predict=delta_train_mean_predict, orig_data=self.orig_train_data)
        dev_predict = self.acquire_prediction(mean_delta_predict=delta_dev_mean_predict, orig_data=self.orig_dev_data)
        fig_train, ax_train = Plotter.plot_prediction(y_true=self.orig_train_data, mean_y_predict=train_predict, std_y_predict=std_train_predict, shift=self.look_back, color=0)
        fig_dev, ax_dev = Plotter.plot_prediction(y_true=self.orig_dev_data, mean_y_predict=dev_predict, std_y_predict=std_dev_predict, shift=self.look_back, color=0)
        ax_train.set_title(f"{self.company} prediction on train set")
        ax_dev.set_title(f"{self.company} prediction on dev set")
        # obtain train and dev error
        train_mse = mean_squared_error(y_true=self.orig_train_data[self.look_back + 1:], y_pred=train_predict[self.look_back + 1:])
        dev_mse = mean_squared_error(y_true=self.orig_dev_data[self.look_back + 1:], y_pred=dev_predict[self.look_back + 1:])
        return (fig_train, ax_train, fig_dev, ax_dev), (train_mse, dev_mse)
    
    def predict(self, test_data_path):
        _field = "Test"
        raw_data = pd.read_csv(test_data_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        test_start = pd.to_datetime(self.config[_field]['start'])
        test_end = pd.to_datetime(self.config[_field]['end'])
        price = company_data(raw_data, self.company, test_start, test_end)
        orig_test_data = price[self.company].values
        test_data = np.diff(orig_test_data, n=1).reshape(-1, 1)
        X_test, _ = data_processing(test_data, seq_length=self.look_back)
        test_mean_predict, test_std_predict = self.gp.predict(X_test, return_std=True)
        std_test_predict = self.rescale_std(test_std_predict)
        test_predict = self.acquire_prediction(mean_delta_predict=test_mean_predict, orig_data=orig_test_data)
        fig_test, ax_test = Plotter.plot_prediction(y_true=orig_test_data, mean_y_predict=test_predict, std_y_predict=std_test_predict, shift=self.look_back, color=0)
        ax_test.set_title(f"{self.company} prediction on test set")
        test_mse = mean_squared_error(y_true=orig_test_data[self.look_back + 1:], y_pred=test_predict[self.look_back + 1:])
        return (fig_test, ax_test), test_mse
    
    def get_kernels(self, ):
        return self.kernel