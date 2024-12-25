from utils.preprocess import Config, data_processing, company_data
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler


def get_config_and_data():
    data_path = "data/train_2005to2019.csv"
    config_path = "configs/data.yaml"
    configs = Config(config_path)
    raw_data = pd.read_csv(data_path)
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    _field = "Config"
    company = configs.query(_field, "company")
    start = pd.to_datetime(configs.query(_field, "start"))
    end = pd.to_datetime(configs.query(_field, "end"))
    look_back = configs.query(_field, "look_back")
    company_stock_price = company_data(raw_data, company, start, end)
    # train test split and normalization
    _field = "Train"
    scaler = StandardScaler()
    price = company_stock_price[company].values
    normalized_price = scaler.fit_transform(price.reshape(-1, 1))
    dev_size = configs.query(_field, "dev_size")
    split = int(len(normalized_price) * (1 - dev_size))
    train_data, dev_data = normalized_price[:split], normalized_price[split:]
    X_train, y_train = data_processing(train_data, look_back)
    X_dev, y_dev = data_processing(dev_data, look_back)
    return configs.config, (X_train, y_train, X_dev, y_dev)

class GPR:
    _field = "GPR"

    def __init__(
        self, 
    ):
        self.config, self.raw_data = get_config_and_data()
        curr_field = 'kernel'
        constant = self.config[self._field][curr_field].pop("constant")
        self.kernel = constant * RBF(**self.config[self._field].pop(curr_field))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, **self.config[self._field])
    
    def fit(self, ):
        X_train, y_train, X_dev, y_dev = self.raw_data
        self.gp.fit(X_train, y_train)
