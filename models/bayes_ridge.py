# models/bayes_ridge.py

from utils.preprocess import Config, data_processing, company_data
from utils.plotter import Plotter
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class BayesRidgeModel:
    _field = "BayesRidge"

    def __init__(self, data_path, config_path):
        configs = Config(config_path)
        raw_data = pd.read_csv(data_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])

        # Config section
        config_field = "Config"
        self.company = configs.query(config_field, "company")
        self.start = pd.to_datetime(configs.query(config_field, "start"))
        self.end = pd.to_datetime(configs.query(config_field, "end"))
        self.look_back = configs.query(config_field, "look_back")

        # Train-test split and normalization
        train_field = "Train"
        company_stock_price = company_data(raw_data, self.company, self.start, self.end)
        price = company_stock_price[self.company].values
        dev_size = configs.query(train_field, "dev_size")
        split = int(len(price) * (1 - dev_size))
        orig_train_data, orig_dev_data = price[:split], price[split:]

        # Change to first order difference
        train_data = np.diff(orig_train_data, n=1).reshape(-1, 1)
        dev_data = np.diff(orig_dev_data, n=1).reshape(-1, 1)

        # Standardize data
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_data)
        dev_data = self.scaler.transform(dev_data)

        X_train, y_train = data_processing(train_data, self.look_back)
        X_dev, _ = data_processing(dev_data, self.look_back)

        # Register variables
        self.config = configs.config
        self.X_train, self.y_train, self.X_dev = X_train, y_train, X_dev
        self.orig_train_data, self.orig_dev_data = orig_train_data, orig_dev_data

        # Initialize BayesianRidge with config parameters
        bayes_params = self.config[self._field].copy()
        self.model = BayesianRidge(
            alpha_1=bayes_params.pop('alpha_1', 1e-6),
            alpha_2=bayes_params.pop('alpha_2', 1e-6),
            lambda_1=bayes_params.pop('lambda_1', 1e-6),
            lambda_2=bayes_params.pop('lambda_2', 1e-6),
            fit_intercept=bayes_params.pop('fit_intercept', True),
            copy_X=bayes_params.pop('copy_X', True),
            tol=bayes_params.pop('tol', 1e-3),
            max_iter=bayes_params.pop('max_iter', 300)
        )

    @classmethod
    def rescale_std(cls, delta_std):
        """Input std for first order difference, scale std back to original range"""
        cov = 1.  # Covariance heuristic for x_t, x_{t-1}
        train_std_reversed = np.sqrt((delta_std ** 2) * 0.5 + cov)
        return train_std_reversed

    def acquire_prediction(self, mean_delta_predict, orig_data):
        predict = np.empty_like(orig_data)
        predict[:self.look_back + 1] = np.nan

        # Inverse transform the predictions
        mean_delta_predict_original = self.scaler.inverse_transform(
            mean_delta_predict.reshape(-1, 1)
        ).flatten()

        predict[self.look_back + 1:] = orig_data[self.look_back:-1] + mean_delta_predict_original
        return predict

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        delta_train_mean_predict = self.model.predict(self.X_train)
        delta_dev_mean_predict = self.model.predict(self.X_dev)

        # BayesianRidge does not provide standard deviation directly
        # Hence, std is set to None or can be approximated if needed

        train_predict = self.acquire_prediction(delta_train_mean_predict, self.orig_train_data)
        dev_predict = self.acquire_prediction(delta_dev_mean_predict, self.orig_dev_data)

        fig_train, ax_train = Plotter.plot_prediction(
            y_true=self.orig_train_data,
            mean_y_predict=train_predict,
            std_y_predict=None,  # No std available
            shift=self.look_back,
            color=1  # Different color for BayesRidge
        )
        fig_dev, ax_dev = Plotter.plot_prediction(
            y_true=self.orig_dev_data,
            mean_y_predict=dev_predict,
            std_y_predict=None,  # No std available
            shift=self.look_back,
            color=1  # Different color for BayesRidge
        )

        ax_train.set_title(f"{self.company} BayesRidge Prediction on Train Set")
        ax_dev.set_title(f"{self.company} BayesRidge Prediction on Dev Set")

        # Obtain train and dev error
        train_mse = mean_squared_error(
            y_true=self.orig_train_data[self.look_back + 1:],
            y_pred=train_predict[self.look_back + 1:]
        )
        dev_mse = mean_squared_error(
            y_true=self.orig_dev_data[self.look_back + 1:],
            y_pred=dev_predict[self.look_back + 1:]
        )
        return (fig_train, ax_train, fig_dev, ax_dev), (train_mse, dev_mse)

    def predict(self, test_data_path):
        test_field = "Test"
        raw_data = pd.read_csv(test_data_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        test_start = pd.to_datetime(self.config[test_field]['start'])
        test_end = pd.to_datetime(self.config[test_field]['end'])
        price = company_data(raw_data, self.company, test_start, test_end)
        orig_test_data = price[self.company].values
        test_data = np.diff(orig_test_data, n=1).reshape(-1, 1)

        # Standardize test data
        test_data = self.scaler.transform(test_data)

        X_test, _ = data_processing(test_data, seq_length=self.look_back)
        test_mean_predict = self.model.predict(X_test)

        # BayesianRidge does not provide standard deviation directly
        # Hence, std_test_predict is set to None
        test_predict = self.acquire_prediction(test_mean_predict, orig_test_data)

        fig_test, ax_test = Plotter.plot_prediction(
            y_true=orig_test_data,
            mean_y_predict=test_predict,
            std_y_predict=None,  # No std available
            shift=self.look_back,
            color=1  # Different color for BayesRidge
        )
        ax_test.set_title(f"{self.company} BayesRidge Prediction on Test Set")
        test_mse = mean_squared_error(
            y_true=orig_test_data[self.look_back + 1:],
            y_pred=test_predict[self.look_back + 1:]
        )
        return (fig_test, ax_test), test_mse

    def get_model_info(self):
        return self.model.get_params()

