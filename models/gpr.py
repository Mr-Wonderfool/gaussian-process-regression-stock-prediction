from utils.preprocess import Config, data_processing, company_data
from utils.plotter import Plotter
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
        
        # 标准化数据
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_data)
        dev_data = self.scaler.transform(dev_data)
        
        X_train, y_train = data_processing(train_data, self.look_back)
        X_dev, _ = data_processing(dev_data, self.look_back)
        
        # register variables
        self.config = configs.config
        self.X_train, self.y_train, self.X_dev = X_train, y_train, X_dev
        self.orig_train_data, self.orig_dev_data = orig_train_data, orig_dev_data
        
        # 核函数选择逻辑
        kernel_config = self.config[self._field]['kernel'].copy()
        constant = kernel_config.pop("constant")
        kernel_type = kernel_config.pop("type")
        
        if kernel_type == "RBF":
            rbf_params = kernel_config.pop("rbf_params")
            self.kernel = constant * RBF(**rbf_params)
        elif kernel_type == "DotProduct":
            dotproduct_params = kernel_config.pop("dotproduct_params")
            self.kernel = constant * DotProduct(**dotproduct_params)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        # 创建 GPR 参数字典，移除 kernel 相关配置
        gpr_params = self.config[self._field].copy()
        gpr_params.pop('kernel')
        
        # 初始化 GPR
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            optimizer='fmin_l_bfgs_b',
            random_state=None,
            **gpr_params
        )
        
        # 设置优化器参数（如果需要）
        # if hasattr(self.gp, '_optimizer'):
        #     self.gp._optimizer.update({
        #         'maxiter': 100000,    # 增加最大迭代次数
        #         'gtol': 1e-6,       # 调整梯度收敛阈值
        #         'ftol': 1e-6,       # 调整函数值收敛阈值
        #     })

    @classmethod
    def rescale_std(cls, delta_std):
        """ input std for first order difference, scale std back to original range """
        cov = 1. # cov heuristic for x_t, x_{t-1}
        train_std_reversed = np.sqrt((delta_std ** 2) * 0.5 + cov)
        return train_std_reversed
        
    def acquire_prediction(self, mean_delta_predict, orig_data):
        predict = np.empty_like(orig_data)
        predict[:self.look_back + 1] = np.nan
        
        # 反标准化预测结果
        mean_delta_predict_original = self.scaler.inverse_transform(
            mean_delta_predict.reshape(-1, 1)
        ).flatten()
        
        predict[self.look_back + 1:] = orig_data[self.look_back:-1] + mean_delta_predict_original
        return predict
    
    def fit(self, ):
        self.gp = self.gp.fit(self.X_train, self.y_train)
        delta_train_mean_predict, delta_train_std_predict = self.gp.predict(self.X_train, return_std=True)
        delta_dev_mean_predict, delta_dev_std_predict = self.gp.predict(self.X_dev, return_std=True)
        
        # 调整标准差的比例
        std_train_predict = self.rescale_std(delta_train_std_predict * self.scaler.scale_[0])
        std_dev_predict = self.rescale_std(delta_dev_std_predict * self.scaler.scale_[0])
        
        train_predict = self.acquire_prediction(delta_train_mean_predict, self.orig_train_data)
        dev_predict = self.acquire_prediction(delta_dev_mean_predict, self.orig_dev_data)
        
        fig_train, ax_train = Plotter.plot_prediction(
            y_true=self.orig_train_data,
            mean_y_predict=train_predict,
            std_y_predict=std_train_predict,
            shift=self.look_back,
            color=0
        )
        fig_dev, ax_dev = Plotter.plot_prediction(
            y_true=self.orig_dev_data,
            mean_y_predict=dev_predict,
            std_y_predict=std_dev_predict,
            shift=self.look_back,
            color=0
        )
        
        ax_train.set_title(f"{self.company} prediction on train set")
        ax_dev.set_title(f"{self.company} prediction on dev set")
        
        # obtain train and dev error
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
        _field = "Test"
        raw_data = pd.read_csv(test_data_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        test_start = pd.to_datetime(self.config[_field]['start'])
        test_end = pd.to_datetime(self.config[_field]['end'])
        price = company_data(raw_data, self.company, test_start, test_end)
        orig_test_data = price[self.company].values
        test_data = np.diff(orig_test_data, n=1).reshape(-1, 1)
        
        # 标准化测试数据
        test_data = self.scaler.transform(test_data)
        
        X_test, _ = data_processing(test_data, seq_length=self.look_back)
        test_mean_predict, test_std_predict = self.gp.predict(X_test, return_std=True)
        
        # 调整标准差的比例
        std_test_predict = self.rescale_std(test_std_predict * self.scaler.scale_[0])
        
        test_predict = self.acquire_prediction(test_mean_predict, orig_test_data)
        fig_test, ax_test = Plotter.plot_prediction(
            y_true=orig_test_data,
            mean_y_predict=test_predict,
            std_y_predict=std_test_predict,
            shift=self.look_back,
            color=0
        )
        ax_test.set_title(f"{self.company} prediction on test set")
        test_mse = mean_squared_error(
            y_true=orig_test_data[self.look_back + 1:],
            y_pred=test_predict[self.look_back + 1:]
        )
        return (fig_test, ax_test), test_mse
    
    def get_kernels(self, ):
        return self.kernel