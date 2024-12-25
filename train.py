from models.gpr import GPR

if __name__ == "__main__":
    train_data_path = "data/train_2005to2019.csv"
    test_data_path = "data/test_2019to2020.csv"
    config_path = "configs/data.yaml"
    gpr = GPR(data_path=train_data_path, config_path=config_path)
    (fig_train, ax_train, fig_dev, ax_dev), (train_mse, dev_mse) = gpr.fit()
    (fig_test, ax_test), test_mse = gpr.predict(test_data_path)