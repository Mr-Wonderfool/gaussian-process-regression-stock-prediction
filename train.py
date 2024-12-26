# train.py

from models.gpr import GPR
from models.bayes_ridge import BayesRidgeModel
import matplotlib.pyplot as plt

if __name__ == "__main__":

    train_data_path = "data/train_2005to2019.csv"
    test_data_path = "data/test_2019to2020.csv"
    config_path = "configs/data.yaml"

    # Initialize GPR model
    gpr = GPR(data_path=train_data_path, config_path=config_path)
    (fig_train_gpr, ax_train_gpr, fig_dev_gpr, ax_dev_gpr), (train_mse_gpr, dev_mse_gpr) = gpr.fit()

    # Initialize BayesRidge model
    bayes_ridge = BayesRidgeModel(data_path=train_data_path, config_path=config_path)
    (fig_train_bayes, ax_train_bayes, fig_dev_bayes, ax_dev_bayes), (train_mse_bayes, dev_mse_bayes) = bayes_ridge.fit()

    # Predict with GPR
    (fig_test_gpr, ax_test_gpr), test_mse_gpr = gpr.predict(test_data_path)

    # Predict with BayesRidge
    (fig_test_bayes, ax_test_bayes), test_mse_bayes = bayes_ridge.predict(test_data_path)

    # Save plots for GPR
    fig_train_gpr.savefig("train_prediction_gpr.png", dpi=300)
    fig_dev_gpr.savefig("dev_prediction_gpr.png", dpi=300)
    fig_test_gpr.savefig("test_prediction_gpr.png", dpi=300)
    print("GPR Plots saved to files with dpi=300.")

    # Save plots for BayesRidge
    fig_train_bayes.savefig("train_prediction_bayes_ridge.png", dpi=300)
    fig_dev_bayes.savefig("dev_prediction_bayes_ridge.png", dpi=300)
    fig_test_bayes.savefig("test_prediction_bayes_ridge.png", dpi=300)
    print("BayesRidge Plots saved to files with dpi=300.")

    # Display all plots
    plt.show()
    print("Plots displayed.")

    # Print MSE for GPR
    print("GPR Model Performance:")
    print(f"Train MSE: {train_mse_gpr}")
    print(f"Dev MSE: {dev_mse_gpr}")
    print(f"Test MSE: {test_mse_gpr}")

    # Print MSE for BayesRidge
    print("\nBayesRidge Model Performance:")
    print(f"Train MSE: {train_mse_bayes}")
    print(f"Dev MSE: {dev_mse_bayes}")
    print(f"Test MSE: {test_mse_bayes}")

    # Print Kernel Information for GPR
    print("\nGPR Final Kernel Information:")
    print(gpr.get_kernels())

    # Print BayesRidge Model Parameters
    print("\nBayesRidge Model Parameters:")
    print(bayes_ridge.get_model_info())
