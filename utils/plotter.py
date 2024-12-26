# utils/plotter.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Plotter:
    # Corrected the typo from 'color_palatte' to 'color_palette'
    color_palette = sns.color_palette("muted")

    def __init__(self):
        pass

    @classmethod
    def plot_prediction(
        cls,
        y_true,
        mean_y_predict,
        std_y_predict,
        shift: int,
        color: int,
        figsize=(15, 6),
    ):
        """
        Plots the true vs. predicted values with an optional confidence interval.

        Parameters:
        - y_true (array-like): True stock prices.
        - mean_y_predict (array-like): Predicted stock prices.
        - std_y_predict (array-like or None): Standard deviation of predictions. If None, confidence interval is not plotted.
        - shift (int): Number of initial data points to shift (look_back).
        - color (int): Index to select color from the color palette.
        - figsize (tuple): Size of the plot.
        
        Returns:
        - fig (Figure): Matplotlib figure object.
        - ax (Axes): Matplotlib axes object.
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the true values
        ax.plot(
            y_true,
            label="True",
            color=cls.color_palette[color + 1],  # Different color for true values
            lw=1
        )

        # Plot the predicted values
        ax.plot(
            mean_y_predict,
            label="Predicted",
            color=cls.color_palette[color],  # Color based on model
            lw=1
        )

        # Plot the confidence interval if std_y_predict is provided
        if std_y_predict is not None:
            # Ensure that the length of std_y_predict matches the prediction range
            if len(std_y_predict) != len(mean_y_predict) - (shift + 1):
                raise ValueError("Length of std_y_predict does not match the prediction range.")

            ax.fill_between(
                x=np.arange(shift + 1, shift + 1 + len(std_y_predict)),
                y1=mean_y_predict[shift + 1:] - 1.96 * std_y_predict,
                y2=mean_y_predict[shift + 1:] + 1.96 * std_y_predict,
                color=cls.color_palette[color],
                alpha=0.2,
                label="95% Confidence Interval"
            )

        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.set_title('Stock Price Prediction')
        plt.style.use('default')
        plt.tight_layout()
        return fig, ax

