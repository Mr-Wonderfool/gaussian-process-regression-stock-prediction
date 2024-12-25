import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Plotter:
    color_palatte = sns.color_palette("muted")

    def __init__(
        self,
    ):
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
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(y_true, label="true", color=cls.color_palatte[color + 1], lw=1)
        ax.plot(mean_y_predict, label="predict", color=cls.color_palatte[color], lw=1)
        ax.fill_between(
            x=np.arange(shift + 1, shift + 1 + len(std_y_predict)),
            y1=mean_y_predict[shift + 1 :] - 1.96 * std_y_predict,
            y2=mean_y_predict[shift + 1 :] + 1.96 * std_y_predict,
            color=cls.color_palatte[color],
            alpha=0.2,
            label=r"95% confidence interval",
        )
        ax.legend()
        plt.style.use('default')
        return fig, ax
