import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sns.set_theme(style="ticks")

    dir = os.path.dirname(__file__)
    file_name = "perf.csv"
    file_path = os.path.join(dir, file_name)

    perfs = pd.read_csv(file_path)

    flags = perfs.Gflops.map(
        lambda gflops: np.isfinite(gflops) and not np.isnan(gflops)
    )

    # Define the palette as a list to specify exact values
    palette = sns.color_palette("rocket_r")

    # Plot the lines on two facets
    fig = sns.lineplot(
        data=perfs[flags],
        x="Dimension",
        y="Gflops",
        hue="Routine",
    )
    plt.show()
