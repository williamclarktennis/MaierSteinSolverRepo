from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MaierSteinSolver.config import FIGURES_DIR, PROCESSED_DATA_DIR,\
                                    X_COORD, Y_COORD, Q_LABEL



app = typer.Typer()

def scatter_plot_csv(filepath: str):
    df = pd.read_csv(filepath)
    df = df[[X_COORD,Y_COORD]]
    pts = df.to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(pts[:,0], pts[:,1], s=0.1)
    plt.show()

def plot_loss_curve(
    input_data,
    output_path: Path = FIGURES_DIR / "plot.png",
):
    pass

def plot_q(filepath: Path = PROCESSED_DATA_DIR / "test_predictions.csv"):
    df = pd.read_csv(filepath)
    df = df[[X_COORD,Y_COORD,Q_LABEL]]
    x = df[X_COORD].to_numpy()
    y = df[Y_COORD].to_numpy()
    q = df[Q_LABEL].to_numpy()
    fig, ax = plt.subplots()
    s = ax.scatter(x, y, c= q, s= 1)
    fig.colorbar(s)
    return fig, ax


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
