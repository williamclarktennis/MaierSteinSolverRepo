from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from torch import nn
import torch

import pandas as pd

from MaierSteinSolver.config import MODELS_DIR, PROCESSED_DATA_DIR,\
                                    IN_SIZE, OUT_SIZE, X_COORD, Y_COORD

from MaierSteinSolver.modeling.pinn import NeuralNetwork, PINNTrainingVarTrainData

app = typer.Typer()



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    boundary_data_path: Path = PROCESSED_DATA_DIR / "BOUNDARY_DATA_2026-01-12T12:05.csv",
    transition_region_path: Path = PROCESSED_DATA_DIR / "CHECKERBOARD_ELLIPSE_2026-01-12T17:43.csv",
    model_path: Path = MODELS_DIR / "model.pt",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------

    layer_array = [IN_SIZE, 20, OUT_SIZE]
    model = NeuralNetwork(layer_array=layer_array)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ellipse_features = pd.read_csv(transition_region_path)
    ellipse_features = ellipse_features[[X_COORD,Y_COORD]]
    ellipse_features = ellipse_features.to_numpy()

    bdy_df = pd.read_csv(boundary_data_path)
    bdy_A_df = bdy_df[bdy_df['Region'] == 'A']
    bdy_B_df = bdy_df[bdy_df['Region'] == 'B']
    bdy_A_features = bdy_A_df[[X_COORD,Y_COORD]].to_numpy()
    bdy_B_features = bdy_B_df[[X_COORD,Y_COORD]].to_numpy()
    # boundary labels are not explicitly passed through

    my_training_object = PINNTrainingVarTrainData(model, optimizer,\
                                    loss_fn, epochs = 20,\
                                    alpha= 10.0,\
                                    transition_region_features=ellipse_features,\
                                    bdy_A_features=bdy_A_features,\
                                    bdy_B_features=bdy_B_features)
    
    loss_plot = my_training_object.train()

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    app()
