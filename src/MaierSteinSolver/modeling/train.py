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
    hidden_layers: str = "20,20",
    learning_rate: str = "1.0",
    transition_region_path: Path = PROCESSED_DATA_DIR / "CHECKERBOARD_ELLIPSE_2026-01-12T17:43.csv",
    boundary_data_path: Path = PROCESSED_DATA_DIR / "BOUNDARY_DATA.csv",
    model_path: Path = MODELS_DIR / "model.pt",
):
    """
    Parameters: 
        - hidden_layers: str. Example: "20,25,30" means the first hidden
            layer has 20 nodes, the second has 25, and the third has 30. 
            Make sure that there are commas. If using typer, use, e.g. 
            `typer train.py run --hidden-layers 20,25,20`
        
        - learning_rate: str. Example: "1.0." The solver sometimes didn't
            converge with 10.0 learning rate.

        - transition_region_path: str of a filepath. This should be a .csv file
            containing the data points in the transition region.

    """

    hidden_layers = hidden_layers.split(",")
    hidden_layers = [int(y) for y in hidden_layers]

    layer_array = [IN_SIZE] + hidden_layers + [OUT_SIZE]
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
                                    alpha= float(learning_rate),\
                                    transition_region_features=ellipse_features,\
                                    bdy_A_features=bdy_A_features,\
                                    bdy_B_features=bdy_B_features)
    
    loss_plot = my_training_object.train()

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    app()
