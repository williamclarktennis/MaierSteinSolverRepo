from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import pandas as pd

from MaierSteinSolver.config import MODELS_DIR, PROCESSED_DATA_DIR, IN_SIZE, OUT_SIZE, X_COORD, Y_COORD, Q_LABEL
from MaierSteinSolver.modeling.pinn import NeuralNetwork

app = typer.Typer()



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "Ellipse_pts.csv",
    model_path: Path = MODELS_DIR / "model.pt",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------

    model = NeuralNetwork(layer_array=[IN_SIZE,20,OUT_SIZE])
    model.load_state_dict(torch.load(model_path))

    df = pd.read_csv(features_path)
    df = df[[X_COORD,Y_COORD]]
    pts = df.to_numpy()
    # work around the error of inputting numpy data into torch nn
    pts_torch = torch.tensor(pts).float()

    q_pinn = model(pts_torch)
    q_pinn_np = q_pinn.detach().numpy().squeeze()

    q_pinn_df = pd.DataFrame(q_pinn_np, columns = [Q_LABEL])
    final_thing = pd.concat([df, q_pinn_df], axis = 1)
    final_thing.to_csv(predictions_path)


if __name__ == "__main__":
    app()
