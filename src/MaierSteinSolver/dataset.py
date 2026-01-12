from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd

from datetime import datetime

from MaierSteinSolver.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, \
                                    EXTERNAL_DATA_DIR, MODELS_DIR, \
                                    CENTER_A, CENTER_B, RADIUS_A, RADIUS_B, \
                                    Q_BDY_A, Q_BDY_B, \
                                    X_COORD, Y_COORD, Q_LABEL,\
                                    LEFT_LIM, RIGHT_LIM, LOW_LIM, UP_LIM

app = typer.Typer()

def make_checkerboard_ellipse_train_dataset_csv():
    # naive way without polar coordinates

    # first take uniform sample 
    # on each axis, for instance for the x-axis:
    # [LEFT_LIM, RIGHT_LIM] \times [LOW_LIM, UP_LIM]. 
    # Note that the resulting mesh grid will 
    # NOT be a uniform distribution, 
    # but rather a checkerboard distribution. 
    # This is a special case of where the marginals
    # are uniform but not the joint distribution.

    logger.info("Processing dataset...") 
    rng = np.random.default_rng()

    num_x_pts = (RIGHT_LIM - LEFT_LIM) * 100
    num_x_pts = int(num_x_pts)
    num_y_pts = (UP_LIM-LOW_LIM) * 100
    num_y_pts = int(num_y_pts)
    x = rng.uniform(low = LEFT_LIM, high = RIGHT_LIM, size = num_x_pts)
    y = rng.uniform(low = LOW_LIM, high = UP_LIM, size = num_y_pts)

    xx, yy = np.meshgrid(x,y)
    x_coord = xx.reshape(xx.size)
    y_coord = yy.reshape(yy.size)

    rectangle = np.concatenate((x_coord[:,np.newaxis],\
                                y_coord[:,np.newaxis]),axis = 1)

    # remove points outside of the ellipse with 
    # major axis being the x axis and 
    # vertices at (LEFT_LIM,0) and (RIGHT_LIM,0)
    # and co-vertices (0,UP_LIM) and (0,LOW_LIM)
    mask_ellipse = ((x_coord**2)/ (RIGHT_LIM**2) + \
                    (y_coord**2) / (UP_LIM**2)) <= 1
    x_coord = x_coord[mask_ellipse]
    y_coord = y_coord[mask_ellipse]

    ellipse = np.concatenate((x_coord[:,np.newaxis],\
                                y_coord[:,np.newaxis]),axis = 1)
    
    # remove points in metastable region A:
    mask_A = (x_coord - CENTER_A[0])**2 + \
             (y_coord - CENTER_A[1])**2 > RADIUS_A**2
    x_coord = x_coord[mask_A]
    y_coord = y_coord[mask_A]

    # remove points in metastable region B:
    mask_B = (x_coord - CENTER_B[0])**2 + \
             (y_coord - CENTER_B[1])**2 > RADIUS_B**2
    x_coord = x_coord[mask_B]
    y_coord = y_coord[mask_B]

    ellipse_no_A_B = np.concatenate((x_coord[:,np.newaxis],\
                                y_coord[:,np.newaxis]),axis = 1)

    ellipse_no_A_B_df = pd.DataFrame(ellipse_no_A_B, \
                                     columns=[X_COORD, Y_COORD])
    
    now = datetime.now().isoformat(timespec='minutes')
    
    ellipse_no_A_B_df.to_csv(RAW_DATA_DIR/f"CHECKERBOARD_ELLIPSE_{now}.csv", \
                             index = False)
    logger.success("Processing dataset complete.")
    pass

def make_ellipse_pts_csv():
    logger.info("Processing dataset...")
    pts_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_pts.csv", \
                         header = None)
    pts_df = pts_df.rename(columns = { 0 : X_COORD, 1 : Y_COORD })
    pts_df.to_csv(RAW_DATA_DIR/"Ellipse_pts.csv", index = False)
    logger.success("Processing dataset complete.")
    pass

def make_finite_elements_prediction_csv():
    logger.info("Processing dataset...")
    pts_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_pts.csv", \
                         header = None)
    FEM_q_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_q_FEM.csv", \
                           header = None)

    pts_df = pts_df.rename(columns = { 0 : X_COORD, 1 : Y_COORD })
    FEM_q_df = FEM_q_df.rename(columns = { 0 : Q_LABEL })

    FEM_pts_q_df = pd.concat([pts_df, FEM_q_df], axis = 1)

    FEM_pts_q_df.to_csv(MODELS_DIR / "FEM_Commitor.csv", index=False)
    logger.success("Processing dataset complete.")
    pass


def make_bdy_csv(num_pts_A: int = 500, num_pts_B: int = 500):
    # make a csv containing a uniform sample 
    # of the boundary points from $A$ and $B$

    logger.info("Generating uniform samples from boundaries of A and B...")
    rng = np.random.default_rng()

    theta_A = rng.uniform(low = 0.0, high = 2 * np.pi, size = num_pts_A)
    pts_A_xcoord = CENTER_A[0] + RADIUS_A * np.cos(theta_A)
    pts_A_ycoord = CENTER_A[1] + RADIUS_A * np.sin(theta_A)
    pts_A = np.concatenate((pts_A_xcoord[:, np.newaxis], \
                            pts_A_ycoord[:, np.newaxis]), axis=1)
    
    theta_B = rng.uniform(low = 0.0, high = 2 * np.pi, size = num_pts_B)
    pts_B_xcoord = CENTER_B[0] + RADIUS_B * np.cos(theta_B)
    pts_B_ycoord = CENTER_B[1] + RADIUS_B * np.sin(theta_B)
    pts_B = np.concatenate((pts_B_xcoord[:, np.newaxis], \
                            pts_B_ycoord[:, np.newaxis]), axis=1)
    
    all_bdy_pts = np.concatenate((pts_A,pts_B))

    logger.info("Labeling with committor values...")
    # add committor value column and region label column:
    pts_A_df = pd.DataFrame(pts_A, columns = [X_COORD,Y_COORD])
    committor_A = np.array([Q_BDY_A for i in range(pts_A_df.shape[0])])
    committor_val_A_series = pd.Series(committor_A,\
                                        name=Q_LABEL)
    region_A = np.array(["A" for i in range(pts_A_df.shape[0])])
    region_A_series = pd.Series(region_A,\
                                        name="Region")
    pts_A_df = pd.concat([region_A_series, pts_A_df, committor_val_A_series],\
                         axis = 1)

    pts_B_df = pd.DataFrame(pts_B, columns = [X_COORD,Y_COORD])
    committor_B = np.array([Q_BDY_B for i in range(pts_B_df.shape[0])])
    committor_val_B_series = pd.Series(committor_B,\
                                        name=Q_LABEL)
    region_B = np.array(["B" for i in range(pts_B_df.shape[0])])
    region_B_series = pd.Series(region_B,\
                                        name="Region")
    pts_B_df = pd.concat([region_B_series, pts_B_df, committor_val_B_series],\
                         axis = 1)
    

    frames = [pts_A_df,pts_B_df]
    pts_A_B_df = pd.concat(frames)
    
    now = datetime.now().isoformat(timespec='minutes')
    pts_A_B_df.to_csv(RAW_DATA_DIR / f"BOUNDARY_DATA_{now}.csv", index=False)
    logger.success("Processing dataset complete.")



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
