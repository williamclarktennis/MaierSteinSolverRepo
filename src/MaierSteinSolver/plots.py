from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MaierSteinSolver.config import FIGURES_DIR, PROCESSED_DATA_DIR,\
                                    X_COORD, Y_COORD, Q_LABEL, \
                                    EXTERNAL_DATA_DIR, \
                                    CENTER_A, CENTER_B, \
                                    BETA, RADIUS_A, RADIUS_B




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

def plot_q(
        filepath: Path = PROCESSED_DATA_DIR / "test_predictions.csv"
    ):
    logger.info("Plotting the committor function.")
    df = pd.read_csv(filepath)
    df = df[[X_COORD,Y_COORD,Q_LABEL]]
    x = df[X_COORD].to_numpy()
    y = df[Y_COORD].to_numpy()
    q = df[Q_LABEL].to_numpy()
    fig, ax = plt.subplots()
    s = ax.scatter(x, y, c= q, s= 1)
    fig.colorbar(s)
    return fig, ax

def compare_fem_absolute(
        model_prediction_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
        fem_path: Path = PROCESSED_DATA_DIR / "FEM_Commitor.csv"
    ):
    # Compare FEM ground truth solution with model solution using absolute 
    # difference

    # process data from the model prediction:
    model_df = pd.read_csv(model_prediction_path)
    model_df = model_df[[X_COORD,Y_COORD,Q_LABEL]]

    x = model_df[X_COORD].to_numpy() # same as for FEM committor features
    y = model_df[Y_COORD].to_numpy()

    model_q = model_df[Q_LABEL].to_numpy()
    
    FEM_df = pd.read_csv(fem_path)
    FEM_df = FEM_df[[X_COORD,Y_COORD,Q_LABEL]]

    fem_q = FEM_df[Q_LABEL].to_numpy()

    # compute absolute difference:
    q_ab = np.absolute(fem_q - model_q)

    # plot absolute difference:
    fig, ax = plt.subplots()
    s = ax.scatter(x, y, c= q_ab, s= 1)
    fig.colorbar(s)
    return fig, ax

def get_color(n_iterations):
    # the paths data points should grow lighter and lighter:

    color = []
    delta = 3/n_iterations
    channel = 0
    for i in tqdm(range(n_iterations)):
        if channel < 1:
            rgb = (0,0,channel)
        elif channel < 2:
            rgb = (0,channel-1.0,1.0)
        elif channel <= 3:
            rgb = (0.5*(channel - 2.0), 1.0, 1.0)
        color.append(rgb)
        channel += delta

    assert len(color) == n_iterations
    return color

def plot_equilibrium_points(ax):
    ax.scatter(CENTER_A[0],CENTER_A[1], s=20, c = ['red'])
    ax.scatter(CENTER_B[0],CENTER_B[1], s=20, c = ['red'])

    # annotate the equilibrium points:
    ax.annotate(r"$O_-$", (-1,0), xytext = (-0.8,0.2), arrowprops = {"width":1, "headwidth": 5, "headlength":5}, bbox = dict(boxstyle="round", fc="0.8", pad=0.3))
    ax.annotate(r"$O_+$", (1,0), xytext=(1.2,0.2), arrowprops = {"width":1, "headwidth": 5, "headlength":5}, bbox = dict(boxstyle="round", fc="0.8", pad=0.3))

def make_direction_field(ax):
    # draws arrows on the plot according to the Maier Stein drift

    x,y = np.meshgrid(np.linspace(-2,2,20),np.linspace(-2,2,20))

    u = x - x**3 - BETA * x * y ** 2
    v = -(1+x**2) * y

    u_shape = u.shape
    v_shape = v.shape
    u_1 = u.reshape((*u_shape,1))
    v_1 = v.reshape((*v_shape,1))
    vector = np.concatenate((u_1,v_1), axis=2)

    norm = np.linalg.norm(vector, axis=2)

    u = u/ (norm)
    v = v/ (norm)

    ax.quiver(x,y,u,v)
    return ax

def put_pts_on_circle(xc,yc,r,n):
    t = np.linspace(0,np.pi*2,n+1)
    pts = np.zeros((n,2))
    pts[:,0] = xc+r*np.cos(t[0:n])
    pts[:,1] = yc+r*np.sin(t[0:n])
    return pts

def put_circles_around_A_and_B(ax):
     # then we put a circle around the regions A and B:
    NA = 1000
    NB = 1000
    xa= CENTER_A[0]
    ya= CENTER_A[1]
    xb= CENTER_B[0]
    yb= CENTER_B[1]
    ra = RADIUS_A
    rb = 0.3
    ptsA = put_pts_on_circle(xa,ya,ra,NA) 
    ptsB = put_pts_on_circle(xb,yb,rb,NB)  
    ax.plot(ptsA[:,0],ptsA[:,1],linewidth = 2,c = 'y')
    ax.plot(ptsB[:,0],ptsB[:,1],linewidth = 2,c = 'r')

def plot_trajectory(sample_path, delta_t = 1e-4):
    # sample_path should be numpy array of shape (a,b,c)
    # delta_t is the time step used for the Euler Maruyama simulation

    # number of points in the sample path:
    n_iterations = sample_path.shape[0]

    # the paths data points should grow lighter and lighter:
    color = get_color(n_iterations=n_iterations)

    fig, ax = plt.subplots()

    # plot the Maier Stein drift field: 
    make_direction_field(ax)

    # plot the trajectory with lighter and lighter color:
    ax.scatter(sample_path[:,0], sample_path[:,1], s=1, c=color)

    # plot the equilibrium points of the Maier Stein system:
    plot_equilibrium_points(ax)

    # indicate the equilibrium regions: 
    put_circles_around_A_and_B(ax)

    # get initial point of the sample path:
    init_pt = sample_path[0]

    # compute the time interval:
    T = int(n_iterations * delta_t)

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title(f"Maier Stein SDE Solution on time interval [0,{T}] with initial point ({init_pt[0,0]}, {init_pt[1,0]})")
    ax.text(-1,-1, "Solution grows lighter as time increases", bbox = dict(boxstyle="round", fc="0.8", pad=0.3))

    plt.show()

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
