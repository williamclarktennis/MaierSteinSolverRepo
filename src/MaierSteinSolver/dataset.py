from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
import torch

import math

from datetime import datetime

from MaierSteinSolver.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, \
                                    EXTERNAL_DATA_DIR, MODELS_DIR, \
                                    CENTER_A, CENTER_B, RADIUS_A, RADIUS_B, \
                                    Q_BDY_A, Q_BDY_B, \
                                    X_COORD, Y_COORD, Q_LABEL,\
                                    LEFT_LIM, RIGHT_LIM, LOW_LIM, UP_LIM,\
                                    EPSILON, BETA

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

def removeA_B(data):
    # the goal is to remove the points that fall in
    # one of the metastable regions A or B.
    # Assume data is shape (n,2).
    mask_A = (data[:,0]- CENTER_A[0])**2 + (data[:,1] - CENTER_A[1]) ** 2 > RADIUS_A ** 2
    mask_B = (data[:,0]- CENTER_B[0])**2 + (data[:,1] - CENTER_B[1]) ** 2 > RADIUS_B ** 2
    mask = np.logical_and(mask_A, mask_B)
    return data[mask]

def points_to_csv(data, filepath = None):
    # goal is to save points in R^2 to 
    # a csv with first line containing "x, y"
    # as column headers.
    # assume data is (n,2).
    df = pd.DataFrame(data, columns = [X_COORD, Y_COORD])
    if filepath == None:
        now = datetime.now().isoformat(timespec='minutes')
        df.to_csv(PROCESSED_DATA_DIR/f"metadynamics_sample_{now}.csv", index = False)
    else: 
        df.to_csv(filepath, index = False)

def make_ellipse_pts_csv():
    # Basically just add headers to the 
    # MaierStein_pts.csv from the external dir, but leave that unchanged
    # and create new one with the headers

    logger.info("Processing dataset...")
    pts_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_pts.csv", \
                         header = None)
    pts_df = pts_df.rename(columns = { 0 : X_COORD, 1 : Y_COORD })
    pts_df.to_csv(RAW_DATA_DIR/"Ellipse_pts.csv", index = False)
    logger.success("Processing dataset complete.")
    pass

def make_finite_elements_prediction_csv():
    # Basically just put the "MaierStein_q_FEM.csv", 
    # which is one column of data, into a csv with 3 columns 
    # of data.

    logger.info("Processing dataset...")
    pts_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_pts.csv", \
                         header = None)
    FEM_q_df = pd.read_csv(EXTERNAL_DATA_DIR/"MaierStein_q_FEM.csv", \
                           header = None)

    pts_df = pts_df.rename(columns = { 0 : X_COORD, 1 : Y_COORD })
    FEM_q_df = FEM_q_df.rename(columns = { 0 : Q_LABEL })

    FEM_pts_q_df = pd.concat([pts_df, FEM_q_df], axis = 1)

    FEM_pts_q_df.to_csv(PROCESSED_DATA_DIR / "FEM_Commitor.csv", index=False)
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


class EulerMaruyama():
    """
    Sample the SDE
    d(x,y) = b(x,y) dt + \sqrt{\epsilon} dw_t

    using the Euler Maruyama discretization.
    """
    def __init__(self, delta_t):
        # delta_t is the time step
        # for example, use delta_t = 1e-4.
        self.delta_t = delta_t

    def get_sample_path(self, n_iterations, init_point):
        init_point = np.array(init_point)
        n_iterations = int(n_iterations)
        sample_path = np.zeros((n_iterations,2))
        sample_path[0] = init_point
        # run simulation
        for i in tqdm(range(n_iterations-1)):
            # get value of b at current sample point
            x = sample_path[i,0]
            y = sample_path[i,1]
            b = np.array([x - x**3 - BETA * x * y ** 2, \
                          - (1 + x**2) * y])
            # get guassian sample with mean 0 and variance delta_t
            # because delta_t increments of Brownian motion have this property.
            # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
            stan_dev = np.sqrt(self.delta_t)
            mean = 0.0
            dW = np.random.normal(loc = mean, scale = stan_dev, size = (2,)) 
            # compute next sample point: 
            sample_path[i+1] = sample_path[i] + b * self.delta_t + np.sqrt(EPSILON) * dW
        return sample_path


class Metadynamics():
    """
    Sampling according to Section 3.1 in 
    "reports/A_neural_network_based_solver_for_quantifying_transition_processes_in_nongradient_SDEs.pdf"
    """
    def __init__(self, delta_t = 1e-3, iterations = int(1e5),\
                  tau = int(1e3), height_param = 0.2, width_param = 0.025, delta = 0.007):
        """
        - delta_t: the time step for the dynamics
        - iterations: how many iterations to run the modified Euler Maruyama simulation
        - tau: deposit a Gaussian every tau steps
        - height_param: scales the height of the Gaussian deposit (e.g.
            higher value means elongated Gaussian bump)
        - width_param: scales the width of the Gaussian bumps
        
        - iterations \approx tau * number of deposits
        So that iterations/tau is approximately the number of Gaussians that
        will be deposited.
        """
        self.delta_t = delta_t
        self.iterations = iterations
        self.tau = tau
        self.height_param = height_param
        self.width_param = width_param
        self.delta = delta

    def get_guassian_locations(self):
        # 2 = in_size of neural network.
        # sample_path = torch.zeros((iterations,2)).requires_grad_(True)
        sample_path = np.zeros((self.iterations,2))

        ind = [] # keep track of the indices for the Gaussian deposits

        sqrt_eps = math.sqrt(EPSILON)

        for i in tqdm(range(self.iterations-1)): # subtract 1 because of sample_path[i+1] below
            if i % self.tau == 0:
                ind += [i]
            
            # get guassian sample with mean 0 and variance delta_t.
            # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
            # stan_dev = math.sqrt(delta_t) * torch.ones(2)
            stan_dev = math.sqrt(self.delta_t)
            # mean = torch.zeros(2)
            mean = 0
            dW = np.random.default_rng().normal(mean, stan_dev, (2,))

            # dW = torch.normal(mean = mean, std = stan_dev) 

            # find out how many guassian deposits have been dropped so far:
            # curr_num_g_l = i//tau + 1 # rounds down to nearest integer

            # pick out the guassian locations from the sample_path: 
            # ind = torch.linspace(0,tau * (curr_num_g_l-1),curr_num_g_l, dtype = int)
            g_l = sample_path[ind]

            b= self.B(sample_path[i], g_l)

            # work around the error: "a view of a leaf Variable that requires grad is being used in an in-place operation."
            #with torch.no_grad():
            #    sample_path[i+1] = sample_path[i] +  b * self.delta_t + sqrt_eps * dW

            sample_path[i+1] = sample_path[i] + b * self.delta_t + sqrt_eps * dW

        # pick out the guassian locations from the sample_path: 
        #ind = torch.linspace(0,tau * (num_g_l-1),num_g_l, dtype=int)
        # ind = np.linspace(0, iterations - 1, self.tau)

        return sample_path[ind], sample_path

    def run_biased_sample_path(self, num_pts, g_l):
        # sample_path = torch.zeros((num_pts,2)).requires_grad_(True)
        sample_path = np.zeros((num_pts, 2))

        sqrt_eps = math.sqrt(EPSILON)
        for i in tqdm(range(num_pts-1)):
            # get guassian sample with mean 0 and variance delta_t.
            # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
            # stan_dev = math.sqrt(self.delta_t) * torch.ones(2)
            stan_dev = math.sqrt(self.delta_t)
            # mean = torch.zeros(2)
            mean = 0.0
            # dW = torch.normal(mean = mean, std = stan_dev)
            dW = np.random.default_rng().normal(mean, stan_dev, (2,))

            # get biased b field vector
            b= self.B(sample_path[i], g_l)

            # work around the error: "a view of a leaf Variable that requires grad is being used in an in-place operation."
            with torch.no_grad():
                sample_path[i+1] = sample_path[i] +  b * self.delta_t + sqrt_eps * dW

        return sample_path

    def V_bias(self, sample_point, g_l):
        """
        This scalar valued function will be differentiated, 
        so it needs to be in Torch, not NumPy.
        """
        # sample_point is shape (2,)
        # g_l is shape (x,2) for x a natural number

        assert sample_point.shape == torch.Size([2])

        # we need |g_l| copies of  
        s = sample_point.repeat(g_l.shape[0],1) # what does this do?
        # s = np.tile(sample_point, (g_l.shape[0], 1))

        arg = ((g_l[:,0]-s[:,0])/self.width_param)**2 + \
            ((g_l[:,1]-s[:,1])/self.width_param)**2

        v_bias = torch.sum(self.height_param * torch.exp(-1/2 * arg))
        # v_bias = np.sum(self.height_param * np.exp(-1/2 * arg))
    
        return v_bias

    def B(self, sample_point, g_l):
        """
        B = b_field(sample_path[i]) + \nabla V_bias(sample_path[i]). 
        """
        # convert to torch tensor to do the differentiation:
        sam_pt_tch = torch.tensor(sample_point, requires_grad=True)
        g_l_tch = torch.tensor(g_l, requires_grad=True)

        # self.V_bias is the function to differentiate
        # self.V_bias is a scalar-valued function
        V_bias_output = self.V_bias(sam_pt_tch, g_l_tch)

        # vector_1 = torch.ones_like(V_bias_output)

        #gradient_V_bias = torch.autograd.grad(outputs= V_bias_output,\
        #    inputs= (sample_point,g_l),grad_outputs=vector_1)[0]

        gradient_V_bias = torch.autograd.grad(outputs= V_bias_output,\
                    inputs= (sam_pt_tch, g_l_tch))[0]
        # I don't think I need grad_outputs...
        # Isn't 
        # what is happening here? 

        assert gradient_V_bias.shape == torch.Size([2]) # first component is 
        # partial derivate wrt 1st coordinate, the second component 
        # is the partial derivative wrt 2nd coordinate

        # Covert to numpy, since the differentiation
        # is over:
        grad_V_bias_np = gradient_V_bias.detach().cpu().numpy()

        x = sample_point[0]
        y = sample_point[1]
        #b_field = torch.tensor([x-x**3-BETA* x* y**2, -(1+x**2)*y])
        b_field = np.array([x-x**3-BETA* x* y**2, -(1+x**2)*y])

        out = b_field - grad_V_bias_np

        assert out.shape == torch.Size([2])

        return out

def get_delta_rarified_sample_path(sample_path, delta):
    """
    - sample_path: the sample path to be delta-rarified
    - delta: parameter for delta-rarification, e.g.
                each point will be delta distance away from any neighbors
    """
    N = len(sample_path)
    mask = torch.ones((N,))
    delta2 = delta**2
    for k in tqdm(range(N)):
        if mask[k]==1:
            dist2 = (sample_path[:,0]-sample_path[k,0])**2 + (sample_path[:,1]-sample_path[k,1])**2
            ind = torch.argwhere(dist2 < delta2)
            mask[ind] = 0
            mask[k]=1

    ind = torch.argwhere(mask==1)
    out = torch.squeeze(sample_path[ind])
    return out

def delta_rarify(data, delta = 7e-3):
    # iteratively enforce that all points 
    # are distance delta from each other.
    # Assume data is numpy array
    # of shape (n,2). 
    N = len(data)
    mask = np.ones((N,))
    delta2 = delta ** 2
    for k in tqdm(range(N)):
        if mask[k]==1:
            # distance to point at index k, squared:
            dist2 = (data[:,0] - data[k,0])**2 + (data[:,1]-data[k,1])**2
            mask[dist2 < delta2] = 0
            mask[k] = 1
    return data[mask == 1]

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # Stuff the Makefile will do: 
    logger.info("Converting External Data to Desired Format")
    make_finite_elements_prediction_csv()
    make_ellipse_pts_csv()
    make_bdy_csv()

    

if __name__ == "__main__":
    app()
