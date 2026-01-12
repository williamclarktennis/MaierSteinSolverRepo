from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import math

from MaierSteinSolver.config import IN_SIZE, OUT_SIZE

app = typer.Typer()


class NeuralNetwork(nn.Module):
    """
    nn.Linear documentation: 
    (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    Input: (*, H_{in}) where * means any number of 
    dimensions including none and H_{in} =in_features.

    Output: (*, H_{out}) where all but the last dimension 
    are the same shape as the input and H_{out} =out_features.
    """
    
    def __init__(self, layer_array):
        super().__init__()

        assert type(layer_array) == type([])

        self.linears = nn.ModuleList(\
            [nn.Linear(layer_array[i],layer_array[i+1]) \
             for i in range(len(layer_array)-1)])

    def forward(self, X):

        assert X.shape[-1] == IN_SIZE

        # define tanh activation func
        tanh = nn.Tanh()
        # define sigmoid activation function
        sig = nn.Sigmoid()

        for i in range(len(self.linears)-1):
            X = self.linears[i](X)
            X = tanh(X)

        X = self.linears[-1](X)
        X = sig(X)

        return X
    

def get_Laplacian_partialx_partialy(q_NN: NeuralNetwork, X: torch.tensor)\
      -> torch.tensor:
    """
    
    """
    assert X.shape[-1] == IN_SIZE

    # get outputs 
    q_NN_outputs = q_NN(X)

    # specify the vector in the vector Jacobian product: 
    vector_1 = torch.ones_like(q_NN_outputs)

    # compute vector Jacobian product
    jacobian_x_y = torch.autograd.grad(outputs= q_NN_outputs,\
        inputs= X,grad_outputs=vector_1,allow_unused=True,\
            retain_graph=True,create_graph=True)

    # get first column of vector Jacobian product, i.e: partial 
    # deriv of q_NN with respect to x
    partial_x = jacobian_x_y[0][:,0]

    # ditto:
    partial_y = jacobian_x_y[0][:,1]

    # specify next vector: 
    vector2 = torch.ones_like(partial_x)
    
    # compute jacobian of partial_x
    jacobian_xx_xy = torch.autograd.grad(outputs = partial_x, inputs = X,\
        grad_outputs= vector2, allow_unused=True, retain_graph=True)

    # compute jacobian of partial_y
    jacobian_yx_yy = torch.autograd.grad(outputs = partial_y, inputs = X,\
        grad_outputs= vector2, allow_unused=True, retain_graph=True)
    
    laplacian = jacobian_xx_xy[0][:,0] + jacobian_yx_yy[0][:,1]

    laplacian = laplacian[:,None]
    partial_x = partial_x[:,None]
    partial_y = partial_y[:,None]

    assert laplacian.shape[-1] == OUT_SIZE
    assert partial_x.shape[-1] == OUT_SIZE
    assert partial_y.shape[-1] == OUT_SIZE

    return laplacian, partial_x, partial_y

def L_q(q_NN: NeuralNetwork, X:torch.tensor) -> torch.tensor:
    """
    Apply the Kolomogorov backwards operator to q_NN at X
    """

    assert X.shape[-1] == IN_SIZE

    x = X[:,0][:,None]
    y = X[:,1][:,None]

    assert x.shape[-1] == OUT_SIZE
    assert y.shape[-1] == OUT_SIZE

    laplacian_q, partial_x, partial_y = \
        get_Laplacian_partialx_partialy(q_NN, X)

    lq = (x-x**3-10*x*y**2)* partial_x - (1+x**2)*y * partial_y \
          + 0.1/2 * laplacian_q

    assert lq.shape[-1] == OUT_SIZE

    return lq

class PINNTrainingVarTrainData():
    """
    Training implements the PINN loss function model where
    the model to be trained will learn the boundary conditions. 
    """
    def __init__(self,NN,optimizer,loss_fn, epochs, alpha, transition_region_features, bdy_A_features, bdy_B_features):
        self.NN = NN
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.alpha = alpha

        # work around the error of inputting numpy data into torch nn:
        mid_pts = torch.tensor(transition_region_features).float()
        bA = torch.tensor(bdy_A_features).float()
        bB = torch.tensor(bdy_B_features).float()

        mid_pts.requires_grad_(True)
        bA.requires_grad_(True)
        bB.requires_grad_(True)

        self.training_points_on_bdy_of_A, self.training_points_on_bdy_of_B = \
            bA, bB

        self.training_points_not_in_A_or_B = mid_pts

        labels_for_training_pts_not_in_A_or_B = torch.zeros(\
            (len(self.training_points_not_in_A_or_B),OUT_SIZE))
        train_data = TensorDataset(self.training_points_not_in_A_or_B,\
                                    labels_for_training_pts_not_in_A_or_B)
        self.train_dataloader = DataLoader(train_data, batch_size=64,\
                                            shuffle=True)
    
    def train(self):
        epochs = self.epochs
        loss_plot = torch.zeros(0)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss_plot = self.single_training_loop(loss_plot)
        print("Done!")

        return loss_plot

    def single_training_loop(self, loss_plot):
        size = len(self.train_dataloader.dataset)
        self.NN.train()
        
        for batch, (X,y) in enumerate(self.train_dataloader):
            
            # compute prediction error
            pred_pts_not_in_A_or_B = L_q(self.NN,X)
            pred_pts_on_bdy_A = math.sqrt(self.alpha)\
                 * self.NN(self.training_points_on_bdy_of_A)
            pred_pts_on_bdy_B = math.sqrt(self.alpha)\
                  * self.NN(self.training_points_on_bdy_of_B)\
                      - math.sqrt(self.alpha)
            
            # put errors together into one vector
            pred_vector = torch.cat((pred_pts_not_in_A_or_B, \
                                     pred_pts_on_bdy_A, \
                                        pred_pts_on_bdy_B))

            assert pred_vector.shape[-1] == OUT_SIZE
            truth = torch.zeros_like(pred_vector)
            loss = self.loss_fn(pred_vector,truth)

            # backpropogation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_plot = torch.cat((loss_plot, torch.tensor([loss])))
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss_plot
