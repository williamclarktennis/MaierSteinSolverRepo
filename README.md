# Maier Stein Solver

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Solve for committor function using neural networks.

## Background and Physics-informed Neural Network Idea

The Maier-Stein system is described by the SDE
$$
    d \begin{bmatrix}
    x \\ y
    \end{bmatrix} = 
    \begin{bmatrix}
        x - x^3 - \beta x y^2 \\
        -(1 + x^2) y
    \end{bmatrix} dt + 
    \sqrt{\epsilon} \, dW,
$$
where $W_t$ is Brownian motion.

Let the drift field be denoted by $b(x,y): \mathbb R^2 \to \mathbb R^2$.
This drift field has two stable equilibria at the points
$(-1,0)$ and $(1,0)$.

Here is how a typical trajectory looks, together with the drift field drawn in. 

![Trajectory](reports/figures/right-start-Eu-Mar.png)

The equilibrium points are surrounded by two circles, 
which are sinks for trajectories: where they are bound to 
settle into, except for the random influence of Brownian motion.

We call these two encircled regions $A$ (on the left) and $B$ (on the right).
If a point is not inside $A$ or $B$, then it is in
$\Omega$, which is called the transition region.

We want to know about probabilities for how a particle travels
in this system.
One particularly important probability is the following one. 
This is the probability that a particle located
at $(x,y)$ will hit $B$ before it hits $A$.
This probability, denoted $q(x,y)$, 
is known as the **committor function**.
For example,
$q|_{\partial A} = 0$ because, how could it hit $B$ before hitting
$A,$ since it is hitting $A$ already! Conversely, 
$q|_{\partial B} = 1$.
 

The committor function satisfies the Kolmogorov backward equation:
$$
\begin{cases}
b \cdot \nabla q + \frac{\epsilon}{2} \Delta  q = 0 & (x,y) \in \Omega \\
q|_{\partial A} = 0, \qquad q|_{\partial_B}  = 1
\end{cases}
$$
where $\Delta = \partial_{xx} + \partial_{yy}$ is the Laplacian.
We denote 
$b \cdot \nabla + \frac{\epsilon}{2} = \mathcal{L}$.

**How do we solve for the committor function?**

The approach in this repository is to initialize
a neural network, and then train it to be the committor function!
Our neural network will have two input nodes, one 
for each coordinate of the spatial region of the system. 
And it will have one output node for the probability of 
hitting $B$ before $A$, given the starting point (the input nodes).

So the neural network is just a function $NN: \mathbb R^2 \to \mathbb R$.
(Of course it will have a bunch of hidden layers.)
So we want to make sure that 
$$NN(x,y) \approx q(x,y)$$
for all $x,y \in \Omega$.

How do we do this? The idea is again simple. 
We just penalize the $NN$ whenever 
$\mathcal{L} (NN) (x,y) \ne 0$ (recall that neural networks
are (at least) twice weakly differentiable).
And we also penalize the neural network
if it doesn't satisfy $NN|_{\partial A} = 0$ and 
$NN|_{\partial B} = 1$.

This means that the loss function during neural network training
will be of the form

$$
\sum_{\text{training pts}}\lVert \mathcal{L} NN \rVert^2 + \sum_{\partial A} (NN - 0)^2 +
\sum_{\partial B} (1 - NN)^2. 
$$

This is called
a Physics-informed neural network 
because the "physics" of the system, e.g. the operator $\mathcal{L}$,
is encoded into the loss function during training.

## Project Documentation

The documentation for this project can be found at the website: <https://williamclarktennis.github.io/MaierSteinSolverRepo/>.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A quarto webpage
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Maier-Stein-Solver and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── src
    └── Maier-Stein-Solver   <- Source code for use in this project.
        │
        ├── __init__.py             <- Makes Maier-Stein-Solver a Python module
        │
        ├── config.py               <- Store useful variables and configuration
        │
        ├── dataset.py              <- Scripts to download or generate data
        │
        ├── features.py             <- Code to create features for modeling
        │
        ├── modeling             
        │   ├── __init__.py 
        │   ├── pinn.py             <- Code to store loss function logic and neural network architecture
        │   ├── predict.py          <- Code to run model inference with trained models          
        │   └── train.py            <- Code to train models
        │
        └── plots.py                <- Code to create visualizations
```

--------

