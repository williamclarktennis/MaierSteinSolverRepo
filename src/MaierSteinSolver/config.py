from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# the maier stein model parameters
CENTER_A = [-1.0,0.0]
CENTER_B = [1.0,0.0]
RADIUS_A = 0.3
RADIUS_B = 0.3
Q_BDY_A = 0.0 # committor on boundary of A
Q_BDY_B = 1.0 # committor on boundary of B
LEFT_LIM = -1.5 # as far to the left in the 2D plane that we want to
                # predict a committor solution
RIGHT_LIM = 1.5 # ditto but for the right
UP_LIM = 0.75 # ditto but for the upper limit on the y-axis
LOW_LIM = -0.75 # ditto but for the lower limit on the y-axis

# input size and output size for neural network based on the Maier Stein model
IN_SIZE = 2
OUT_SIZE = 1

# common csv headers:
X_COORD = "x"
Y_COORD = "y"
Q_LABEL = "committor value"