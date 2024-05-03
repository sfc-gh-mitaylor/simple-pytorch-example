#!/opt/conda/bin/python3

import argparse
import logging
import os
import sys

import json
import numpy as np
import pandas as pd
import os
import sys
import time

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import sproc, col
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T

from snowflake.snowpark.types import PandasDataFrameType, IntegerType, StringType, FloatType, Variant
from snowflake.snowpark.exceptions import SnowparkSQLException

import torch
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from snowflake.ml.fileset import fileset

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import *

# Environment variables below will be automatically populated by Snowflake.
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# Custom environment variables
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")


def get_arg_parser():
    """
    Input argument list.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="query text to execute")
    parser.add_argument(
        "--result_table",
        required=True,
        help=
        "name of the table to store result of query specified by flag --query")

    return parser


def get_logger():
    """
    Get a logger for local logging.
    """
    logger = logging.getLogger("job-tutorial")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_login_token():
    """
    Read the login token supplied automatically by Snowflake. These tokens
    are short lived and should always be read right before creating any new connection.
    """
    with open("/snowflake/session/token", "r") as f:
        return f.read()


def get_connection_params():
    """
    Construct Snowflake connection params from environment variables.
    """
    if os.path.exists("/snowflake/session/token"):
        return {
            "account": SNOWFLAKE_ACCOUNT,
            "host": SNOWFLAKE_HOST,
            "authenticator": "oauth",
            "token": get_login_token(),
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA
        }
    else:
        return {
            "account": SNOWFLAKE_ACCOUNT,
            "host": SNOWFLAKE_HOST,
            "user": SNOWFLAKE_USER,
            "password": SNOWFLAKE_PASSWORD,
            "role": SNOWFLAKE_ROLE,
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA
        }


def run_job():
    """
    Main body of this job.
    """
    logger = get_logger()
    logger.info("Job started")

    # Generate Some Data
    from sklearn.datasets import make_classification
    import pandas as pd
    columns = [str(i) for i in range(0,10)]
    X,y = make_classification(n_samples=100000, n_features=10, n_classes=2)
    X = np.array(X, dtype=np.float32)
    df = pd.DataFrame(X, columns=columns)
    feature_cols = ["COL" + i for i in df.columns]
    df.columns = feature_cols
    df['Y'] = y
    X_tens = torch.tensor(X)
    y_tens = torch.tensor(y)
    # convert into PyTorch tensors
    X_tens = torch.empty_like(X_tens).copy_(X_tens)
    y_tens = torch.empty_like(y_tens).copy_(y_tens).reshape(-1, 1)
    loader = DataLoader(list(zip(X_tens,y_tens)), shuffle=True, batch_size=16)

    logger.info("Tensors created and DataLoader prepped")

    # Defin a Model
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.ReLU(),
            )

        def forward(self, tensor:torch.Tensor):
            return self.model(tensor)

    def train_model(loader):
        n_epochs = 5
        device = 'cuda'
        model = MyModel()

        logger.info("Do the CUDA Thing")
        # import os
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)
        # rank = 1 #  int(os.environ['LOCAL_RANK']) # Used to identify the local node
        # world_size = 4 # int(os.environ['WORLD_SIZE']) # Total number of GPUs available
        # dist.init_process_group("nccl", rank=rank, world_size=world_size) # Use NCCL backend for distributed GPU training
        # torch.cuda.set_device(rank)

        device = torch.cuda.current_device()
        # print(f"rank = {rank}, using device {device}, in world size {world_size}")

        model = model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        start_time = time.time()

        logger.info("Run the Training")

        # Training step
        for epoch in range(n_epochs):
            current_loss = 0.0
        
            for batch, (X, y) in enumerate(loader):
        
                X_batch, y_batch = X.to(device), y.to(device)
                # forward pass
                y_pred = model(X_batch)
        
                # compute loss
                loss = loss_fn(y_pred.float(), y_batch.float())
        
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                current_loss += loss.item()
        
            if epoch % 10 == 0:
                print(f"Loss after epoch {epoch}: {current_loss}")
                for param in model.parameters():
                    print(param.data)
        
        end_time = time.time()
        print('Model training complete.')
        print(f'Training time: {end_time-start_time}')
        return model

    logger.info("Start Training")

    model = train_model(loader)

    logger.info("Job finished")


if __name__ == "__main__":
    run_job()

