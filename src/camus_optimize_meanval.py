import time
import os, sys
import argparse
from argparse import Namespace
from multiprocessing import Process
from typing import Dict, Any, Optional
import random
import math

import torch
from torch import nn
import numpy as np

import dopt
from dopt import NEIOptimizer, Trainer, Server
from dopt.utils import get_output_shape
from camus_objective_5folds_meanval import run_kfolds # The objective function

import warnings
warnings.filterwarnings("ignore")


# The configurations
CONFIG = {}
# List of computers you have ssh access to. We'll optimize our model on these computers
#
CONFIG["computer_list"] = {
    "acet": [
        '[some-username]@[some-hostname]', # TODO: replace [some-username] and [some-hostname]
    ],
#     "tung-torch": ['john@jvs008.bucknell.edu'] # Example
}
# Commands to run on target machines
CONFIG["commands"] = {
    "acet": "sleep 30 && module switch python/3.9-deeplearn" + \
                   " && pkill -9 python" + \
                   # "; export LD_LIBRARY_PATH=/usr/remote/lib:/usr/remote/anaconda-3.7-2020-05-28/lib" + \
                   " && python3 ~/[some-path]/camus_optimize_meanval.py --run_as trainer", # This is required. TODO: replace [some-path]
    # "localhost": "/opt/anaconda/envs/jupyter37/bin/python ~/pj/camus_segmentation/src/camus_optimize_meanval.py --run_as trainer" # Example
}
# Server and trainer info
CONFIG["server"] = {
    "host": "[your-hostname]", # TODO: replace [your-hostname] (your server hostname)
    "port": 15555,
}
CONFIG["trainer"] = {
    "username": "[your-username]", # TODO: replace [your-username]
    "num_constraints": 1,
    "verbose": False
}
CONFIG["optimizer"] = {
    "view": "2CH",
    "bounds": {
        "log_learning_rate": [-9, 2], # -6.2146080984222
        "batch_size": [2, 10],        # 16
        "group_vs_batch_norm": [0, 1], # Group 0
        "n_filters_0": [16, 32],      # 32
        "n_filters_1": [57, 128],     # 64
        "n_filters_2": [153, 256],    # 128
        "n_filters_3": [281, 512],    # 256
        "n_filters_4": [537, 1024],   # 512
        "num_groups": [2, 24],        # 16
        "log_weight_decay": [-9, -2]  # -13.815510557964
    },
    "initial_candidates": [],
    "device": "cuda:0",
    "seed": 9237, # arbitrary seed
    "filename": "camus_UNetlike_2CH_meanval.dopt" # Result file
}
print("Using dopt version:", dopt.__version__)


def better_round(x):
    return math.floor(x-0.5) + 1

def n_filters_constraint(n_filters):
    feasibility = -0.9 # Default: Feasible
    for i in range(1, 5): # Encoders should increase filter size
        if n_filters[i-1] >= n_filters[i]:
            feasibility += 1
    return feasibility
        
def objective_function(candidate):
    norm_type = "groupnorm" if candidate["group_vs_batch_norm"] < 0.5 else "batchnorm"
    num_groups = better_round(candidate["num_groups"])
    
    n_filters = []
    for i in range(5):
        float_filters = candidate["n_filters_"+str(i)]
        if norm_type == "batchnorm":
            n_filters.append(better_round(float_filters))
        elif norm_type == "groupnorm":
            n_filters.append(better_round(float_filters / num_groups) * num_groups)
        else:
            raise Exception("Unknown norm_type: " + norm_type)
            
    feasibility = n_filters_constraint(n_filters)
    if feasibility > 0:
        observation = {
            "objective": [0.001, 0.001],
            "constraints": [feasibility]
        }
        return observation
            
    # Simulate input
    input_args = Namespace(
        view=CONFIG["optimizer"]["view"],
        num_folds=5,
        loss="better",
        windowing_scale=[0.5, 1],
        rotation_scale=5.0,
        noise_scale=[0.0, 0.15],
        training_augment=True,
        n_channels=1, 
        n_classes=4, 
        n_filters=n_filters,
        normalization=norm_type, 
        num_groups=num_groups,
        num_epochs=300, #<--------------------------------------------------------- 
        batch_size=better_round(candidate["batch_size"]),
        effective_batchsize=1,
        learning_rate=np.exp(candidate["log_learning_rate"]),
        weight_decay=np.exp(candidate["log_weight_decay"]),
        patienceLimit=41,
        patienceToLRcut=10,
        howOftenToReport=10,
        loss_weights=[1,1,1,1],
        image_size=[256, 256],
    )
    
    mean, var, all_dices = run_kfolds(input_args, candidate["id"])
    
    observation = {
        "objective": [1-mean, var],
        "constraints": [feasibility],
        "all_dices": all_dices
    }

    return observation

# Calls start_optimizer and start_trainers simultaneously
def start_server():
    optimizer = NEIOptimizer(
        CONFIG["optimizer"]["filename"], 
        CONFIG["optimizer"]["bounds"], 
        initial_candidates=CONFIG["optimizer"]["initial_candidates"],
        device=CONFIG["optimizer"]["device"],
        seed=CONFIG["optimizer"]["seed"]
    )
    server = Server(optimizer, CONFIG) 
                    # verbose=CONFIG["server"]["verbose"])
    server.run()
    
def start_trainers():
    trainer = Trainer(
        objective_function, 
        CONFIG["trainer"]["username"],
        CONFIG["server"]["host"],
        CONFIG["server"]["port"],
        num_constraints=CONFIG["trainer"]["num_constraints"],
        verbose=CONFIG["trainer"]["verbose"]
    )
    trainer.run()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='distributed_optimizer.py',
                                     description='''Optimize objective function of specified by a `Trainer`''')
    parser.add_argument('--run_as', action='store', dest='run_as',
                           help='Specify the role of the machine (server or trainer). Defaults to server',
                           type=str, required=False,
                           default="server")
    args = parser.parse_args()
    
    # Can modify these code to accomodate more options
    # E.g. Run different Trainers on same task
    if args.run_as == "server":
        start_server()
    elif args.run_as == "trainer":
        start_trainers()
    
    
    
