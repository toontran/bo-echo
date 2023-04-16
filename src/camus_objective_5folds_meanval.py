import sys
import os
from argparse import Namespace
from contextlib import contextmanager
import json
import math
import logging

import numpy as np

# torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from torch.autograd import Variable
from torch.nn import Module, Conv2d, Parameter

# Our code
from camus_train_segment import build_DataLoaders, run_actual_experimental_loop
from utils.camus_config import CAMUS_CONFIG
from utils.camus_load_info import (make_camus_echo_dataset,
                                   split_camus_echo,
                                   make_camus_EHR,
                                   camus_generate_folds)
from utils.torch_utils import run_validation, BetterLoss

# set random seeds for reproducibility
import random
torch.manual_seed(420)
torch.cuda.manual_seed_all(420)
np.random.seed(420)
random.seed(420)


# Util: Mute prints within
@contextmanager
def mute_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    yield
    sys.stdout = original_stdout


class GroupNorm2D(Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm2D, self).__init__()
        self.weight = Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
    

class ResidualConvBlock(Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', expand_chan=False, num_groups=16):
        super(ResidualConvBlock, self).__init__()

        self.expand_chan = expand_chan
        if self.expand_chan:
            ops = []

            ops.append(nn.Conv2d(n_filters_in, n_filters_out, 1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out, num_groups=num_groups))

            ops.append(nn.ReLU(inplace=True))

            self.conv_expan = nn.Sequential(*ops)

        ops = []
        for i in range(n_stages):
            if normalization != 'none':
                ops.append(nn.Conv2d(n_filters_in, n_filters_out, 3, padding=1))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm2d(n_filters_out))
                if normalization == 'groupnorm':
                    ops.append(GroupNorm2D(n_filters_out, num_groups=num_groups))
            else:
                ops.append(nn.Conv2d(n_filters_in, n_filters_out, 3, padding=1))

            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    # I think this is adding and not concatenating...
    def forward(self, x):
        if self.expand_chan:
            x = self.conv(x) + self.conv_expan(x)
        else:
            x = (self.conv(x) + x)

        return x
    
# Now the down and upsampling layers

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', num_groups=16):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out, num_groups=num_groups))
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', num_groups=16):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out, num_groups=num_groups))
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

'''
UNetLike is a UNet-like 2D pixel-wise (semantic) segmentation pipeline with residual (+, not concat) 
connections. 
'''    
class UNetLike(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters, normalization='none', num_groups=16):
        super(UNetLike, self).__init__()

        if n_channels > 1:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters[0], normalization=normalization, num_groups=num_groups, expand_chan=True)
        else:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters[0], normalization=normalization, num_groups=num_groups)

        self.block_one_dw = DownsamplingConvBlock(n_filters[0], n_filters[1], normalization=normalization, num_groups=num_groups)

        self.block_two = ResidualConvBlock(2, n_filters[1], n_filters[1], normalization=normalization, num_groups=num_groups)
        self.block_two_dw = DownsamplingConvBlock(n_filters[1], n_filters[2], normalization=normalization, num_groups=num_groups)

        self.block_three = ResidualConvBlock(3, n_filters[2], n_filters[2], normalization=normalization, num_groups=num_groups)
        self.block_three_dw = DownsamplingConvBlock(n_filters[2], n_filters[3], normalization=normalization, num_groups=num_groups)

        self.block_four = ResidualConvBlock(3, n_filters[3], n_filters[3], normalization=normalization, num_groups=num_groups)
        self.block_four_dw = DownsamplingConvBlock(n_filters[3], n_filters[4], normalization=normalization, num_groups=num_groups)
        
        # if concat: torch upsampling instead

        self.block_five = ResidualConvBlock(3, n_filters[4], n_filters[4], normalization=normalization, num_groups=num_groups)
        self.block_five_up = UpsamplingDeconvBlock(n_filters[4], n_filters[3], normalization=normalization, num_groups=num_groups)

        self.block_six = ResidualConvBlock(3, n_filters[3], n_filters[3], normalization=normalization, num_groups=num_groups)
        self.block_six_up = UpsamplingDeconvBlock(n_filters[3], n_filters[2], normalization=normalization, num_groups=num_groups)

        self.block_seven = ResidualConvBlock(3, n_filters[2], n_filters[2], normalization=normalization, num_groups=num_groups)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters[2], n_filters[1], normalization=normalization, num_groups=num_groups)

        self.block_eight = ResidualConvBlock(2, n_filters[1], n_filters[1], normalization=normalization, num_groups=num_groups)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters[1], n_filters[0], normalization=normalization, num_groups=num_groups)

        self.block_nine = ResidualConvBlock(1, n_filters[0], n_filters[0], normalization=normalization, num_groups=num_groups)

        self.out_conv = nn.Conv2d(n_filters[0], n_classes, 1, padding=0)
        
        self.final_sig = torch.nn.Sigmoid()
        
        # Cross entroy loss obviates the need to use this
        # See https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
#         self.softmax = nn.Softmax2d(dim=0)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4 # Here's where we could try concat instead.

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        f_out = self.out_conv(x9)
        
        out = self.final_sig(f_out)

        return out

  
    
def report_validation_sets(net_seg, datadict, args):
    '''Report dice performance on the train/valid/test sets. '''
    weights = np.array(args.loss_weights)
    train_loss, train_output_seg, train_input_img, train_label, train_dices = \
        run_validation(net_seg,
                       data_iterator=datadict['train'],
                       criterion=BetterLoss(weights), 
                       keys = ['images', 'labels'], 
                       do_dice = True)
    valid_loss, valid_output_seg, valid_input_img, val_label, valid_dices = \
        run_validation(net_seg, 
                       data_iterator=datadict['valid'],
                       criterion=BetterLoss(weights),
                       keys = ['images', 'labels'], 
                       do_dice = True)
    test_loss, test_output_seg, test_input_img, test_label, test_dices = \
        run_validation(net_seg, 
                       data_iterator=datadict['test'],
                       criterion=BetterLoss(weights),
                       keys = ['images', 'labels'], 
                       do_dice = True)
    
    dices = {"train": {},
             "valid": {},
             "test": {}}
    for key in sorted(list(train_dices.keys())): # should have same keys as valid_dices
        dices["train"][key] = train_dices[key].mean(), train_dices[key].std()
        dices["valid"][key] = valid_dices[key].mean(), valid_dices[key].std()
        dices["test"][key] = test_dices[key].mean(),  test_dices[key].std()
    
    return test_loss, dices, (train_dices, valid_dices, test_dices)
    
def run_kfolds(args, num_id):
    global CAMUS_CONFIG
    
    # Create directories and generate kfolds
#     camus_prep()
    results = []
    all_dices = {"train": {},
             "valid": {},
             "test": {}}
    net_segs = []
    is_saving_models = True
    
    for fold in range(args.num_folds):
        # Remove the fold
        # Select fold number
        print(f"Fold: {fold+1}")
        args.fold = fold
        
        # Get the data
        datadict = build_DataLoaders(args)

        # Build the network and place on the gpu.
#         print('Constructing UNetLike:')
        unet_args = {
            'n_channels': args.n_channels, 
            'n_classes': args.n_classes, 
            'n_filters': args.n_filters, 
            'normalization': args.normalization,
            'num_groups': args.num_groups
        }
        net_seg = UNetLike(**unet_args)    
        net_seg = torch.nn.DataParallel(net_seg)
        net_seg.cuda();
        #print('\tTrainable params: {}'.format(sum(p.numel() for p in net_seg.parameters() if p.requires_grad)))

        # run_actual_experimental_loop() uses CAMUS_CONFIG so we're modifying it directly
        CAMUS_CONFIG["training"] = {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'effective_batchsize': args.effective_batchsize,
            'learning_rate': args.learning_rate, #0.001, # 0.002,
            'weight_decay': args.weight_decay,
            'patienceLimit': args.patienceLimit,
            'patienceToLRcut': args.patienceToLRcut,
            'howOftenToReport': args.howOftenToReport,
            'loss_weights': args.loss_weights,
            'image_size': args.image_size
        }
        # Run the experimental loop
        with mute_print():
            net_seg = run_actual_experimental_loop(net_seg, datadict, args)
        test_loss, dices, pointwise_dices = report_validation_sets(net_seg, datadict, args)
        for dataset in ("train", "valid", "test"):
            for key in dices[dataset]:
                if fold == 0:
                    all_dices[dataset][key] = [0,0]
            
                all_dices[dataset][key][0] += dices[dataset][key][0]
                all_dices[dataset][key][1] += dices[dataset][key][1]**2 # Turn back to variance to take average
        results.append(test_loss)
        
        if test_loss > 0.81:
            is_saving_models = False
            pass
        else:
            net_segs.append(net_seg)
    
    for dataset in ("train", "valid", "test"):
        for key in dices[dataset]:
            all_dices[dataset][key][0] = all_dices[dataset][key][0] / args.num_folds
            all_dices[dataset][key][1] = np.sqrt(all_dices[dataset][key][1] / args.num_folds) # Take sqrt of mean
    
    if is_saving_models:
        for fold, net_seg in enumerate(net_segs):
            # Save model for the fold
            model_fname = f"meanval_view_{args.view}_fold_{fold}_candidatenum_{num_id}.pth"
            data_folder = CAMUS_CONFIG["paths"]['CAMUS_DIR']
            torch.save(net_seg.state_dict(), data_folder+"saved_models/"+model_fname)
    
    return np.mean(results), np.std(results)**2, all_dices

def better_round(x):
    return math.floor(x-0.5) + 1

def n_filters_constraint(n_filters):
    feasibility = -0.9 # Default: Feasible
    for i in range(1, 5): # Encoders should increase filter size
        if n_filters[i-1] >= n_filters[i]:
            feasibility += 1
    return feasibility

def objective_function(candidate, num_id, view="2CH"):
    global CAMUS_CONFIG
    
    CAMUS_CONFIG["paths"]["folds_file"] = "bayes_folds.pkl"
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
        view=view,
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
        num_epochs=300, #<---- 
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
    
    mean, var, all_dices = run_kfolds(input_args, num_id)
    
    observation = {
        "objective": [1-mean, var],
        "constraints": [feasibility],
        "all_dices": all_dices
    }

    return observation


def get_observations(filename):
    observations = []
    best_mean = 0
    best_var = 10**9
    worst_mean = 10**9
    best_observation = None
    worst_observation = None
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            observation = json.loads(line)
            observation["num"] = i + 1
            observations.append(observation)
    return observations


if __name__ == '__main__':
    CAMUS_CONFIG["paths"]["folds_file"] = "bayes_folds.pkl"
    
    candidate = {
        "log_learning_rate": -6.2146080984222,
        "batch_size": 16,
        "group_vs_batch_norm": 0, # Group 0, batch 1
        "n_filters_0": 32, 
        "n_filters_1": 64,
        "n_filters_2": 128, 
        "n_filters_3": 256, 
        "n_filters_4": 512, 
        "num_groups": 16,
        "log_weight_decay": -13.815510557964
    }
    
#     objective_function(candidates_to_test[0], view="2CH")
#     objective_function(candidates_to_test[1], view="2CH")
#     objective_function(candidates_to_test[2], view="4CH")
    import time
    start = time.time()
    print(objective_function(candidate, 0, view="4CH"))
    print("Time elapsed:", start - time.time(), "(s)")
    
    
            