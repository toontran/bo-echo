'''
Joshua Stough 07/19

Train the UNetLIke model and segment the test fold, recording results in hdf5. 
Though scripted, much of this code is ripped from, for example,
notesbooks/CAMUS_UNet3.ipynb. 

run in bash like:
for i in {0..9}; do python camus_train_segment.py --fold $i --view 4CH 2>&1 | tee logs/4CH_$i.log; done

'''

import numpy as np
import pandas as pd


import sys
import tempfile
import h5py
import random
import pickle


# Parsing of arguments: https://docs.python.org/3/library/argparse.html
import argparse


# CAMUS-related
from utils.camus_config import CAMUS_CONFIG
from utils.camus_load_info import (make_camus_echo_dataset,
                                   split_camus_echo,
                                   make_camus_EHR)

from utils.camus_transforms import (LoadSITKFromFilename,
                                    SitkToNumpy,
                                    ResizeImagesAndLabels,
                                    WindowImagesAndLabels,
                                    RotateImagesAndLabels,
                                    GaussianNoiseEcho,
                                    OneHot)
from utils.camus_validate import (labNameMap,
                                  dict_extend_values,
                                  camus_dice_by_name)
from utils.torch_utils import (TransformDataset,
                               torch_collate,
                               run_training,
                               run_validation,
                               run_validation_returnAll,
                               BetterLoss,
                               DiceLoss)
from torch_models import UNetLike


# torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from torch.autograd import Variable
from torch.nn import Module

from torch.utils.data import DataLoader

# For timing.
import time
tic, toc = (time.time, time.time)

# Useful
import os
cpu_count = os.cpu_count

# For getting the segmentation from the network output
from scipy.special import softmax


# set random seeds for reproducibility
# torch.manual_seed(420)
# torch.cuda.manual_seed_all(420)
# np.random.seed(420)
# random.seed(420)
torch.manual_seed(tic())
torch.cuda.manual_seed_all(tic())
np.random.seed(int(tic()))
random.seed(int(tic()))



# Overly complicated to parse boolean: 
# https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# Function to parse the command line arguments
def parse(sample=None):
    global CAMUS_CONFIG
    parser = argparse.ArgumentParser(prog='camus_train_segment.py',
                                     description='''train UnetLike CNN for echo segmentation
                                                     on provided view and fold.''')
    
    req_group = parser.add_argument_group('required arguments:')
    req_group.add_argument('--view', action='store', dest='view',
                           help='view ["2CH","4CH"] to train', 
                           type=str, required=True)
    req_group.add_argument('--fold', action='store', dest='fold',
                           help='fold to train on and test',
                           type=int, required=True)
    
    aug_defaults = CAMUS_CONFIG['augment']
    aug_group = parser.add_argument_group('data augmentation')
    
    # boolean is a weird one.
    aug_group.add_argument('--training_augment', action='store', dest='training_augment',
                           help='perform augmentation on the training set',
                           type=boolean_string, required=False,
                           default=aug_defaults['training_augment'])
    
    aug_group.add_argument('--windowing_scale', action='store', dest='windowing_scale',
                           help='windowing augment e.g. ".5 1"', nargs=2,
                           type=float, required=False,
                           default=aug_defaults['windowing_scale'])
    aug_group.add_argument('--rotation_scale', action='store', dest='rotation_scale',
                           help='rotation aubment e.g. "5.0"', 
                           type=float, required=False,
                           default=aug_defaults['rotation_scale'])
    aug_group.add_argument('--noise_scale', action='store', dest='noise_scale',
                           help='noise augment e.g. "0 .15"', nargs=2,
                           type=float, required=False,
                           default=aug_defaults['noise_scale'])
    
    opt_group = parser.add_argument_group('optional arguments')
    opt_group.add_argument('--loss', action='store', dest='loss',
                           help='loss function to use among [better, dice]',
                           type=str, required=False,
                           default='better')
    
    if sample: # For debug
        args = parser.parse_args(sample.split())
    else:
        args = parser.parse_args()
    return parser, args

# The augment transforms I define in camus_transforms use numpy
# which is not thread-safe. D'oh!
# https://github.com/xingyizhou/CenterNet/issues/233
# https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
# https://github.com/numpy/numpy/issues/9650
# SO I'm following the advice of this link, where we use the worker_init_fn to 
# reseed numpy for each worker that the DataLoader instantiates.
# See: https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def build_DataLoaders(args):
    '''
    build_DataLoaders: return torch DataLoader objects for training,
    validation, and testing sets according to the provided fold.
    '''
    global CAMUS_CONFIG
    pathconfig = CAMUS_CONFIG['paths']
    
    # Load the dataset and ehr
    camusByPat = make_camus_echo_dataset(pathconfig['CAMUS_TRAINING_DIR'], args.view)
    
    ehrfilename = os.path.join(pathconfig['CAMUS_RESULTS_DIR'], pathconfig['ehr_file'])
    ehr = pd.read_pickle(ehrfilename)
    
    # Load the needed fold. 
    foldfilename = os.path.join(pathconfig['CAMUS_RESULTS_DIR'], pathconfig['folds_file'])
    with open(foldfilename, 'rb') as fid:
        kf = pickle.load(fid)
    train_idx, val_idx, test_idx = kf[args.fold]
    
    
    # Report info on the fold. ripped from camus_load_info
    print('build_DataLoaders: building loaders for fold {}.\n'.format(args.fold))
    print('                   :Good Med Poor; <45 >55 else')
    
    for setname, idx in zip(['training', 'valid', 'testing'], [train_idx, val_idx, test_idx]):
        foldquality = ehr.ImageQuality.loc[idx].value_counts().values
        efbins=ehr.efbin.loc[idx].value_counts()
        print('Set {:8s} ({:3d}): {:3d}  {:3d}  {:3d}; {:3d} {:3d} {:3d}'.format(setname, len(idx),\
                    foldquality[0], foldquality[1], foldquality[2],\
                    efbins[0], efbins[2], efbins[1]))

    
    # Collates the patient file names to train, val, test split.
    training_dataset, validation_dataset, test_dataset = \
        split_camus_echo(camusByPat, 
                         train_idx=train_idx, 
                         val_idx=val_idx,
                         test_idx=test_idx)
    print('{} training images, {} validation, {} test'.\
          format(len(training_dataset), len(validation_dataset), len(test_dataset)))
    
    
    
    # Define the image transforms for loading to the network.
    training_config = CAMUS_CONFIG['training']
    global_transforms = [
        LoadSITKFromFilename('images'),
        LoadSITKFromFilename('labels'),
        SitkToNumpy('images'),
        SitkToNumpy('labels', normed=False),
        ResizeImagesAndLabels(size=training_config['image_size'], image_field='images', label_field='labels'),
    ]
    
#     if args.loss == 'dice':  # Need to one-hot the outputs
#         global_transforms.append(OneHot(field = 'labels', labelCount = len(labNameMap)))

    augment_transforms = [
        WindowImagesAndLabels(args.windowing_scale, image_field='images', label_field='labels'),
        RotateImagesAndLabels(args.rotation_scale, rtype='normal', image_field='images', label_field='labels'),
        GaussianNoiseEcho(args.noise_scale, field='images')
    ]
    
    # Thinking of how best to add dice loss and its requisite one-hotted float32 output. I think I want to make
    # it a transform, so that the run_training and all that doesn't need to care later. But the rotation augmentation
    # is really weird about deciding whether labels should be long or float.  So the one-hot needs to be done at the end,
    # when it needs to be done.
    just_one_hot = [OneHot(field = 'labels', labelCount = len(labNameMap), outtype = np.float32)]
    
    
    # Construct the three data loaders
    param_Loader = {'collate_fn': torch_collate,
                    'batch_size': training_config['batch_size'],
                    'shuffle': True,
                    'num_workers': max(4, cpu_count()//2),
                    'worker_init_fn': worker_init_fn}
    
    datadict = {}
    for setname, dataset in zip(['train', 'valid', 'test'],
                                [training_dataset, validation_dataset, test_dataset]):
        
        add_transforms = augment_transforms if (args.training_augment and setname=='train') else []
        print('Constructing {} dataset, size {}, {} augmented.'.format(\
                setname, len(dataset), ['not', 'with'][len(add_transforms)>0]))
        
        if args.loss == 'dice':
            add_transforms.extend(just_one_hot)
        
        curDataset = TransformDataset(data=dataset,
                                      global_transforms=global_transforms,
                                      augment_transforms = add_transforms)
        datadict[setname] = DataLoader(curDataset, **param_Loader)
    
    
    print('build_DataLoaders: finished.')
    return datadict


def report_validation_sets(net_seg, datadict, args):
    '''
    run_validation_sets: Report dice performance on the train/valid/test sets.
    '''
    global CAMUS_CONFIG
    weights = np.array(CAMUS_CONFIG['training']['loss_weights'])
    
    train_loss, train_output_seg, train_input_img, train_label, train_dices = \
        run_validation(net_seg,
                       data_iterator=datadict['train'],
                       criterion=DiceLoss() if args.loss=='dice' else BetterLoss(weights), 
                       keys = ['images', 'labels'], 
                       do_dice = True)

    valid_loss, valid_output_seg, valid_input_img, val_label, valid_dices = \
        run_validation(net_seg, 
                       data_iterator=datadict['valid'],
                       criterion=DiceLoss() if args.loss=='dice' else BetterLoss(weights),
                       keys = ['images', 'labels'], 
                       do_dice = True)
    
    test_loss, test_output_seg, test_input_img, test_label, test_dices = \
        run_validation(net_seg, 
                       data_iterator=datadict['test'],
                       criterion=DiceLoss() if args.loss=='dice' else BetterLoss(weights),
                       keys = ['images', 'labels'], 
                       do_dice = True)
    
        
    for key in sorted(list(train_dices.keys())): # should have same keys as valid_dices
        print('{:15s}: train {:.3f}({:.3f}), valid {:.3f}({:.3f}), test {:.3f}({:.3f})'.\
              format(key, 
                     train_dices[key].mean(), train_dices[key].std(),
                     valid_dices[key].mean(), valid_dices[key].std(),
                     test_dices[key].mean(),  test_dices[key].std())) 
    
    print('\n')
    

def run_actual_experimental_loop(net_seg, datadict, args):
    '''
    run_actual_experimental_loop: As in CAMUS_UNet3.ipynb, repeatedly run a
    training epoch and test against the validation set for optimal performance.
    Return the best network.
    Some of the code is commented out from the .ipynb, but I've kept it in case
    I want to reimplement at some point.
    '''
    
    global CAMUS_CONFIG
    # We run training + validation by executing this cell
#     train_losses = []
#     validation_losses = []

    # Give all classes equal weight through the prior

#     priors = get_class_priors(datadict['train']['False'])
#     # priors = get_class_priors(train_iterator)
#     InversePriorWeights = np.max(priors)/priors # little over on LA, maybe .929/.922 for epi/endo
#     LVHeavyWeights = np.array([1, 10, 8, 5]) # hardcoded, sorry. leads to maybe .934/.924
#     EqualWeights = np.array([1,1,1,1]) # .94+/.93ish. very good.

    # For saving and restoring the best val loss model.
    best_model_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    best_epoch = -1
    min_val_loss = 1e15

    
    # Get training parameters:
    config = CAMUS_CONFIG['training']
    
    # To ensure we don't go on too long
    patienceLimit, patience = config['patienceLimit'], 0
    patienceToLRcut = config['patienceToLRcut']


    # Report dices every some number epochs.
    howOftenToReport = config['howOftenToReport'] 

    # For updating the learning rate during training.
    cur_learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']

    
    # weights = InversePriorWeights
    weights = np.array(config['loss_weights'])
    print('weights: {}'.format([(key, val) for key,val in zip(labNameMap, weights)]))
    

    print('STARTING TRAINING')

    for i in range(1, num_epochs+1):

        # Do we cut the learning rate, and if so let's reload the best model so far.
#         if patience == patienceToLRcut: # == so it only happens the first time it gets there.
#             cur_learning_rate /= 2 
#             print('\n\ncutting learning rate to {}'.format(cur_learning_rate))
#             print('Reloading best model, from epoch {}'.format(best_epoch))
#             net_seg.load_state_dict(torch.load(best_model_file.name))

        # Do we cut the learning rate, and if so let's reload the best model so far maybe.
        # The reloading didn't seem to work great.
        # Also changed the condition here, so I'll cut every patienceToLRcut epochs.
        if patience == patienceToLRcut or (patience > patienceToLRcut 
                                           and patience % patienceToLRcut == 0): 
            cur_learning_rate /= 2 
            print('\n\ncutting learning rate to {}'.format(cur_learning_rate))
            print('Reloading best model, from epoch {}'.format(best_epoch))
            net_seg.load_state_dict(torch.load(best_model_file.name))


        # Run the training epoch and validation test, and report.
        start = tic()
        train_loss, train_output_seg, train_input_img, train_label = \
            run_training(net_seg, 
                         data_iterator=datadict['train'], # train_iterator_augmented
                         effective_batchsize=config['effective_batchsize'],
                         criterion=DiceLoss() if args.loss=='dice' else BetterLoss(weights),
                         cur_learning_rate=cur_learning_rate,
                         cur_weight_decay=config['weight_decay'],
                         keys = ['images', 'labels'],
                         do_dice=False)

        valid_loss, valid_output_seg, valid_input_img, val_label = \
            run_validation(net_seg, 
                           data_iterator=datadict['valid'], # valid_iterator 
                           criterion=DiceLoss() if args.loss=='dice' else BetterLoss(weights),
                           keys = ['images', 'labels'],
                           do_dice=False)

        print('\n\nEPOCH {} of {} ({:.3f} sec)'.format(i, num_epochs, toc()-start))
        print('-- train loss {} -- valid loss {} --'.format(train_loss, valid_loss))

#         train_losses.append(train_loss)
#         validation_losses.append(valid_loss)

        # Initially, whether to visualize this iteration.
        visThisIter = (i % howOftenToReport) == 1


        ########### Check for best so far model 

        # Now save if this is the best model so far
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
#             if (i > 20):
#                 visThisIter = True # Only sometimes viz best so far. kind of slow.
            patience = 0
            best_epoch = i
            print('Epoch {}, saving new best loss model, {}'.format(i, valid_loss))
            torch.save(net_seg.state_dict(), best_model_file.name) 
        else:
            patience += 1
            if patience >= patienceLimit:
                print('Breaking on patience, epoch {}.'.format(i))
                break


        ########### Potentially report results this iter.

        if visThisIter:
            report_validation_sets(net_seg=net_seg, datadict=datadict, args=args)
            


    # After all the training, now reload the best model.
    print('Reloading best model, from epoch {}'.format(best_epoch))
    net_seg.load_state_dict(torch.load(best_model_file.name))
    
    print('Results with (best) epoch {}:'.format(best_epoch))    
    report_validation_sets(net_seg=net_seg, datadict=datadict, args=args)
    
    return net_seg


def write_test_results(net_seg, 
                       datadict,
                       args,
                       keys = ['images', 'labels'],
                       infokey = 'pat_view_phase',
                       do_cleaning = False):
    '''
    write_test_results: this function:
    - writes the pth model state for potential later use.
    - updates the results h5 with the ED/ES segs for all test patients.
    - reports the dices.
    '''

    global CAMUS_CONFIG
    pathsconfig = CAMUS_CONFIG['paths']
    
    
    print('write_test_results: view {}, fold {}.'.format(args.view, args.fold))
    
    ###########
    # Write the torch file
    #################
    
    # First, save the model file
    # Thanks: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    if args.training_augment:
        model_fname = 'aug_{}_fold_{}_win_{}_{}_rot_{}_noise_{}_{}.pth'.format(\
                        args.view, args.fold, \
                        args.windowing_scale[0], args.windowing_scale[1],\
                        args.rotation_scale,\
                        args.noise_scale[0], args.noise_scale[1])
    else:
        model_fname = '{}_fold_{}.pth'.format(args.view, args.fold)

    print('write_test_results: writing model {}.'.format(model_fname))
    modelpathname = os.path.join(pathsconfig['CAMUS_RESULTS_DIR'], 
                                 'saved_models',
                                 model_fname)
    torch.save(net_seg.state_dict(), modelpathname)
    
    
    
    
    ###########
    # Compute test results and report.
    #################

    test_outputs, info, dices = run_validation_returnAll(net_seg, 
                                                         datadict['test'], keys,
                                                         infokey, do_dice = True,
                                                         do_cleaning=do_cleaning)
    
    # Report, for completeness.
    print('write_test_results: Dice results on test set:')
    for key in sorted(list(dices.keys())):
        print('{:15s}: {:.3f}({:.3f})'.\
              format(key, dices[key].mean(), dices[key].std()))
    
    
    
    
    ###########
    # record the hdf5 results.
    #################
    
    # Need the name of the hdf5 
    if args.training_augment:
        dest_hd5 = 'aug_{}_win_{}_{}_rot_{}_noise_{}_{}.hdf5'.format(\
                        args.view, \
                        args.windowing_scale[0], args.windowing_scale[1],\
                        args.rotation_scale,\
                        args.noise_scale[0], args.noise_scale[1])
    else:
        dest_hd5 = '{}.hdf5'.format(args.view)
        
    dest_hd5 = os.path.join(pathsconfig['CAMUS_RESULTS_DIR'], dest_hd5)
        
    print('write_test_results: writing hdf5 {}.'.format(dest_hd5))
    
    # test_outputs should be batch x h x w if cleaned, otherwise 
    # batch x 4 x h x w if not cleaned. info is the corresponding 
    # information for the test_outputs. There are ED and ES segs 
    # for every patient listed in the test set. Let's collect them.
    info_df = pd.DataFrame(info, columns=['pat', 'view', 'phase'])
    
    # Pre-process test_outputs if it still has the channels dimension.
    if len(test_outputs.shape) == 4:
        temp_softmax = softmax(test_outputs, axis=1)
        test_outputs = np.argmax(temp_softmax, axis=1) 
        # test_outputs should now be batch x h x w
    
    
    with h5py.File(dest_hd5, 'a') as dfile:
        pats = info_df.pat.unique()
        for pat in pats:
            # Get this pat info
            inds = info_df.index[info_df.pat == pat]
            thispat = info_df.iloc[inds]
            # Sort to ED/ES
            thispat = thispat.sort_values(by='phase', ascending=True)
            
            segs = test_outputs[thispat.index.tolist()]
            # segs should be 2 x 256 x 256 (or whatever network image_size)
            
            assert segs.shape[0] == 2, 'write_test_results: pat {} needs both '\
                                        'ED/ES phases, seg shape {}.'.format(pat, segs.shape)
            # Can't do this if we want to partially rerun.
            # assert pat not in dfile, 'write_test_results: found redundant pat {}.'.format(pat)
            
            # Reassigning a dataset: https://stackoverflow.com/questions/22922584/how-to-overwrite-array-inside-h5-file-using-h5py
            if pat not in dfile:
                dfile.create_dataset(pat, data=segs, dtype='|u1')
            else:
                data = dfile[pat]
                data[...] = segs.astype(np.uint8)
    
    print('write_test_results: done.')
    


if __name__ == '__main__':
    
    print('{} execution begins.'.format(os.path.basename(__file__)))
    
    parser, args = parse()
    # For testing:
#     parser, args = parse('--view 2CH --fold 1 --training_augment False')
        
    # Thank you: https://stackoverflow.com/questions/27181084/how-to-iterate-over-arguments
    print('Arguments (sent, or defaults):')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))# , eval('args.{}'.format(arg)) also works.
   

    # Get the data
    datadict = build_DataLoaders(args)
    
    
    # Build the network and place on the gpu.
    print('Constructing UNetLike:')
    net_seg = UNetLike(**CAMUS_CONFIG['unet'])
    net_seg = torch.nn.DataParallel(net_seg)
    net_seg.cuda();
    print('\tTrainable params: {}'.format(sum(p.numel() for p in net_seg.parameters() if p.requires_grad)))
    
    
    # Run the experimental loop
    net_seg = run_actual_experimental_loop(net_seg, datadict, args)
    
    
    # Get all the results for writing out.
    write_test_results(net_seg, 
                       datadict,
                       args,
                       keys = ['images', 'labels'],
                       infokey = 'pat_view_phase',
                       do_cleaning = True)
    
    print('{} execution completed.\n\n'.format(os.path.basename(__file__)))