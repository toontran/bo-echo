'''
Joshua Stough, 6/19

CAMUS echo segmentation cross-validation experimental setup.
'''

from utils.camus_config import CAMUS_CONFIG
from utils.camus_load_info import (make_camus_echo_dataset,
                                   split_camus_echo,
                                   make_camus_EHR,
                                   camus_generate_folds)
import sys
import os
import pickle
import pandas as pd


if __name__ == '__main__':
    config_dict = CAMUS_CONFIG['paths']
    
    # Make the output folder. 
    if not os.path.exists(config_dict['CAMUS_RESULTS_DIR']):
        print('camus_prep: {} folder missing, making it.'.format(config_dict['CAMUS_RESULTS_DIR']))
        os.mkdir(config_dict['CAMUS_RESULTS_DIR'])
        
    
    foldfilename = os.path.join(config_dict['CAMUS_RESULTS_DIR'], config_dict['folds_file'])
    if not os.path.exists(foldfilename):
        print('camus_prep: folds file {} missing, generating...'.format(foldfilename))
        
        # Generate the stratified folds like LeClerc.
        kf, ehr = camus_generate_folds(10, random_state=None)
        
        with open(foldfilename, 'wb') as fid:
           pickle.dump(kf, fid)    
        
        ehrfilename = os.path.join(config_dict['CAMUS_RESULTS_DIR'], config_dict['ehr_file'])
        # with open(ehrfilename, 'wb') as fid:
        #     pickle.dump(ehr, fid)
        # We're going to read with pandas, let's write too.
        ehr.to_pickle(ehrfilename)
    
    print('camus_prep: done.')    
        