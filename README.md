# Echo Segmentation
Joshua Stough

Starting with [LeClerc et al., TMI 2019](https://www.creatis.insa-lyon.fr/~bernard/publis/tmi_2019_leclerc.pdf) (or [paid version](https://ieeexplore.ieee.org/abstract/document/8649738)), where the authors provide a [2- and 4-chamber echo dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) and test U-Net and other deep learning schemes against it. 

I basically want to implement that experiment here in [pytorch](https://pytorch.org/docs/stable/index.html). I'm also using for inspiration an [Nvidia DLI course on V-Net](https://courses.nvidia.com/courses/course-v1:DLI+L-HX-10+V1/info) (requires login) in pytorch.


## Description of notable elements

- [CAMUS_UNet3](notebooks/CAMUS_UNet3.ipynb): Learn to segment echo on one 90-10 split. Pretty deep U-Net like model, advanced data augmentation. 
- [CAMUS_Segment_Video](notebooks/CAMUS_Segment_Video.ipynb): Use the model saved in CAMUS_UNet3 to segment one of the example CAMUS videos. Visualizes this overlay and computes dice overlaps on the frames (ED, ES) for which the labels are known.
- [CAMUS_Validate_Volume_Calculation](notebooks/CAMUS_Validate_Volume_Calculation.ipynb): Validate that I can use the CAMUS manual segmentations to compute volumes consistent with the EHR reported values.

- [camus_prep.py](src/camus_prep.py): script to prepare 10-fold cross-validation experiment as in [LeClerc et al., TMI 2019](https://www.creatis.insa-lyon.fr/~bernard/publis/tmi_2019_leclerc.pdf), stratified on image quality and on EF. This is for the publicly relased training data.
- [camus_train_segment.py](src/camus_train_segment.py): script performs one train/validation and write test to results file, for one view/fold pair. This one hdf5 results file can then be read and processed by... 
- [CAMUS_Validate_Results](notebooks/CAMUS_Validate_Results.ipynb): Compute volumes and dices for all test folds. Basically populates the figures in the SPIE-MI submission (publicly-released).

- [CAMUS_Make_Test_Submission](notebooks/CAMUS_Make_Test_Submission.ipynb): After camus_prep and all the view/fold pairs from camus_train_segment have been generated, this notebook takes the learned models and applies them to the CAMUS testing test, whose images were released but not the segmentations. It writes all the mhd/raw files necessary to submit to the [CAMUS Challenge website](https://www.creatis.insa-lyon.fr/Challenge/camus/participation.html).