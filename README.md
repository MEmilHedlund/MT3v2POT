## Introduction
This repository contains the code for the paper "Can Deep Learning be Applied to Model-Based Multi-Object Tracking?" (https://arxiv.org/abs/2202.07909). The code for MT3v2 was developed as joint effort by Juliano Pinto, Georg Hess, and William Ljungbergh, and was partially based on the code available at the repositories for [DETR](https://github.com/facebookresearch/detr) and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).



## Setting up
In order to set up a conda environment with all the necessary dependencies, run the command:
  ```
   conda env create -f conda-env/environment-<gpu/cpu>.yml
  ```





## Running experiments

Run an experiment using the `training.py` script. Example usage:

```
src/training.py -tp configs/tasks/task1.yaml -mp configs/models/mt3v2.yaml
```
-tp configs/tasks/task1.yaml -mp configs/models/mt3v2.yaml --continue_training_from Task4

Training hyperparameters such as batch size, learning rate, checkpoint interval, etc, are found in the file `configs/models/mt3v2.yaml`. 



## Evaluating experiments

You can plot metrics of interest using the `util/plot_results.py` script, during and after training. 


## Trained models

Trained models for each of the 4 tasks described in the paper can be found [here](https://chalmersuniversity.box.com/s/kuf5wpjy8ufemy8ynpak7rrfh9p2exft).

The ablations trained for task 4 are available [here](will-upload-soon).
