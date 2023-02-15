# Diffeo-RADMts

Repository for our <b>KDD 2023</b> paper, [Diffeomorphic Aligned Robust Anomaly Detection for MUltivariate Time Series] co-authored by: Abilasha S, Sahely Bhadra.



## Author of this software
Abilasha S (email: 111814001@smail.iitpkd.ac.in)


## Operation system: 
For the native PyTorch implementation (slower), we support all operating systems. 
For the fast CUDA implementation of libcpab, we only support Linux.

## Installation
We recommend installing a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via Anaconda.
For instance:
```
conda create -n drad python=3.7 numpy matplotlib seaborn tqdm
```
### libcpab
licpab [2] is a python package supporting the CPAB transformations [1] in Numpy, Tensorflow and Pytorch.
For your convince, we have added a lightweight version of libcpab at DTAN/libcpab. 

That being said, you are still encouraged to install the full package. 

Install [libcpab](https://github.com/SkafteNicki/libcpab) <br>
Note 1: you might have to recompile the dynamic libraries under /libcpab/tensorflow/ <br>
```
git clone https://github.com/SkafteNicki/libcpab
```
Add libcpab to your python path:
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/libcpab
```
Make sure libcpab was installed properly (Run one of the demos).

