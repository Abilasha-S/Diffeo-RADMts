# Diffeo-RADMts

Repository for our <b>KDD 2023</b> paper, [Diffeomorphic Aligned Robust Anomaly Detection for MUltivariate Time Series] co-authored by: Abilasha S, Sahely Bhadra.


## Author of this software
Abilasha S (email: 111814001@smail.iitpkd.ac.in)


## Installation
We recommend installing a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via Anaconda.
For instance:
```
conda create -n drad
```
Or you can create docker container by following instructions given [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and work on it

## Get Started

1. Install required packages by running
pip install -r requirements.txt

2. Download data. SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/). Other datasets are given in data folder on each modules separately. All datasets are preprocessed and this can be done by executing.
```
python .


3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
