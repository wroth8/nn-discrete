# Discrete-valued neural networks using weight distributions

This repository contains python code using Theano to reproduce the experiments from our paper

```
@INPROCEEDINGS{Roth2019,
    AUTHOR="Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf",
    TITLE="Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions",
    BOOKTITLE="European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)",
    YEAR=2019
}
```

## Usage

1. Clone this repository: `git clone https://github.com/wroth8/nn-discrete.git`
2. Create a virtual environment from the included environment.yml and activate it. Please note that we observed that the code does not run with the newer numpy version 1.16.
    1. Create using conda: `conda env create -f environment.yml`
    2. Activate using conda: `conda activate nn-discrete-ecml19`
3. Set the python path using `export PYTHONPATH="/path/to/nn-discrete"` and change directory using `cd /path/to/nn-discrete`
4. Run the experiments (replace `<dataset>` with one of `mnist/mnist_pi/cifar10/cifar100/svhn`). Please note that you have to set the `$THEANO_FLAGS` environment variable according to your system.
    1. To train a model with real weights and tanh activation function run `python experiments/<dataset>/experiment_<dataset>_real.py`. The resulting model will be used as initial model for the discrete-valued models.
    2. To train a model with ternary weights and sign activation function run `python experiments/<dataset>/experiment_<dataset>_ternary_sign.py`. Requires that i. has finished first.
    3. To train a model with ternary weights and tanh activation function run `python experiments/<dataset>/experiment_<dataset>_ternary_tanh.py`. Requires that i. has finished first.
    4. To train a model with ternary weights and sign activation function initialized with the model using ternary weights and tanh activation run `python experiments/<dataset>/experiment_<dataset>_ternary_sign_from_tanh.py`. Requires that iii. has finished first.
