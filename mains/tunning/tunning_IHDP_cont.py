import numpy as np
import pandas as pd
import sys
import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

sys.path.append("..")
from functools import partial

from hnet import HNETTrainer
from hyperopt import fmin, tpe, hp, space_eval


VCNET_PATH = "/varying-coefficient-net-with-functional-tr-main/"
sys.path.append(VCNET_PATH)

from hnet import HNETTrainer
import metrics

DATA_PATH = VCNET_PATH + "dataset/ihdp"
DATA_PATH_SPLIT = DATA_PATH + "/tune/0"


# Aux func --------
def t_x_y(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[4]
    x5 = x[5]

    # v1
    factor1 = 0.5
    factor2 = 1.5

    # v2
    factor1 = 1.5
    factor2 = 0.5

    y = (
        1.0
        / (1.2 - t)
        * torch.sin(t * 3.0 * 3.14159)
        * (
            factor1 * torch.tanh((torch.sum(x[cate_idx1]) / 10.0 - cate_mean1) * alpha)
            + factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4))
        )
    )
    return y


# LOAD DATA ------------------------------------------------------------

# Load raw data, I'll be used later for computing new Ys
ihdp = pd.read_csv(DATA_PATH + "/ihdp.csv")
ihdp = ihdp.to_numpy()
ihdp = ihdp[
    :, 2:27
]  # delete the first column (data idx)/ delete the second coloum (treatment)
ihdp = torch.from_numpy(ihdp)
ihdp = ihdp.float()

n_feature = ihdp.shape[1]
n_data = ihdp.shape[0]


# normalize the data
for _ in range(n_feature):
    minval = min(ihdp[:, _]) * 1.0
    maxval = max(ihdp[:, _]) * 1.0
    ihdp[:, _] = (1.0 * (ihdp[:, _] - minval)) / maxval

# cate_idx = torch.tensor([3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
cate_idx2 = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

alpha = 5.0
cate_mean1 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
cate_mean2 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
# END LOAD RAW DATA

data_matrix = torch.load(DATA_PATH + "/data_matrix.pt")
t_grid_all = torch.load(DATA_PATH + "/t_grid.pt")

#  ---------------------------------------------------------------

search_space = {
    "hidden_layers": hp.choice("hidden_layers", [3, 4, 5]),
    "neurons_per_layer": hp.choice("neurons_per_layer", [20, 35, 50, 65, 80]),
    "learning_rate": hp.choice("learning_rate", [0.05, 0.005, 0.001, 0.0005, 0.0001]),
    "output_dim": hp.choice("output_dim", list(range(12, 100, 2))),
    "hidden_net_layers": hp.choice("hidden_net_layers", list(range(4))),
    "hidden_net_activation": hp.choice(
        "hidden_net_activation", ["elu", "relu", "linear"]
    ),
    "spectral_norm": hp.choice("spectral_norm", [True, False]),
    "use_propensity": hp.choice("use_propensity", [True]),
    "alpha": hp.choice("alpha", [0, 0.1, 0.25, 0.5, 1]),
    "neg_gaussian": hp.choice("neg_gaussian", [True, False]),
}


def train_model_nn(params=None):
    # print(params)
    params["batch_size"] = 471
    params["epochs"] = 800
    params["val_split"] = 0

    HyperNN = HNETTrainer(X, T, Y, params=params)
    HyperNN.train()

    return HyperNN


def pred_HyperNN(model, X_test, T_feature, coefs_test, *args):
    return model.eval_hypernet_full(coefs_test, X_test[T_feature].to_numpy())


def hyperopt_objective(space):
    params = space
    # print(params)
    mises = []

    random_numbers = random.sample(range(20), 5)
    for i in random_numbers:
        it_path = DATA_PATH_SPLIT[:-1] + str(i)
        idx_train = torch.load(it_path + "/idx_train.pt")
        idx_test = torch.load(it_path + "/idx_test.pt")

        train_matrix = data_matrix[idx_train, :]

        test_matrix = data_matrix[idx_test, :]
        t_grid = t_grid_all[:, idx_test]

        global X, T, Y
        X = np.array(train_matrix[:, 1:-1])
        T = np.array(train_matrix[:, 0])
        Y = np.array(train_matrix[:, -1])

        # Test thinkgs
        X_test = np.array(test_matrix[:, 1:-1])
        T_test = np.array(test_matrix[:, 0])

        global T_feature, num_feature
        num_feature = X_test.shape[1]
        T_feature = "T"

        global num_integration_samples, step_size, treatment_strengths
        samples_power_of_two = 6
        num_integration_samples = 2**samples_power_of_two + 1
        # step_size = 1.0 / num_integration_samples
        treatment_strengths = np.linspace(min(T), max(T), num_integration_samples)

        HyperNN = train_model_nn(params)

        models = {"HyperNN": (HyperNN, pred_HyperNN)}
        average_mises = metrics.compute_MISE(
            X_test,
            models,
            t_x_y,
            treatment_strengths,
            num_integration_samples,
            T_feature,
        )

        mises.append(average_mises["HyperNN"])

    hyperopt_loss = np.mean(mises)

    return hyperopt_loss


def run_hyperopt_optimization(
    nnet_search_space, n_startup_jobs=10, max_hyperopt_evals=20
):
    algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs, verbose=False)

    return fmin(
        hyperopt_objective,
        nnet_search_space,
        algo=algo,
        max_evals=max_hyperopt_evals,
    )


best_params1 = run_hyperopt_optimization(search_space, 20, 40)
best_params1 = space_eval(search_space, best_params1)
print("Best_params 1")
print(best_params1)
best_params2 = run_hyperopt_optimization(search_space, 20, 40)
best_params2 = space_eval(search_space, best_params2)
print("Best_params 2")
print(best_params2)
best_params3 = run_hyperopt_optimization(search_space, 200, 400)
best_params3 = space_eval(search_space, best_params3)
print("Best_params 3")
print(best_params3)
best_params4 = run_hyperopt_optimization(search_space, 200, 400)
best_params4 = space_eval(search_space, best_params4)

print("Best_params 1")
print(best_params1)
print("Best_params 2")
print(best_params2)
print("Best_params 3")
print(best_params3)
print("Best_params 4")
print(best_params4)
