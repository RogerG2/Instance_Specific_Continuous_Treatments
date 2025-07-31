import numpy as np
import pandas as pd
import sys
import pickle
import torch
import os

VCNET_PATH = "/varying-coefficient-net-with-functional-tr-main/"
sys.path.append("..")
sys.path.append(VCNET_PATH)

import sklearn.metrics as m
from sklearn.ensemble import RandomForestRegressor
from scipy.integrate import romb
import matplotlib.pyplot as plt

from hnet import HNETTrainer
from predictive_net import predictiveNetTrainer
from nnet_hyperparam import hyperparamTunner
from utils import set_seed, disable_print, enable_print
from ext_models import train_VCNet_IHDP, predict_VCNet
import metrics

set_seed(0)
disable_print()


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


def pred_HyperNN(model, X_test, T_feature, coefs_test, *args):
    return model.eval_hypernet_full(coefs_test, X_test[T_feature].to_numpy())


def pred_VCNet(model, X_test_cf, T_feature, coefs_test, ts_array, X_test, *args):
    return predict_VCNet(
        model, torch.from_numpy(X_test).float(), torch.from_numpy(ts_array).float()
    )


list_mises = []
list_amses = []
list_amses_mean = []

for i in range(100):

    print("iteration: " + str(i))
    DATA_PATH = VCNET_PATH + "dataset/ihdp"
    DATA_PATH_SPLIT = DATA_PATH + f"/eval/{i}"

    # Load data ------------------------------------------------------------

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

    # 0 1 2 4 5 -> continuous

    # normalize the data
    for _ in range(n_feature):
        minval = min(ihdp[:, _]) * 1.0
        maxval = max(ihdp[:, _]) * 1.0
        ihdp[:, _] = (1.0 * (ihdp[:, _] - minval)) / maxval

    cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    cate_idx2 = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    alpha = 5.0
    cate_mean1 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    cate_mean2 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    # END LOAD RAW DATA

    data_matrix = torch.load(DATA_PATH + "/data_matrix.pt")
    t_grid_all = torch.load(DATA_PATH + "/t_grid.pt")

    idx_train = torch.load(DATA_PATH_SPLIT + "/idx_train.pt")
    idx_test = torch.load(DATA_PATH_SPLIT + "/idx_test.pt")

    train_matrix = data_matrix[idx_train, :]

    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    X = np.array(train_matrix[:, 1:-1])
    T = np.array(train_matrix[:, 0])
    Y = np.array(train_matrix[:, -1])

    # Train mdoels  ---------------------------------------------------------------
    # RF - SLEARN -----
    rf = RandomForestRegressor()
    XT = pd.DataFrame(X)
    XT["T"] = T
    rf.fit(XT.to_numpy(), Y)

    # NN - SLEARN -----
    dfXcol = ["c" + str(i) for i in range(X.shape[1])]
    dfnn = pd.DataFrame(X, columns=dfXcol)
    dfnn["T"] = T
    dfnn["Y"] = Y
    tunner = hyperparamTunner(
        dfnn,
        dfXcol + ["T"],
        ["Y"],
        predictiveNetTrainer,
        max_hyperopt_evals=30,
        n_startup_jobs=20,
    )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # # print("SL NN params")
    # # print(best_params)

    best_params = {
        "batch_size": 16,
        "dropout": 0.0,
        "hidden_layers": 5,
        "learning_rate": 0.0001,
        "neurons_per_layer": 200,
    }

    XT = pd.DataFrame(X)
    XT["T"] = T

    baseNN = predictiveNetTrainer(XT.to_numpy(), Y, params=best_params)
    baseNN.train()

    total_params = sum(p.numel() for p in baseNN.model.parameters() if p.requires_grad)
    # print("Total number of parameters: ", total_params)

    d = {
        "train_loss": baseNN.train_losses,
        "val_loss": baseNN.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    # Hypernetwork  -----
    dfXcol = ["c" + str(i) for i in range(X.shape[1])]
    dfnn = pd.DataFrame(X, columns=dfXcol)
    dfnn["Y"] = Y
    tunner = hyperparamTunner(
        dfnn,
        dfXcol,
        ["Y"],
        HNETTrainer,
        t=T,
        max_hyperopt_evals=40,
        n_startup_jobs=20,
    )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # # print(best_params)

    best_params = {
        "hidden_layers": 4,
        "hidden_net_activation": "relu",
        "hidden_net_layers": 0,
        "learning_rate": 0.05,
        "neurons_per_layer": 50,
        "output_dim": 72,
        "spectral_norm": True,
        "use_propensity": True,
        "use_propensity": False,
        "alpha": 0.5,
        "neg_gaussian": True,
    }

    best_params["batch_size"] = 471
    best_params["epochs"] = 800
    best_params["val_split"] = 0

    with open(f"best_params_hnet.pickle", "wb") as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"best_params_hnet.pickle", "rb") as handle:
        best_params = pickle.load(handle)

    HyperNN = HNETTrainer(X, T, Y, params=best_params)
    total_params = sum(p.numel() for p in HyperNN.model.parameters() if p.requires_grad)
    # print("Total number of parameters: ", total_params)
    # self = HyperNN
    HyperNN.train()
    d = {
        "train_loss": HyperNN.train_losses,
        "val_loss": HyperNN.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    # VCNet -----------
    VCNet_model, _, losses = train_VCNet_IHDP(np.array(train_matrix))
    pd.Series(losses).plot()
    total_params = sum(p.numel() for p in VCNet_model.parameters() if p.requires_grad)
    # print("Total number of parameters: ", total_params)

    # Test performance  ---------------------------------------------------
    X_test = np.array(test_matrix[:, 1:-1])
    T_test = np.array(test_matrix[:, 0])

    num_feature = X_test.shape[1]

    T_feature = "T"

    samples_power_of_two = 6
    num_integration_samples = 2**samples_power_of_two + 1
    step_size = 1.0 / num_integration_samples
    treatment_strengths = np.linspace(min(T), max(T), num_integration_samples)

    models = {
        "rf": (rf, lambda *args: args[0].predict(args[1].to_numpy())),
        "baseNN": (baseNN, lambda *args: args[0].predict(args[1].to_numpy())),
        "HyperNN": (HyperNN, pred_HyperNN),
        "VCNet": (VCNet_model, pred_VCNet),
    }
    average_mises = metrics.compute_MISE(
        X_test, models, t_x_y, treatment_strengths, num_integration_samples, T_feature
    )

    average_amses, average_amses_mean = metrics.compute_AMSE(
        X_test, models, t_x_y, T_test, T_feature
    )

    list_mises.append(average_mises)
    list_amses.append(average_amses)
    list_amses_mean.append(average_amses_mean)

df1 = pd.DataFrame(list_mises)
# df2 = pd.DataFrame(list_amses)
df3 = pd.DataFrame(list_amses_mean)
enable_print()
print("df means")
print(df1.mean())
# print(df2.mean())
print(df3.mean())

from scipy.stats import sem

print("df std")
print(df1.apply(sem))
# print(df2.apply(sem))
print(df3.apply(sem))
