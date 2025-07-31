import numpy as np
import pandas as pd
import sys
import pickle
import torch


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
import random
from ext_models import train_VCNet_synth, predict_VCNet
import metrics

from utils import set_seed, disable_print, enable_print

set_seed(0)
disable_print()


# Aux func --------
def t_x_y(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t - 0.5) * 3.14159 * 2.0) * (
        t**2 + (4.0 * torch.maximum(x1, x6) ** 3) / (1.0 + 2.0 * x3**2) * torch.sin(x4)
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
    # for i in range(2):
    print("iteration: " + str(i))
    DATA_PATH = VCNET_PATH + f"dataset/simu1/eval/{i}"

    # Load data ------------------------------------------------------------
    data = pd.read_csv(DATA_PATH + "/train.txt", header=None, sep=" ")
    train_matrix = data.to_numpy()
    data = pd.read_csv(DATA_PATH + "/test.txt", header=None, sep=" ")
    test_matrix = data.to_numpy()
    data = pd.read_csv(DATA_PATH + "/t_grid.txt", header=None, sep=" ")
    t_grid = data.to_numpy()

    X = train_matrix[:, 1:-1]
    T = train_matrix[:, 0]
    Y = train_matrix[:, -1]

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
    # tunner = hyperparamTunner(
    #     dfnn,
    #     dfXcol + ["T"],
    #     ["Y"],
    #     predictiveNetTrainer,
    #     max_hyperopt_evals=max_hyperopt_evals,
    #     n_startup_jobs=n_startup_jobs,
    # )
    best_params = {
        "batch_size": 16,
        "dropout": 0.4,
        "hidden_layers": 5,
        "learning_rate": 0.0011,
        "neurons_per_layer": 400,
    }
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print("SL NN params")
    # print(best_params)

    XT = pd.DataFrame(X)
    XT["T"] = T

    baseNN = predictiveNetTrainer(XT.to_numpy(), Y, params=best_params)
    baseNN.train()

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
        max_hyperopt_evals=50,
        n_startup_jobs=30,
    )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params)
    best_params = {
        "use_propensity": True,
        "batch_size": 8,
        "dropout": 0.5,
        "hidden_layers": 5,
        "learning_rate": 0.0051,
        "neurons_per_layer": 100,
        "output_dim": 24,
        "spectral_norm": True,
        "epochs": 800,
        "early_stopping": 50,
        "alpha": 0.5,
        "neg_gaussian": True,
    }

    with open(f"best_params_hnet.pickle", "wb") as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"best_params_hnet.pickle", "rb") as handle:
        best_params = pickle.load(handle)

    HyperNN = HNETTrainer(X, T, Y, params=best_params)
    # self = model
    HyperNN.train()
    d = {
        "train_loss": HyperNN.train_losses,
        "val_loss": HyperNN.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    # VCNet -----------
    VCNet_model, _ = train_VCNet_synth(train_matrix)

    # Test performance  ---------------------------------------------------
    X_test = test_matrix[:, 1:-1]
    T_test = test_matrix[:, 0]

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
print(df3.apply(sem))
# print(df2.apply(sem))
