import pandas as pd
import numpy as np
import sys
import random
import torch

from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt


sys.path.append("..")


from utils import set_seed, disable_print, enable_print

disable_print()
set_seed(0)

import datasets_bin
from hnet import HNETTrainer

print(torch.set_default_dtype(torch.float32))

IHDP_DATA_PATH = "../../../data/"


# AUX FUNC -----------------------------
def compute_ATE(real_ITE, pred_ITE):
    return np.abs((np.mean(real_ITE) - np.mean(pred_ITE)))


def compute_PEHE(real_ITE, pred_ITE):
    return np.sqrt(np.mean((real_ITE - pred_ITE) ** 2))


def slearn_pred_ite(XT, m, T="T"):

    df = XT.copy()
    df[T] = 0
    preds0 = m.predict(df.to_numpy())

    df[T] = 1
    preds1 = m.predict(df.to_numpy())

    return preds1 - preds0


# Data  ------------------------------------------------------------------------
dataset_name = "ihdp"
metrics_dict = dict()
metrics_dict["ATE"] = compute_ATE
metrics_dict["PEHE"] = compute_PEHE

PEHES_RF = []
ATES_RF = []
PEHES_HNET = []
ATES_HNET = []

it_range = range(1000)
for i in it_range:

    print(i)
    # n ∈ {500, 1000}, d ∈ {6, 12} and σ ∈ {0.5, 1, 2, 3},
    data = datasets_bin.dataLoader(
        data_path=IHDP_DATA_PATH,
        test_perc=0.2,
        # realcause_df=df,
        cv_splits=5,
        ihdp_i=i,
    )
    data.load_dataset(dataset_name)

    # Modelling --------------------------------------------------------------------
    # RF
    rf = RandomForestRegressor()
    XT = data.train_df[data.X_features]
    rf.fit(XT.to_numpy(), data.train_df[data.Y_feature[0]].to_numpy())

    # NNET
    best_params = {
        "add_propensity": False,
        "batch_size": 128,
        "dropout": 0.0,
        "hidden_layers": 3,
        "learning_rate": 0.0051,
        "neurons_per_layer": 200,
        "output_dim": 48,
        "spectral_norm": True,
    }
    # from nnet_hyperparam import hyperparamTunner

    # tunner = hyperparamTunner(
    #     data.train_df,
    #     data.train_df.drop(["T", "Y"], axis=1).columns.to_list(),
    #     ["Y"],
    #     HNETTrainer,
    #     t=data.train_df["T"].to_numpy(),
    #     max_hyperopt_evals=20,
    #     n_startup_jobs=15,
    # )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params)

    Hnet = HNETTrainer(
        data.train_df[data.X_features].drop(["T"], axis=1).to_numpy(),
        data.train_df[data.X_features]["T"].to_numpy(),
        data.train_df[data.Y_feature].to_numpy(),
        params=best_params,
    )
    Hnet.train()
    d = {
        "train_loss": Hnet.train_losses,
        "val_loss": Hnet.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    # CI ----------------
    # Real ITE from the test values
    real_ITE = data.test_ite

    # Random Forest
    pred_ITE = slearn_pred_ite(data.test_df[data.X_features], rf)

    metric_values = dict((k, []) for k in metrics_dict.keys())
    for k, v in metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE), 4)

    PEHES_RF.append(metric_values["PEHE"])
    ATES_RF.append(metric_values["ATE"])

    # HNET
    coefs_test = Hnet.predict(
        data.test_df[data.X_features].drop("T", axis=1).to_numpy()
    )
    pred0 = Hnet.eval_hypernet_full(coefs_test, np.zeros(data.test_df.shape[0]))
    pred1 = Hnet.eval_hypernet_full(coefs_test, np.ones(data.test_df.shape[0]))
    pred_ITE = np.array(pred1) - np.array(pred0)

    metric_values = dict((k, []) for k in metrics_dict.keys())
    for k, v in metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE), 4)

    PEHES_HNET.append(metric_values["PEHE"])
    ATES_HNET.append(metric_values["ATE"])

    if i % 50 == 0:
        results_df = pd.DataFrame(
            {
                "PEHES_RF": PEHES_RF,
                "ATES_RF": ATES_RF,
                "PEHES_dragonnet": PEHES_HNET,
                "ATES_dragonnet": ATES_HNET,
            }
        )
        results_df.to_csv("main_dragonnet_loop_results.csv", index=False)

results_df = pd.DataFrame(
    {
        "PEHES_RF": PEHES_RF,
        "ATES_RF": ATES_RF,
        "PEHES_dragonnet": PEHES_HNET,
        "ATES_dragonnet": ATES_HNET,
    }
)
results_df.to_csv("main_loop_results.csv", index=False)

enable_print()
print(f"RF PEHE: {np.mean(PEHES_RF)}   ------  ATE: {np.mean(ATES_RF)}")
print(f"HNET PEHE: {np.mean(PEHES_HNET)}   --   ATE: {np.mean(ATES_HNET)}")

print("sd")
import scipy

print(scipy.stats.sem(ATES_HNET))

print(f"RF PEHE: {scipy.stats.sem(PEHES_RF)}   ------  ATE: {scipy.stats.sem(ATES_RF)}")
print(
    f"HNET PEHE: {scipy.stats.sem(PEHES_HNET)}   --   ATE: {scipy.stats.sem(ATES_HNET)}"
)
