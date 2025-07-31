import numpy as np
import pandas as pd
import sys
import time

sys.path.append("..")
sys.path.append("../SCIGAN_main")

import sklearn.metrics as m
from sklearn.ensemble import RandomForestRegressor
from scipy.integrate import romb
import matplotlib.pyplot as plt

from hnet import HNETTrainer
from predictive_net import predictiveNetTrainer
from nnet_hyperparam import hyperparamTunner

from SCIGAN_main.data_simulation import get_dataset_splits, TCGA_Data
from SCIGAN_main.utils.evaluation_utils import get_patient_outcome


from utils import set_seed, disable_print, enable_print

disable_print()
set_seed(0)

DATA_PATH = "../data/tcga/"

n_hyper_tunning = 250
n_random = 200
start = time.time()

all_iterations = []
n_iterations = 10
for _ in range(n_iterations):
    # Data ----------------------------------------------------------------------
    dataset_params = dict()
    dataset_params["num_treatments"] = 3
    dataset_params["treatment_selection_bias"] = 2
    dataset_params["dosage_selection_bias"] = 2
    dataset_params["save_dataset"] = False
    dataset_params["validation_fraction"] = 0.1
    dataset_params["test_fraction"] = 0.2
    dataset_params["data_path"] = DATA_PATH + "tcga.p"

    data_class = TCGA_Data(dataset_params)
    dataset = data_class.dataset
    dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)

    dataset_train.keys()
    df_train = pd.DataFrame(dataset_train["x"])  #  .iloc[:500]
    df_train.columns = [f"X_{i+1}" for i in range(df_train.shape[1])]
    df_train["t"] = dataset_train["t"]  # [:500]
    df_train["d"] = dataset_train["d"]  #  [:500]
    df_train["y"] = dataset_train["y_normalized"]  # [:500]

    df_train_0 = df_train[df_train["t"] == 0].copy().drop("t", axis=1)
    df_train_1 = df_train[df_train["t"] == 1].copy().drop("t", axis=1)
    df_train_2 = df_train[df_train["t"] == 2].copy().drop("t", axis=1)

    df_test = pd.DataFrame(dataset_test["x"])
    df_test["t"] = dataset_test["t"]
    df_test["d"] = dataset_test["d"]
    df_test["y"] = dataset_test["y_normalized"]

    df_test_0 = df_test[df_test["t"] == 0].copy().drop("t", axis=1)
    df_test_1 = df_test[df_test["t"] == 1].copy().drop("t", axis=1)
    df_test_2 = df_test[df_test["t"] == 2].copy().drop("t", axis=1)
    dfs_test = [df_test_0, df_test_1, df_test_2]

    # Model ----------------------------------------------------------------------
    # RF - SLEARN -----------
    rf_0 = RandomForestRegressor()
    rf_1 = RandomForestRegressor()
    rf_2 = RandomForestRegressor()
    rf_0.fit(df_train_0.drop("y", axis=1).to_numpy(), df_train_0["y"].to_numpy())
    rf_1.fit(df_train_1.drop("y", axis=1).to_numpy(), df_train_1["y"].to_numpy())
    rf_2.fit(df_train_2.drop("y", axis=1).to_numpy(), df_train_2["y"].to_numpy())

    # NNET - SLEARN -----------
    # tunner = hyperparamTunner(
    #     df_train_0,
    #     df_train_0.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     predictiveNetTrainer,
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params)

    # tunner = hyperparamTunner(
    #     df_train_1,
    #     df_train_1.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     predictiveNetTrainer,
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params)

    # tunner = hyperparamTunner(
    #     df_train_2,
    #     df_train_2.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     predictiveNetTrainer,
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params)

    # best_params = {
    #     "batch_size": 16,
    #     "dropout": 0.0,
    #     "hidden_layers": 5,
    #     "learning_rate": 0.0001,
    #     "neurons_per_layer": 200,
    # }

    best_params_0 = {
        "batch_size": 32,
        "dropout": 0.0,
        "hidden_layers": 2,
        "learning_rate": 0.0071,
        "neurons_per_layer": 450,
    }
    best_params_1 = {
        "batch_size": 32,
        "dropout": 0.25,
        "hidden_layers": 4,
        "learning_rate": 0.0081,
        "neurons_per_layer": 50,
    }
    best_params_2 = {
        "batch_size": 32,
        "dropout": 0.3,
        "hidden_layers": 5,
        "learning_rate": 0.0026,
        "neurons_per_layer": 200,
    }

    baseNN_0 = predictiveNetTrainer(
        df_train_0.drop("y", axis=1).to_numpy(),
        df_train_0["y"].to_numpy(),
        params=best_params_0,
    )
    baseNN_0.train()
    baseNN_1 = predictiveNetTrainer(
        df_train_1.drop("y", axis=1).to_numpy(),
        df_train_1["y"].to_numpy(),
        params=best_params_1,
    )
    baseNN_1.train()
    baseNN_2 = predictiveNetTrainer(
        df_train_2.drop("y", axis=1).to_numpy(),
        df_train_2["y"].to_numpy(),
        params=best_params_2,
    )
    baseNN_2.train()

    # # Hypernetwork -----------
    # # # Tune
    # n_hyper_tunning = 50
    # n_random = 25
    # tunner = hyperparamTunner(
    #     df_train_0,
    #     df_train_0.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     HNETTrainer,
    #     t=df_train_0["d"].to_numpy(),
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params_0 = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params_0)

    # n_hyper_tunning = 100
    # n_random = 50
    # tunner = hyperparamTunner(
    #     df_train_0,
    #     df_train_0.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     HNETTrainer,
    #     t=df_train_0["d"].to_numpy(),
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params_0 = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params_0)

    # n_hyper_tunning = 200
    # n_random = 100
    # tunner = hyperparamTunner(
    #     df_train_0,
    #     df_train_0.drop(["y", "d"], axis=1).columns.to_list(),
    #     ["y"],
    #     HNETTrainer,
    #     t=df_train_0["d"].to_numpy(),
    #     max_hyperopt_evals=n_hyper_tunning,
    #     n_startup_jobs=n_random,
    # )
    # best_params_0 = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params_0)

    # best_params_0 = {
    #     "batch_size": 16,
    #     "dropout": 0.0,
    #     "hidden_layers": 4,
    #     "learning_rate": 0.005,
    #     "neurons_per_layer": 100,
    #     "output_dim": 34,
    #     "spectral_norm": False,
    #     "epochs": 150,
    # }

    best_params_0 = {
        "batch_size": 16,
        "dropout": 0.1,
        "hidden_layers": 5,
        "learning_rate": 0.0096,
        "neurons_per_layer": 50,
        "output_dim": 38,
        "spectral_norm": False,
        "epochs": 100,
    }

    tunner = hyperparamTunner(
        df_train_1,
        df_train_1.drop(["y", "d"], axis=1).columns.to_list(),
        ["y"],
        HNETTrainer,
        t=df_train_1["d"].to_numpy(),
        max_hyperopt_evals=n_hyper_tunning,
        n_startup_jobs=n_random,
    )
    # best_params_1 = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params_1)
    best_params_1 = {
        "batch_size": 16,
        "dropout": 0.0,
        "hidden_layers": 5,
        "learning_rate": 0.0071,
        "neurons_per_layer": 200,
        "output_dim": 34,
        "spectral_norm": False,
    }

    tunner = hyperparamTunner(
        df_train_2,
        df_train_2.drop(["y", "d"], axis=1).columns.to_list(),
        ["y"],
        HNETTrainer,
        t=df_train_2["d"].to_numpy(),
        max_hyperopt_evals=n_hyper_tunning,
        n_startup_jobs=n_random,
    )
    # best_params_2 = tunner.train_and_evaluate_hyperopt_nnet()
    # print(best_params_2)
    best_params_2 = {
        "batch_size": 32,
        "dropout": 0.35,
        "hidden_layers": 3,
        "learning_rate": 0.0061,
        "neurons_per_layer": 300,
        "output_dim": 46,
        "spectral_norm": True,
    }

    hypernet_0 = HNETTrainer(
        df_train_0.drop(["y", "d"], axis=1).to_numpy(),
        df_train_0["d"].to_numpy(),
        df_train_0["y"].to_numpy(),
        params=best_params_0,
    )
    hypernet_0.train()
    d = {
        "train_loss": hypernet_0.train_losses,
        "val_loss": hypernet_0.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    hypernet_1 = HNETTrainer(
        df_train_1.drop(["y", "d"], axis=1).to_numpy(),
        df_train_1["d"].to_numpy(),
        df_train_1["y"].to_numpy(),
        params=best_params_1,
    )
    hypernet_1.train()
    d = {
        "train_loss": hypernet_1.train_losses,
        "val_loss": hypernet_1.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    hypernet_2 = HNETTrainer(
        df_train_2.drop(["y", "d"], axis=1).to_numpy(),
        df_train_2["d"].to_numpy(),
        df_train_2["y"].to_numpy(),
        params=best_params_2,
    )
    hypernet_2.train()
    d = {
        "train_loss": hypernet_2.train_losses,
        "val_loss": hypernet_2.val_losses,
    }
    # pd.DataFrame({k: pd.Series(v) for k, v in d.items()}).plot()
    # plt.show()

    print("End of training", time.time() - start)
    # Eval ----------------------------------------------------------------------
    print("Start eval")

    def get_true_and_pred_y(
        model, df_test, patient, hypernet, dataset, treatment_idx, step_size
    ):

        if hypernet:
            coefs_test = model.predict(df_test.drop("d", axis=1).to_numpy())
            pred_dose_response = model.eval_hypernet_full(
                coefs_test, df_test["d"].to_numpy()
            )
            pred_dose_response = np.array(pred_dose_response)
        else:
            pred_dose_response = model.predict(df_test.to_numpy())

        pred_dose_response = (
            pred_dose_response
            * (dataset["metadata"]["y_max"] - dataset["metadata"]["y_min"])
            + dataset["metadata"]["y_min"]
        )

        true_outcomes = [
            get_patient_outcome(patient, dataset["metadata"]["v"], treatment_idx, d)
            for d in df_test["d"]
        ]

        return true_outcomes, pred_dose_response

    def eval_model(model, test_patients, treatment_idx=0, hypernet=False):

        samples_power_of_two = 6
        num_integration_samples = 2**samples_power_of_two + 1
        step_size = 1.0 / num_integration_samples
        treatment_strengths = np.linspace(
            np.finfo(float).eps, 1, num_integration_samples
        )

        # MISE
        mises = []
        for patient in test_patients:

            test_data = dict()
            test_data["x"] = np.repeat(
                np.expand_dims(patient, axis=0), len(treatment_strengths), axis=0
            )
            df_test = pd.DataFrame(test_data["x"])
            df_test["d"] = treatment_strengths

            true_outcomes, pred_dose_response = get_true_and_pred_y(
                model, df_test, patient, hypernet, dataset, treatment_idx, step_size
            )
            mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
            mises.append(mise)

        treatment_strengths_amse = dfs_test[treatment_idx]["d"]
        amses = []
        for t in treatment_strengths_amse:
            # AMSE
            df_test = pd.DataFrame(test_patients)
            df_test["d"] = t
            true_outcomes, pred_dose_response = get_true_and_pred_y(
                model, df_test, patient, hypernet, dataset, treatment_idx, step_size
            )
            amse = m.mean_squared_error(true_outcomes, pred_dose_response)
            #  TODO fer mean sq error entre mean(true outcomes) i mean(pred)
            amses.append(amse)

        return mises, amses

    test_patients = dataset_test["x"]

    def evaluate_model(model, test_patients, indices, hypernet=False):
        all_mises = []
        sqrt_mise_values = []

        all_amses = []
        amses_values = []

        for i in indices:
            # print(i, hypernet)
            mises, amses = eval_model(model[i], test_patients, i, hypernet=hypernet)
            all_mises.extend(mises)
            sqrt_mise = np.sqrt(np.mean(mises))
            sqrt_mise_values.append(sqrt_mise)

            all_amses.extend(amses)
            amses_values.append(np.mean(amses))

        final_mise = np.sqrt(np.mean(all_mises))
        final_amse = np.mean(all_amses)

        return sqrt_mise_values, final_mise, amses_values, final_amse

    # Define your models and parameters
    models_rf = [rf_0, rf_1, rf_2]
    models_baseNN = [baseNN_0, baseNN_1, baseNN_2]
    models_hypernet = [hypernet_0, hypernet_1, hypernet_2]

    indices = [0, 1, 2]

    # Evaluate models
    rf_sqrt_mise_values, rf_final_mise, rf_amses, rf_final_amse = evaluate_model(
        models_rf, test_patients, indices
    )
    (
        baseNN_sqrt_mise_values,
        baseNN_final_mise,
        baseNN_amses,
        baseNN_final_amse,
    ) = evaluate_model(models_baseNN, test_patients, indices)
    (
        hypernet_sqrt_mise_values,
        hypernet_final_mise,
        hypernet_amses,
        hypernet_final_amse,
    ) = evaluate_model(models_hypernet, test_patients, indices, hypernet=True)

    # Create a DataFrame to store the results
    data = {
        "Model": [
            "RF 0",
            "RF 1",
            "RF 2",
            "Final RF",
            "BaseNN 0",
            "BaseNN 1",
            "BaseNN 2",
            "Final BaseNN",
            "Hypernet 0",
            "Hypernet 1",
            "Hypernet 2",
            "Final Hypernet",
        ],
        "Sqrt MISE": rf_sqrt_mise_values
        + [rf_final_mise]
        + baseNN_sqrt_mise_values
        + [baseNN_final_mise]
        + hypernet_sqrt_mise_values
        + [hypernet_final_mise],
        "AMSE": rf_amses
        + [rf_final_amse]
        + baseNN_amses
        + [baseNN_final_amse]
        + hypernet_amses
        + [hypernet_final_amse],
    }

    results_df = pd.DataFrame(data)
    all_iterations.append(results_df)

print("End of Eval", time.time() - start)

enable_print()
# Print the table
from scipy.stats import sem

full_df = pd.concat(all_iterations, ignore_index=True)
# Group by Model and compute mean and SEM
summary_df = (
    full_df.groupby("Model")
    .agg({"Sqrt MISE": ["mean", sem], "AMSE": ["mean", sem]})
    .reset_index()
)

# Optional: flatten the MultiIndex columns
summary_df.columns = [
    "Model",
    "Sqrt MISE Mean",
    "Sqrt MISE SEM",
    "AMSE Mean",
    "AMSE SEM",
]

print(summary_df)
