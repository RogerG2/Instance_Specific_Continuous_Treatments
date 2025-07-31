import pandas as pd
import numpy as np
import torch

from scipy.integrate import romb
from sklearn import metrics as m


# def compute_MISE(pred_test, X_test, model, num_integration_samples, step_size):
def compute_MISE(
    X_test,
    models,
    t_x_y,
    treatment_strengths,
    num_integration_samples,
    T_feature="T",
    v1=None,
    v2=None,
    v3=None,
):
    # models = ["rf", "baseNN", "HyperNN", "VCNet"]
    # Posar la prediccio a dins de la funcio
    step_size = 1.0 / num_integration_samples

    pred_test = {m: [] for m in models.keys()}
    pred_test["Y"] = []

    coefs_test = None
    if "HyperNN" in models.keys():
        coefs_test = models["HyperNN"][0].predict(X_test)

    # ts = treatment_strengths[0]
    for ts in treatment_strengths:

        X_test_cf = pd.DataFrame(X_test.copy())
        ts_array = np.array([ts] * X_test.shape[0])

        if v1 is None:
            Y_test_cf = []
            for i in range(X_test_cf.shape[0]):

                val = t_x_y(
                    torch.from_numpy(np.array(ts)),
                    torch.from_numpy(X_test[i, :]),
                )
                Y_test_cf.append(val)

            Y_test_cf = np.array(Y_test_cf)

        else:
            Y_test_cf = t_x_y(
                ts_array,
                X_test,
                v1,
                v2,
                v3,
            )

        X_test_cf[T_feature] = ts_array

        pred_test["Y"].append(Y_test_cf)
        for model, (model_object, model_pred) in models.items():
            pred_test[model].append(
                model_pred(
                    model_object, X_test_cf, T_feature, coefs_test, ts_array, X_test
                )
            )

        # pred_test["rf"].append(rf.predict(X_test_cf.to_numpy()))
        # pred_test["baseNN"].append(baseNN.predict(X_test_cf.to_numpy()))
        # pred_test["HyperNN"].append(
        #     HyperNN.eval_hypernet_full(coefs_test, X_test_cf[T_feature].to_numpy())
        # )
        # pred_test["VCNet"].append(
        #     predict_VCNet(
        #         VCNet_model,
        #         torch.from_numpy(X_test).float(),
        #         torch.from_numpy(ts_array).float(),
        #     )
        # )

    converted_data = {}
    for key, list_of_arrays in pred_test.items():
        flat_list = [item for array in list_of_arrays for item in array]
        converted_data[key] = flat_list

    df_pred = pd.DataFrame(converted_data)
    df_pred["sample_id"] = list(range(X_test.shape[0])) * num_integration_samples

    mises = {model: [] for model in models.keys()}
    for sid in range(X_test.shape[0]):
        # print(f"IteratioN: {sid}")
        true_outcomes = df_pred[df_pred["sample_id"] == sid]["Y"]

        for model in models.keys():
            pred = df_pred[df_pred["sample_id"] == sid][model]
            # Compute the MISE for the model
            mise = romb(np.square(true_outcomes - pred), dx=step_size)
            mises[model].append(mise)

    # Convert to a DataFrame for easier analysis
    df_mises = pd.DataFrame(mises)
    average_mises = df_mises.mean()
    return average_mises


def compute_AMSE(
    X_test,
    models,
    t_x_y,
    T_test,
    T_feature,
    v1=None,
    v2=None,
    v3=None,
):
    # AMSE ----

    coefs_test = None
    if "HyperNN" in models.keys():
        coefs_test = models["HyperNN"][0].predict(X_test)

    pred_test = {m: [] for m in models.keys()}
    pred_test["Y"] = []

    for ts in T_test:

        X_test_cf = pd.DataFrame(X_test.copy())
        ts_array = np.array([ts] * X_test.shape[0])

        if v1 is None:
            Y_test_cf = []
            for i in range(X_test_cf.shape[0]):

                val = t_x_y(
                    torch.from_numpy(np.array(ts)),
                    torch.from_numpy(X_test[i, :]),
                )
                Y_test_cf.append(val)

            Y_test_cf = np.array(Y_test_cf)

        else:
            Y_test_cf = t_x_y(
                ts_array,
                X_test,
                v1,
                v2,
                v3,
            )

        X_test_cf[T_feature] = ts_array

        pred_test["Y"].append(Y_test_cf)
        for model, (model_object, model_pred) in models.items():
            pred_test[model].append(
                model_pred(
                    model_object, X_test_cf, T_feature, coefs_test, ts_array, X_test
                )
            )

    # Compute amse
    converted_data = {}
    for key, list_of_arrays in pred_test.items():
        flat_list = [item for array in list_of_arrays for item in array]
        converted_data[key] = flat_list

    df_pred = pd.DataFrame(converted_data)
    # df_pred["sample_id"] = list(range(X_test.shape[0])) * X_test.shape[0]

    # Initialize t_id column and model dictionaries
    df_pred["t_id"] = np.repeat(np.arange(X_test.shape[0]), X_test.shape[0])

    amses = {model: [] for model in models.keys()}
    amses_mean = {model: [] for model in models.keys()}

    # Group by t_id to avoid repeated filtering
    grouped = df_pred.groupby("t_id")

    # Iterate over each group
    for tid, group in grouped:
        true_outcomes = group["Y"].values
        true_outcomes_mean = np.mean(true_outcomes)

        for model in models.keys():
            pred = group[model].values
            if isinstance(pred[0], torch.Tensor):
                pred = [p.data.numpy() for p in pred]
            pred_mean = np.mean(pred)

            amse = m.mean_squared_error(true_outcomes, pred)
            amses[model].append(amse)
            amses_mean[model].append((true_outcomes_mean - pred_mean) ** 2)
    ###############################################################################
    # Convert to a DataFrame for easier analysis
    df_amses = pd.DataFrame(amses)
    average_amses = df_amses.mean()
    df_amses_mean = pd.DataFrame(amses_mean)

    return average_amses, df_amses_mean.mean()
