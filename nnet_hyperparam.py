import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from functools import partial
import json
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# from hyperopt import fmin, tpe, hp, space_eval
# import xgboost
# from xgboost import XGBRegressor
import copy

from hyperopt import fmin, tpe, hp, space_eval
from predictive_net import predictiveNetTrainer

# aa = [      22,
#        22,]


class hyperparamTunner:
    def __init__(
        self,
        train_df,
        X_features,
        Y_feature,
        model,
        t=None,
        model_params=None,
        cv_indexes=None,
        task="regression",
        max_hyperopt_evals=10,
        n_startup_jobs=5,
    ) -> None:

        self.trainer = model
        self.df = train_df[X_features + Y_feature]

        self.t = t
        self.X_features = X_features
        self.Y_feature = Y_feature
        self.model_params = model_params
        self.cv_indexes = cv_indexes
        self.task = task

        self.metrics_dict_regression = {
            "MSE": metrics.mean_squared_error,
            "MAE": metrics.mean_absolute_error,
            "MAPE": metrics.mean_absolute_percentage_error,
        }
        self.metrics_dict_classification = {
            "Acc": metrics.accuracy_score,
        }
        self.metrics_dict = (
            self.metrics_dict_regression
            if task == "regression"
            else self.metrics_dict_classification
        )

        # Hyperopt params
        self.search_space = self.hyperopt_params_models(model._return_name())
        self.max_hyperopt_evals = max_hyperopt_evals
        self.n_startup_jobs = n_startup_jobs

    def train_model_nn(self, params=None):

        X = self.df[self.X_features].to_numpy()
        y = self.df[self.Y_feature[0]].to_numpy()

        if self.t is not None:
            self.model = self.trainer(X, self.t, y, params=params)
        else:
            self.model = self.trainer(X, y, params=params)

        self.model.train()
        self.method = "Vanilla"

    def hyperopt_objective(self, space):

        params = space
        # print(params)
        self.hyperopt_train_func(params)
        val_loss = self.model.best_val_loss

        return val_loss

    def run_hyperopt_optimization(self):

        algo = partial(tpe.suggest, n_startup_jobs=self.n_startup_jobs, verbose=False)

        return fmin(
            self.hyperopt_objective,
            self.nnet_search_space,
            algo=algo,
            max_evals=self.max_hyperopt_evals,
        )

    def train_and_evaluate_hyperopt_nnet(self):

        self.nnet_search_space = self.search_space
        self.hyperopt_train_func = self.train_model_nn
        self.best_params_json_name = "best_params.json"

        best_params = self.run_hyperopt_optimization()
        best_params = space_eval(self.nnet_search_space, best_params)
        self.best_params = best_params.copy()
        with open(self.best_params_json_name, "w") as f:
            json.dump(best_params, f)
        f.close()

        return best_params

    @staticmethod
    def hyperopt_params_models(model):
        # Hyperopt params
        hyperparam_dicts = {
            "baseNN": {
                # "num_epochs": hp.choice("num_epochs", [10, 100]),
                "batch_size": hp.choice("batch_size", [16, 32, 64]),
                "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4, 5]),
                "neurons_per_layer": hp.choice(
                    "neurons_per_layer", list(range(50, 600, 50))
                ),
                "learning_rate": hp.choice(
                    "lr", [x / 10000 for x in list(range(1, 100, 5))]
                ),
                "dropout": hp.choice(
                    "dropout", [x / 100 for x in list(range(0, 50, 5))]
                ),
                # "early_stopping": hp.choice("early_stopping", [True]),
                # "early_stopping_value": hp.choice(
                #     "early_stopping_value", list(range(1, 11))
                # ),
            },
            "HNet": {
                # "num_epochs": hp.choice("num_epochs", [10, 100]),
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128, 256]),
                "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4, 5]),
                "neurons_per_layer": hp.choice(
                    "neurons_per_layer", list(range(50, 500, 50))
                ),
                "learning_rate": hp.choice(
                    "learning_rate", [x / 10000 for x in list(range(1, 100, 5))]
                ),
                "output_dim": hp.choice("output_dim", list(range(4, 50, 2))),
                "spectral_norm": hp.choice("spectral_norm", [True, False]),
                # "divide_npl": hp.choice("divide_npl", [True, False]),
                "add_propensity": hp.choice("add_propensity", [False]),
                "dropout": hp.choice(
                    "dropout", [x / 100 for x in list(range(0, 50, 5))]
                ),
                # "early_stopping": hp.choice("early_stopping", [True]),
                # "early_stopping_value": hp.choice(
                #     "early_stopping_value", list(range(1, 11))
                # ),
            },
        }
        return hyperparam_dicts[model]
