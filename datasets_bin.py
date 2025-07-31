import pandas as pd
import numpy as np
from scipy.special import expit, logit
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from copy import deepcopy
from scipy.special import rel_entr

# from causalml.dataset import synthetic_data
# import openml
from custom_synthetic import CustomGraph, CustomNode


class dataLoader:
    def __init__(
        self,
        data_path="../../../data/",
        n_instances=1000,
        n_vars=5,
        X_noise=0,
        Y_noise=1,
        test_perc=0.2,
        cv_splits=5,
        experiment_params=None,
        intervene_test=False,
        realcause_df=False,
        ihdp_i=0,
    ) -> None:

        self.data_path = data_path

        self.datasets = {
            "ihdp": self.load_ihdp_data,
        }
        self.X_noise = X_noise
        self.Y_noise = Y_noise
        self.n_instances = n_instances
        self.n_vars = n_vars
        self.test_perc = test_perc
        self.cv_splits = cv_splits
        self.intervene_test = intervene_test
        self.realcause_df = realcause_df
        self.experiment_params = experiment_params
        if experiment_params is not None:
            if "cv_splits" in experiment_params.keys():
                self.cv_splits = experiment_params["cv_splits"]
            if "n_instances" in experiment_params.keys():
                self.n_instances = experiment_params["n_instances"]
            if "n_vars" in experiment_params.keys():
                self.n_vars = experiment_params["n_vars"]
            if "X_noise" in experiment_params.keys():
                self.X_noise = experiment_params["X_noise"]
            if "Y_noise" in experiment_params.keys():
                self.Y_noise = experiment_params["Y_noise"]
            if "test_perc" in experiment_params.keys():
                self.test_perc = experiment_params["test_perc"]
            if "intervene_test" in experiment_params.keys():
                self.test_perc = experiment_params["intervene_test"]

        self.test_weights = None
        self.ihdp_i = ihdp_i
        self.train_df_orig = None

    def load_dataset(self, dataset_name):
        self.datasets[dataset_name]()
        kf = KFold(
            n_splits=self.cv_splits,
            # shuffle=self.random_state is not None,
            shuffle=True,
            # random_state=self.random_state,
        )
        self.cv_indexes = list(kf.split(self.train_df))

    def load_ihdp_data(self):

        # data= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
        # col =  ["T", "y_factual", "y_cfactual", "mu0", "mu1"]
        # X = []
        # for i in range(1,26):
        #     X.append("x"+str(i))
        # data.columns = col + X
        # data = data.astype({"treatment":'bool'}, copy=False)
        i = self.ihdp_i
        data = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.train.npz")
        data_test = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.test.npz")

        train_df = pd.DataFrame(data["x"][:, :, i])
        colnames = [f"x{c}" for c in range(train_df.shape[1])]
        train_df.columns = colnames
        train_df["Y"] = data["yf"][:, i]
        train_df["T"] = data["t"][:, i]

        test_df = pd.DataFrame(data_test["x"][:, :, i])
        test_df.columns = colnames
        test_df["Y"] = data_test["yf"][:, i]
        test_df["T"] = data_test["t"][:, i]

        self.X_features = colnames + ["T"]
        self.Y_feature = ["Y"]

        self.train_df = train_df
        self.test_df = test_df
        self.train_ite = data["mu1"][:, i] - data["mu0"][:, i]
        self.test_ite = data_test["mu1"][:, i] - data_test["mu0"][:, i]
        self.mu0_train = data["mu0"][:, i]
        self.mu1_train = data["mu1"][:, i]
        self.mu0_test = data_test["mu0"][:, i]
        self.mu1_test = data_test["mu1"][:, i]

    def categorize_numeric_target_hist(self, nbins=10):

        _, division = np.histogram(self.train_df[self.Y_feature].values, bins=nbins)
        division[0] -= 1
        categories_ids = pd.cut(
            self.train_df[self.Y_feature[0]], bins=division
        ).cat.codes
        self.train_df[self.Y_feature[0]] = categories_ids

    def categorize_numeric_target_unif(self, nbins=10):
        def interval_divisions(min_val, max_val, num_points):
            interval = (max_val - min_val) / (num_points - 1)
            divisions = []
            for i in range(num_points):
                divisions.append(min_val + i * interval)

            return divisions

        division = interval_divisions(
            self.train_df[self.Y_feature[0]].min(),
            self.train_df[self.Y_feature[0]].max(),
            nbins + 1,
        )
        division[0] -= 1

        categories_ids = pd.cut(
            self.train_df[self.Y_feature[0]], bins=division
        ).cat.codes
        self.train_df[self.Y_feature[0]] = categories_ids

    def swap_target_treatment(self):

        self.X_features = [x for x in self.X_features if x != "T"]  # + self.Y_feature
        self.Y_feature = ["T"]

    def bootstrap_train(self):
        if self.train_df_orig is None:
            self.train_df_orig = self.train_df.copy()
        self.train_df = self.train_df.sample(
            n=len(self.train_df_orig), replace=True
        ).reset_index(drop=True)
