import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def fixed_network(x, weights_pred, num_layers, activation="elu"):
    weights_pred = weights_pred.reshape(-1, 1).T

    if num_layers == 0:
        n = int((weights_pred.shape[1] - 1) / 3)
    else:
        n = (
            -(3 + num_layers)
            + np.sqrt(
                (3 + num_layers) ** 2 - 4 * num_layers * (1 - weights_pred.shape[1])
            )
        ) / (2 * num_layers)
        n = int(n)

    # Define the layer shapes dynamically
    layer_shapes = []

    # Input layer (size 1 -> n)
    layer_shapes.append([(n, 1), (n,)])

    # Hidden layers (size n -> n)
    for _ in range(num_layers - 1):
        layer_shapes.append([(n, n), (n,)])

    # Output layer (size n -> 1)
    layer_shapes.append([(1, n), (1,)])

    idx, params = 0, []
    for layer in layer_shapes:
        layer_params = []
        for shape in layer:
            offset = np.prod(shape)
            layer_params.append(weights_pred[:, idx : idx + offset].reshape(shape))
            idx += offset
        params.append(layer_params)

    # Pass through layers
    # act = getattr(F, activation)
    activations = {"elu": F.elu, "relu": F.relu, "linear": lambda x: x}
    for i, (w, b) in enumerate(params):
        if i == 0:
            x = activations[activation](F.linear(x.reshape(1, 1), w, b))
        elif i == len(params) - 1:
            x = F.linear(x, w, b)  # Output layer
        else:
            x = activations[activation](F.linear(x, w, b))  # Hidden layers

    return x


def fixed_network_no_hidden(x, weights_pred):

    weights_pred = weights_pred.reshape(-1, 1).T
    n = int((weights_pred.shape[1] - 1) / 3)

    layer_shapes = [
        [(n, 1), (n,)],  # (w0, b0)
        [(1, n), (1,)],  # (w1, b1)
    ]

    idx, params = 0, []
    for layer in layer_shapes:
        layer_params = []
        for shape in layer:
            offset = np.prod(shape)
            layer_params.append(weights_pred[:, idx : idx + offset].reshape(shape))
            idx += offset
        params.append(layer_params)

    w0, b0 = params[0]
    w1, b1 = params[1]

    x = F.elu(F.linear(x.reshape(1, 1), w0, b0))
    # print(x)
    x = F.linear(x, w1, b1)
    # print(x)
    return x


def MSE_loss(pred, t_pred, y_true, t_true, **kwargs):
    return nn.MSELoss()(pred, y_true)


def propensity_loss(pred, t_pred, y_true, t_true, alpha=1, neg_gaussian=False):

    pred_loss = nn.MSELoss()(pred, y_true)
    # print(alpha)
    if neg_gaussian:
        # print("Using negative gaussian loss")
        mu, logvar = t_pred[:, 0], t_pred[:, 1]
        propensity_loss = (
            0.5 * (((t_true - mu) ** 2) / torch.exp(logvar) + logvar).mean()
        )
    else:
        propensity_loss = nn.MSELoss()(t_pred, t_true)
    return pred_loss + alpha * propensity_loss


# Define the network class
class HNET(nn.Module):
    # Initialize the network layers
    def __init__(self, p):
        """p=params"""
        self.p = p
        super(HNET, self).__init__()
        torch.set_default_dtype(torch.float32)

        # Representation
        self.commonlayers = nn.ModuleList()
        self.commonlayers.append(nn.Linear(p["input_dim"], p["neurons_per_layer"]))
        for i in range(p["hidden_layers"] - 1):
            # if p["divide_npl"]:
            #     previous_neurons = p["neurons_per_layer"]
            #     p["neurons_per_layer"] = int(previous_neurons / 2)
            # else:
            #     previous_neurons = p["neurons_per_layer"]

            self.commonlayers.append(
                nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
            )
            self.commonlayers.append(nn.Dropout(p["dropout"]))

        self.output_layer = nn.Linear(p["neurons_per_layer"], p["output_dim"])
        self.t_layer = nn.Linear(p["neurons_per_layer"], 2 if p["neg_gaussian"] else 1)

        if p["spectral_norm"]:
            for layer in self.commonlayers + [self.output_layer]:
                if isinstance(layer, nn.Linear):
                    torch.nn.utils.parametrizations.spectral_norm(layer)

    # Define the forward pass
    def forward(self, x):

        for layer in self.commonlayers:
            x = torch.nn.ReLU()(layer(x))

        out = self.output_layer(x)
        t_pred = self.t_layer(x) if self.p["use_propensity"] else 0

        return out, t_pred
        # return torch.nn.Sigmoid()(out)


class HNETTrainer:
    def __init__(self, train_X, train_t, train_y, dag_edges=None, params=None):

        # Hyperparameters --
        params_default = {
            "neurons_per_layer": 200,
            "reg_l2": 0.01,
            "val_split": 0.2,
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 1e-5,
            "momentum": 0.9,
            "output_dim": 5,
            "hidden_net_layers": 0,
            "hidden_net_activation": "elu",
            "early_stopping": 30,
            "loss": "hypernetwork",
            "hidden_layers": 3,
            "dropout": 0,
            "spectral_norm": True,
            # "divide_npl": False,
            "use_propensity": False,
            "neg_gaussian": False,
            "alpha": 1,
        }
        if params is not None:
            for key, value in params.items():
                params_default[key] = value

        params = params_default
        self.params = params

        self.batch_size = params["batch_size"]
        self.num_epochs = params["epochs"]
        self.early_stopping_value = params["early_stopping"]
        params["input_dim"] = train_X.shape[1]

        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 500

        self.train_losses = []
        self.val_losses = []

        # --- SCALE ----
        self.y_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
        train_y = self.y_scaler.transform(train_y.reshape(-1, 1))

        self.t_scaler = StandardScaler().fit(train_t.reshape(-1, 1))
        train_t = self.t_scaler.transform(train_t.reshape(-1, 1))

        self.X_scaler = StandardScaler().fit(train_X)
        train_X = self.X_scaler.transform(train_X)
        # --------------

        self.batched_X = (
            torch.from_numpy(self.array_to_batch(train_X, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_t = (
            torch.from_numpy(self.array_to_batch(train_t, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_y = (
            torch.from_numpy(self.array_to_batch(train_y, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        train_ids = np.random.choice(
            range(len(self.batched_X)),
            int(len(self.batched_X) * (1 - params["val_split"])),
        )
        if params["val_split"] != 0:
            val_ids = [i for i in range(len(self.batched_X)) if i not in train_ids]
        else:
            val_ids = train_ids

        self.batched_train_X = self.batched_X[train_ids]
        self.batched_train_t = self.batched_t[train_ids]
        self.batched_train_y = self.batched_y[train_ids]
        self.batched_val_X = self.batched_X[val_ids]
        self.batched_val_t = self.batched_t[val_ids]
        self.batched_val_y = self.batched_y[val_ids]

        self.model = HNET(p=params)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params["learning_rate"],
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=50,
            verbose=True,
            # mode="auto",
            eps=1e-8,
            cooldown=0,
            min_lr=1e-6,
        )

        if params["loss"] == "polynomial":
            self.criterion = polynomial_loss
        elif params["loss"] == "hypernetwork":
            self.criterion = self.hypernetwork_loss
        elif params["loss"] == "MSE":
            self.criterion = MSE_loss
        else:
            raise ValueError("Loss must be polynomial, hypernetwork, MSE")

    def train(self):
        self.model.train()
        self.best_val_loss = float("inf")
        loss_counter = 0
        for epoch in range(1, self.num_epochs + 1):
            losses = []
            # i= 0
            # aa =  zip(
            #     self.batched_train_X, self.batched_train_t, self.batched_train_y
            # )
            # X_batch, t_batch, y_batch = next(aa)
            for X_batch, t_batch, y_batch in zip(
                self.batched_train_X, self.batched_train_t, self.batched_train_y
            ):
                # print(i)
                # i += 1
                self.optimizer.zero_grad()
                y_pred, t_pred = self.model(X_batch)

                loss = self.criterion(y_pred, t_pred, y_batch, t_batch)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            self.scheduler.step(np.mean(losses))
            self.train_losses.append(np.mean(losses))

            # Validation
            losses = []
            with torch.no_grad():
                for X_batch, t_batch, y_batch in zip(
                    self.batched_val_X, self.batched_val_t, self.batched_val_y
                ):
                    # self.optimizer.zero_grad()
                    y_pred, t_pred = self.model(X_batch)

                    loss = self.criterion(y_pred, t_pred, y_batch, t_batch)
                    losses.append(loss.item())

            self.val_losses.append(np.mean(losses))

            # Save best model
            val_loss = self.val_losses[-1]
            if val_loss < self.best_val_loss:
                torch.save(self.model.state_dict(), "best_model.pth")
                self.best_val_loss = val_loss

            # Early stopping
            if self.early_stopping and epoch > 1:
                if self.val_losses[-1] > self.best_val_loss:

                    loss_counter += 1
                    if loss_counter == self.early_stopping_value:
                        # print(f"Early stopping epoch: {epoch}")
                        break

                else:
                    loss_counter = 0

            if epoch % self.print_every == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {self.train_losses[-1]}, , Val Loss: {self.val_losses[-1]}"
                )

        self.model.load_state_dict(torch.load("best_model.pth"))

    def predict(self, test_X):
        self.model.eval()
        test_X = self.X_scaler.transform(test_X)
        test_X = torch.from_numpy(test_X).to(torch.float32).to(self.device)
        pred, _ = self.model(test_X)
        return pred.detach().numpy()

    @staticmethod
    def array_to_batch(data, batch_size):

        num_batches = np.floor(len(data) / batch_size)

        if len(data) % batch_size == 0:
            batches = np.array_split(data, num_batches)
        else:
            batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

        return np.array(batches)

    @staticmethod
    def eval_polynomial(coef, T):
        return sum([coef[i] * T**i for i in range(len(coef))])

    @staticmethod
    def eval_polynomial_full(coefs, T):
        return [
            sum([coefs[j, i] * T[j] ** i for i in range(coefs.shape[1])])
            for j in range(coefs.shape[0])
        ]

    def eval_hypernet(self, coef, T):
        coef = torch.from_numpy(coef).to(torch.float32)
        T = self.t_scaler.transform(T.reshape(-1, 1))
        T = torch.tensor(T).to(torch.float32)
        y_pred = np.array(
            [
                [
                    fixed_network(
                        T,
                        coef,
                        self.params["hidden_net_layers"],
                        self.params["hidden_net_activation"],
                    )[0][0].item()
                ]
            ]
        )
        return self.y_scaler.inverse_transform(y_pred)[0][0]

    def eval_hypernet_full(self, coefs, T):
        coefs = torch.from_numpy(coefs).to(torch.float32)
        T = self.t_scaler.transform(T.reshape(-1, 1))
        T = torch.from_numpy(T).to(torch.float32)
        pred = [
            self.y_scaler.inverse_transform(
                np.array(
                    [
                        [
                            fixed_network(
                                T[j],
                                coefs[j, :],
                                self.params["hidden_net_layers"],
                                self.params["hidden_net_activation"],
                            )[0][0].item()
                        ]
                    ]
                )
            )[0]
            for j in range(coefs.shape[0])
        ]
        return [x[0] for x in pred]

    @staticmethod
    def _return_name():
        return "HNet"

    def hypernetwork_loss(self, pred, t_pred, y_true, t_true):
        # pred = y_pred

        # hyper_pred = torch.zeros(pred.shape[0], requires_grad=True)
        hyper_pred = []
        # i = 0
        for i in range(pred.shape[0]):

            weights_pred = pred[i, :]
            # weights1, biases1, weights2, biases2 = torch.chunk(point_pred, 4, dim=0)
            # model = FixedNetwork(n_half, weights1, biases1, weights2, biases2)
            hyper_pred.append(
                fixed_network(
                    t_true[i],
                    weights_pred,
                    self.params["hidden_net_layers"],
                    self.params["hidden_net_activation"],
                )
            )

        loss_fn = propensity_loss if self.params["use_propensity"] else MSE_loss
        hyper_pred = torch.cat(hyper_pred, dim=0)  # .reshape(-1)
        return loss_fn(
            hyper_pred,
            t_pred,
            y_true,
            t_true,
            alpha=self.params["alpha"],
            neg_gaussian=self.params["neg_gaussian"],
        )
