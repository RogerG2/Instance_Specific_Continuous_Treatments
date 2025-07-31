import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

# from torchviz import make_dot
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def MSE_loss(pred, y_true):
    return nn.MSELoss()(pred, y_true)


# Define the network class
class predictiveNet(nn.Module):
    # Initialize the network layers
    def __init__(self, p):
        """p=params"""
        super(predictiveNet, self).__init__()
        torch.set_default_dtype(torch.float32)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(p["input_dim"], p["neurons_per_layer"]))
        for i in range(p["hidden_layers"] - 1):
            self.layers.append(
                nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
            )
            self.layers.append(nn.Dropout(p["dropout"]))
        self.layers.append(nn.Linear(p["neurons_per_layer"], p["output_size"]))

    # Define the forward pass
    def forward(self, x):

        for layer in self.layers[:-1]:
            # Apply linear transformation and ReLU activation
            x = torch.relu(layer(x))
        # Apply the last linear transformation without activation
        x = self.layers[-1](x)
        return x


class predictiveNetTrainer:
    def __init__(self, train_X, train_y, params=None):

        # Hyperparameters --
        params_default = {
            "neurons_per_layer": 200,
            "hidden_layers": 3,
            "reg_l2": 0.01,
            "val_split": 0.2,
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 1e-5,
            "momentum": 0.9,
            "output_dim": 5,
            "early_stopping": 40,
            "loss": "MSE",
            "dropout": 0,
            "output_size": 1,
        }
        if params is not None:
            for key, value in params.items():
                params_default[key] = value

        params = params_default

        self.batch_size = params["batch_size"]
        self.num_epochs = params["epochs"]
        self.early_stopping_value = params["early_stopping"]
        params["input_dim"] = train_X.shape[1]

        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 50

        self.train_losses = []
        self.val_losses = []

        # --- SCALE ----
        self.y_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
        train_y = self.y_scaler.transform(train_y.reshape(-1, 1))

        self.X_scaler = StandardScaler().fit(train_X)
        train_X = self.X_scaler.transform(train_X)
        # --------------

        self.batched_X = (
            torch.from_numpy(self.array_to_batch(train_X, self.batch_size))
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
        val_ids = [i for i in range(len(self.batched_X)) if i not in train_ids]
        self.batched_train_X = self.batched_X[train_ids]
        self.batched_train_y = self.batched_y[train_ids]
        self.batched_val_X = self.batched_X[val_ids]
        self.batched_val_y = self.batched_y[val_ids]

        self.model = predictiveNet(p=params)
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=params["learning_rate"],
        #     momentum=params["momentum"],
        #     nesterov=True,
        # )
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

        if params["loss"] == "MSE":
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
            # X_batch, y_batch = next(aa)
            for X_batch, y_batch in zip(self.batched_train_X, self.batched_train_y):
                # print(i)
                # i += 1
                self.optimizer.zero_grad()
                pred = self.model(X_batch)

                loss = self.criterion(pred, y_batch)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            self.scheduler.step(np.mean(losses))
            self.train_losses.append(np.mean(losses))

            # Validation
            losses = []
            with torch.no_grad():
                for X_batch, y_batch in zip(self.batched_val_X, self.batched_val_y):
                    # self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)

                    loss = self.criterion(y_pred, y_batch)
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
        y_test = self.model(test_X)
        return self.y_scaler.inverse_transform(y_test.detach().numpy())

    @staticmethod
    def array_to_batch(data, batch_size):

        num_batches = np.floor(len(data) / batch_size)

        if len(data) % batch_size == 0:
            batches = np.array_split(data, num_batches)
        else:
            batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

        return np.array(batches)
    
    @staticmethod
    def _return_name():
        return "baseNN"
