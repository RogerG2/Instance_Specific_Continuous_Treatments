import numpy as np
import torch
import sys
import math

VCNET_PATH = (
    "/Users/rogerprosrius/projects/varying-coefficient-net-with-functional-tr-main/"
)

sys.path.append("..")
sys.path.append(VCNET_PATH)


from models.dynamic_net import Vcnet, Drnet, TR
from data.data import get_iter


from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR


def train_VCNet_synth(train_matrix, model_name="Vcnet"):
    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}

    train_matrix_torch = torch.from_numpy(train_matrix).float()
    train_loader = get_iter(train_matrix_torch, batch_size=500, shuffle=True)
    verbose = 100

    def criterion(out, y, alpha=0.5, epsilon=1e-6):
        return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(
            out[0] + epsilon
        ).mean()

    def criterion_TR(out, trg, y, beta=1.0, epsilon=1e-6):
        # out[1] is Q
        # out[0] is g
        return (
            beta
            * (
                (
                    y.squeeze()
                    - trg.squeeze() / (out[0].squeeze() + epsilon)
                    - out[1].squeeze()
                )
                ** 2
            ).mean()
        )

    cfg_density = [(6, 50, 1, "relu"), (50, 50, 1, "relu")]
    num_grid = 10
    cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
    degree = 2
    knots = [0.33, 0.66]
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()

    isTargetReg = "tr" in model_name  # 1
    if isTargetReg:
        tr_knots = list(np.arange(0.1, 1, 0.1))
        tr_degree = 2
        TargetReg = TR(tr_degree, tr_knots)
        TargetReg._initialize_weights()

    if model_name == "Vcnet":
        init_lr = 0.0001
        alpha = 0.5
    elif model_name == "Vcnet_tr":
        init_lr = 0.0001
        alpha = 0.5
        tr_init_lr = 0.001
        beta = 1.0

    # optimizer
    lr_type = "fixed"
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=momentum,
        weight_decay=wd,
        nesterov=True,
    )

    if isTargetReg:
        tr_optimizer = torch.optim.SGD(
            TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd
        )

    num_epoch = 800
    for epoch in range(num_epoch):

        for idx, (inputs, y) in enumerate(train_loader):
            t = inputs[:, 0]
            x = inputs[:, 1:]

            if isTargetReg:
                optimizer.zero_grad()
                out = model.forward(t, x)
                trg = TargetReg(t)
                loss = criterion(out, y, alpha=alpha) + criterion_TR(
                    out, trg, y, beta=beta
                )
                loss.backward()
                optimizer.step()

                tr_optimizer.zero_grad()
                out = model.forward(t, x)
                trg = TargetReg(t)
                tr_loss = criterion_TR(out, trg, y, beta=beta)
                tr_loss.backward()
                tr_optimizer.step()
            else:
                optimizer.zero_grad()
                out = model.forward(t, x)
                loss = criterion(out, y, alpha=alpha)
                loss.backward()
                optimizer.step()

        if epoch % verbose == 0:
            print("current epoch: ", epoch)
            print("loss: ", loss.data)

    return model, model_name


def train_VCNet_IHDP(train_matrix, model_name="Vcnet"):
    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}

    train_matrix_torch = torch.from_numpy(train_matrix).float()
    train_loader = get_iter(train_matrix_torch, batch_size=500, shuffle=True)
    verbose = 100

    def criterion(out, y, alpha=0.5, epsilon=1e-6):
        return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(
            out[0] + epsilon
        ).mean()

    def criterion_TR(out, trg, y, beta=1.0, epsilon=1e-6):
        # out[1] is Q
        # out[0] is g
        return (
            beta
            * (
                (
                    y.squeeze()
                    - trg.squeeze() / (out[0].squeeze() + epsilon)
                    - out[1].squeeze()
                )
                ** 2
            ).mean()
        )

    cfg_density = [(25, 50, 1, "relu"), (50, 50, 1, "relu")]
    num_grid = 10
    cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
    degree = 2
    knots = [0.33, 0.66]
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()

    if model_name == "Vcnet":
        init_lr = 0.001
        alpha = 0.5
        tr_init_lr = 0.001
        beta = 1.0

    # optimizer
    lr_type = "fixed"
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 1e-3

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=momentum,
        weight_decay=wd,
        nesterov=True,
    )

    num_epoch = 800
    losses = []
    for epoch in range(num_epoch):

        for idx, (inputs, y) in enumerate(train_loader):
            t = inputs[:, 0]
            x = inputs[:, 1:]

            optimizer.zero_grad()
            out = model.forward(t, x)
            loss = criterion(out, y, alpha=alpha)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        if epoch % verbose == 0:
            print("current epoch: ", epoch)
            print("loss: ", loss.data)

    return model, model_name, losses


def predict_VCNet(model, x, t):

    out = model.forward(t, x)
    return out[1].data.squeeze()
