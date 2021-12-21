#!/usr/bin/env python
"""
This is single-file script for performing GVA-net- based classification of ModelNet40
on its original train-validation split.

In aspects of data preprocessing (including: downloading, loading, batch creation) 
as well as kNN graph construction (methods: knn & get_graph_feature)
this code heavily borrows from DGCNN implementation of:

Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019). 
Dynamic graph cnn for learning on point clouds. Acm Transactions On Graphics (tog), 38(5), 1-12
(https://github.com/AnTao97/dgcnn.pytorch)


"""
from comet_ml import Experiment
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import h5py
from sklearn import metrics
from torch.utils.data import Dataset
import random

__author__ = "Jakub Walczak"
__copyright__ = "Copyright 2021"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Jakub Walczak"
__email__ = "jakub.walczak@p.lodz.pl"

# #####################################
#       SETTING-UP ENV
# #####################################
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


SEED = 4077961
seed_everything(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.use_deterministic_algorithms(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #####################################
#           DATA HANDLING
# #####################################
def load_data_cls(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_data = []
    all_label = []
    for h5_name in glob.glob(
        os.path.join(DATA_DIR, "modelnet40*hdf5_2048", "*%s*.h5" % partition)
    ):
        f = h5py.File(h5_name, "r")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition="train"):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition
        self.idx = {i: [] for i in range(40)}
        for i, l in enumerate(self.label.squeeze()):
            self.idx[l].extend([i])

    def __getitem__(self, item):
        anchor = self.data[item][: self.num_points]
        label = self.label[item]
        if self.partition == "train":
            anchor = translate_pointcloud(anchor)
            np.random.shuffle(anchor)
        return anchor, label

    def __len__(self):
        return self.data.shape[0]


# #####################################
#           MODEL DEFINITION
# #####################################
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, sorted=True, largest=False)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, md=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class GVANet(nn.Module):
    def __init__(self, k=32, embeds=256, out_dim=40):
        super(GVANet, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 8), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64 * 2),
            nn.Conv2d(64 * 2, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 8), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64 + 64, 64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embeds, kernel_size=1, bias=True),
            nn.BatchNorm1d(embeds),
            nn.ReLU(),
            nn.Conv1d(embeds, out_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x0):
        batch_size = x0.size(0)

        x = get_graph_feature(x0, k=self.k)
        x1 = self.conv1(x).max(-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x).max(-1)[0]

        x = torch.cat([x1, x2], dim=1)
        x = self.cls(x)

        return x.view(batch_size, -1)


class FocalLoss(nn.modules.loss._WeightedLoss):
    # https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
    def __init__(self, weight=None, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight
        )
        ce_loss = ce_loss.mean()
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# #####################################
#       TRAIN-VALIDATE PROCEDURE
# #####################################
def train(
    epochs, k, embeds, lr, momentum, used_pts, weight_decay, gamma, bs, test_freq=5
):
    NUM_CLS = 40
    experiment = Experiment(
        project_name="gva-net-classification",
        workspace="james1",
        log_code=True,
        disabled=False,
    )

    print("Creating Model...")
    model = GVANet(k=k, out_dim=NUM_CLS, embeds=embeds).to(device)
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}")
    print("Done...\n")
    experiment.add_tag("cls")
    experiment.add_tag("deterministic")
    experiment.log_parameters(
        {
            "lr": lr,
            "embeds": embeds,
            "k": k,
            "used_pts": used_pts,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "batch_size": bs,
            "focal_loss_gamma": gamma,
            "trainable_params": trainable_params,
        }
    )
    print("Creating DataLoader...")
    train_loader = torch.utils.data.DataLoader(
        ModelNet40(partition="train", num_points=used_pts), batch_size=bs, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        ModelNet40(partition="test", num_points=used_pts), batch_size=bs, shuffle=False
    )
    print("Done...\n")

    print("Setting-up optimizer, loss function, and scheduler...")
    criterion = FocalLoss(gamma=gamma)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=100 * lr, weight_decay=weight_decay, momentum=momentum
    )

    schedular = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: -0.0000028 * epoch * epoch + 0.7
    )
    print("Done...\n")

    print("Start training...")
    best_macc = 0
    associated_allacc = 0

    for epoch in range(1, epochs + 1):
        ####################
        # Train
        ####################
        model.train()

        print(f"--------Epoch {epoch}--------")
        tqdm_batch = tqdm(train_loader, desc=f"Epoch-{epoch} training")

        # train
        model.train()
        curr_pred = []
        curr_true = []
        for data, label in tqdm_batch:
            curr_true.append(label)
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            pred = model(data)
            loss = criterion(pred, label.view(-1))
            curr_pred.append(pred.max(1)[1].cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del data, label
        schedular.step()

        experiment.log_metric("lr", optimizer.param_groups[0]["lr"], epoch=epoch)
        curr_pred = np.concatenate(curr_pred)
        curr_true = np.concatenate(curr_true)
        allAcc = metrics.accuracy_score(curr_true, curr_pred)
        mAcc = metrics.balanced_accuracy_score(curr_true, curr_pred)
        experiment.log_metric("train_allAcc", allAcc, epoch=epoch)
        experiment.log_metric("train_mAcc", mAcc, epoch=epoch)
        print(f"train_allAcc={allAcc}, train_mAcc={mAcc}")
        tqdm_batch.close()

        if epoch % test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc=f"Epoch-{epoch} testing")

            model.eval()
            curr_pred = []
            curr_true = []

            with torch.no_grad():
                for data, label in tqdm_batch:
                    curr_true.append(label)
                    data, label = data.to(device), label.to(device)
                    data = data.permute(0, 2, 1)
                    pred = model(data)
                    curr_pred.append(pred.max(-1)[1].cpu().numpy())

            curr_pred = np.concatenate(curr_pred)
            curr_true = np.concatenate(curr_true)
            allAcc = metrics.accuracy_score(curr_true, curr_pred)
            mAcc = metrics.balanced_accuracy_score(curr_true, curr_pred)
            experiment.log_metric("val_allAcc", allAcc, epoch=epoch)
            experiment.log_metric("val_mAcc", mAcc, epoch=epoch)
            print(f"val_allAcc={allAcc}, val_mAcc={mAcc}")
            if best_macc < mAcc:
                best_macc = mAcc
                associated_allacc = allAcc
                torch.save(model.state_dict(), "gvanet_best_weights.t7")

            tqdm_batch.close()
    experiment.log_metric("best_mAcc", best_macc)
    experiment.log_metric("associated_allAcc", associated_allacc)


if __name__ == "__main__":
    for _ in range(10):
        args = {
            "epochs": 500,
            "k": 32,
            "embeds": 256,
            "lr": 0.001,
            "momentum": 0.95,
            "used_pts": 1024,
            "weight_decay": 0.8e-4,
            "gamma": 1,
            "bs": 16,
            "test_freq": 2,
        }
        train(**args)
