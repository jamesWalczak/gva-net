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

from lovash import lovasz_softmax

__author__ = "Jakub Walczak"
__copyright__ = "Copyright 2021"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Jakub Walczak"
__email__ = "jakub.walczak@p.lodz.pl"


seg_classes = {
    5: [16, 17, 18],
    10: [30, 31, 32, 33, 34, 35],
    13: [41, 42, 43],
    3: [8, 9, 10, 11],
    9: [28, 29],
    2: [6, 7],
    14: [44, 45, 46],
    11: [36, 37],
    6: [19, 20, 21],
    1: [4, 5],
    8: [24, 25, 26, 27],
    15: [47, 48, 49],
    0: [0, 1, 2, 3],
    12: [38, 39, 40],
    4: [12, 13, 14, 15],
    7: [22, 23],
}

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
def load_data_partseg(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_data = []
    all_label = []
    all_seg = []
    if partition == "trainval":
        file = glob.glob(
            os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*train*.h5")
        ) + glob.glob(os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*val*.h5"))
    else:
        file = glob.glob(
            os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*%s*.h5" % partition)
        )
    for h5_name in file:
        f = h5py.File(h5_name, "r+")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        seg = f["pid"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition="train", class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {
            "airplane": 0,
            "bag": 1,
            "cap": 2,
            "car": 3,
            "chair": 4,
            "earphone": 5,
            "guitar": 6,
            "knife": 7,
            "lamp": 8,
            "laptop": 9,
            "motor": 10,
            "mug": 11,
            "pistol": 12,
            "rocket": 13,
            "skateboard": 14,
            "table": 15,
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        label = self.label[item]
        seg = self.seg[item][: self.num_points]
        if self.partition == "trainval":
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

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
            nn.Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 8), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(32 * 2, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
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

        self.conv3 = nn.Sequential(
            nn.Conv1d(64 + 32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embeds, kernel_size=1, bias=False),
            nn.BatchNorm1d(embeds),
            nn.ReLU(),
        )

        self.conv_labels = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embeds, kernel_size=1, bias=False),
            nn.BatchNorm1d(embeds),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(embeds * 2 + 64, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, x0, l):
        batch_size = x0.size(0)
        num_points = x0.size(2)

        x = get_graph_feature(x0, k=self.k)
        x1 = self.conv1(x).max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x).max(dim=-1)[0]

        x = self.conv3(torch.cat([x1, x2], dim=1))

        l = l.view(batch_size, -1, 1)
        l = self.conv_labels(l)

        x = torch.cat([x.mean(dim=-1, keepdim=True), l], dim=1)
        x = x.repeat(1, 1, num_points)
        x = torch.cat([x, x2], dim=1)
        x = self.cls(x)

        return x


class FocalLoss(nn.modules.loss._WeightedLoss):
    # https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


def calculate_sample_iou(sample_gt, sample_pred, label):
    # Based on https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space/blob/master/S3_network_testing.py
    ious = []
    for part_labels in seg_classes[label]:
        tp = np.sum((sample_pred == part_labels) * (sample_gt == part_labels))
        fp = np.sum((sample_pred == part_labels) * (sample_gt != part_labels))
        fn = np.sum((sample_pred != part_labels) * (sample_gt == part_labels))

        ious.append((tp + 1e-12) / (tp + fp + fn + 1e-12))
        return np.nanmean(ious)


def calculate_iou_metrics(shape_ious):
    instance_iou = np.concatenate(shape_ious).mean()
    class_iou = np.nanmean([np.nanmean(_) for _ in shape_ious])
    return instance_iou, class_iou


# #####################################
#       TRAIN-VALIDATE PROCEDURE
# #####################################
def train(
    epochs, k, embeds, lr, momentum, used_pts, weight_decay, gamma, bs, test_freq=5
):
    NUM_CAT = 16
    NUM_CLS = 50
    experiment = Experiment(
        api_key="HqoNNVC2FhBqv4wwsPg3S2nBH",
        project_name="gva-net-segmentation",
        workspace="james1",
        log_code=True,
        disabled=False,
    )
    print("Creating Model...")
    model = GVANet(k=k, out_dim=NUM_CLS, embeds=embeds).to(device)
    print("Done...\n")
    experiment.add_tag("part-seg")
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
            "trainable_params": sum(p.numel() for p in model.parameters()),
        }
    )

    print("Creating DataLoader...")
    train_loader = torch.utils.data.DataLoader(
        ShapeNetPart(partition="trainval", num_points=used_pts),
        batch_size=bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ShapeNetPart(partition="test", num_points=used_pts),
        batch_size=bs,
        shuffle=False,
    )
    print("Done...\n")

    print("Setting-up optimizer, loss function, and scheduler...")
    criterion = FocalLoss(gamma=gamma)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    schedular = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: -0.0000028 * epoch * epoch + 0.7
    )
    print("Done...\n")

    print("Start training...")
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
        shape_ious = [[] for _ in range(NUM_CAT)]
        for data, label, seg in tqdm_batch:
            data = data.permute(0, 2, 1).cuda()
            seg = seg.cuda()
            label_one_hot = np.zeros((label.shape[0], NUM_CAT))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32)).to(data)
            y_hat = model(data, label_one_hot)
            y_hat = y_hat.permute(0, 2, 1).contiguous()
            # Instance-IoU and class-IOU based on https://github.com/qq456cvb/Point-Transformers/blob/master/train_partseg.py
            for yh1, seg1, l1 in zip(
                y_hat.max(-1)[1].cpu().numpy(),
                seg.cpu().numpy(),
                label.cpu().long().squeeze().numpy(),
            ):
                shape_ious[l1].append(calculate_sample_iou(seg1, yh1, l1))
            y_hat_sq = y_hat.view(-1, NUM_CLS)
            seg = seg.view(-1, 1).squeeze()
            loss = criterion(y_hat_sq, seg) + lovasz_softmax(
                F.softmax(y_hat_sq, dim=-1), seg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del data, label
        schedular.step()

        instance_iou, class_iou = calculate_iou_metrics(shape_ious)
        experiment.log_metric("train_class_iou", class_iou, epoch=epoch)
        experiment.log_metric("train_instance_iou", instance_iou, epoch=epoch)
        experiment.log_metric("lr", optimizer.param_groups[0]["lr"], epoch=epoch)
        print(f"train_cls_iou={class_iou}, train_ins_iou={instance_iou}")
        tqdm_batch.close()

        if epoch % test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc=f"Epoch-{epoch} testing")

            model.eval()
            curr_pred = []
            curr_true = []
            shape_ious = [[] for _ in range(NUM_CAT)]

            with torch.no_grad():
                for data, label, seg in tqdm_batch:
                    data = data.permute(0, 2, 1).cuda()
                    seg = seg.cuda()
                    label_one_hot = np.zeros((label.shape[0], NUM_CAT))
                    for idx in range(label.shape[0]):
                        label_one_hot[idx, label[idx]] = 1
                    label_one_hot = torch.from_numpy(
                        label_one_hot.astype(np.float32)
                    ).to(data)
                    y_hat = model(data, label_one_hot)
                    y_hat = y_hat.permute(0, 2, 1).contiguous()
                    # Instance-IoU and class-IOU based on https://github.com/qq456cvb/Point-Transformers/blob/master/train_partseg.py
                    for yh1, seg1, l1 in zip(
                        y_hat.max(-1)[1].cpu().numpy(),
                        seg.cpu().numpy(),
                        label.cpu().long().squeeze().numpy(),
                    ):
                        shape_ious[l1].append(calculate_sample_iou(seg1, yh1, l1))
                    y_hat_sq = y_hat.view(-1, NUM_CLS)
                    seg = seg.view(-1, 1).squeeze()
                    loss = criterion(y_hat_sq, seg) + lovasz_softmax(
                        F.softmax(y_hat_sq, dim=-1), seg
                    )
                    del data, label

            instance_iou, class_iou = calculate_iou_metrics(shape_ious)
            experiment.log_metric("val_class_iou", class_iou, epoch=epoch)
            experiment.log_metric("val_instance_iou", instance_iou, epoch=epoch)
            print(f" val_cls_iou={class_iou}, val_ins_iou={instance_iou}")
            tqdm_batch.close()


if __name__ == "__main__":
    for _ in range(10):
        args = {
            "epochs": 500,
            "k": 32,
            "embeds": 128,
            "lr": 0.1,
            "momentum": 0.9,
            "used_pts": 1024,
            "weight_decay": 0.4e-4,
            "gamma": 1,
            "bs": 32,
            "test_freq": 2,
        }
        train(**args)
