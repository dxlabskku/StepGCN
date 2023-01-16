import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchmetrics import F1Score


class ResBlock(nn.Module):
    def __init__(self, nfeat, dropout, type):
        super(ResBlock, self).__init__()

        self.type = type
        self.dropout = dropout
        self.conv = GCNConv(nfeat, nfeat)
        self.conv.lin.weight = nn.Parameter(torch.zeros(nfeat, nfeat))
        if self.type == "linear":
            self.res = nn.Linear(nfeat, nfeat, bias=False)
            self.res.weight = nn.Parameter(torch.eye(nfeat))
        elif self.type == "graph":
            self.res = GCNConv(nfeat, nfeat, bias=False)
            self.res.lin.weight = nn.Parameter(torch.eye(nfeat), requires_grad=False)
        elif self.type == "graph_linear":
            self.res = GCNConv(nfeat, nfeat, bias=False)
            self.res.lin.weight = nn.Parameter(torch.eye(nfeat))
    
    def forward(self, x, edge_index):
        if self.type == "none":
            x = F.relu(self.conv(x, edge_index)) + x
        elif self.type == "linear":
            x = F.relu(self.conv(x, edge_index)) + self.res(x)
        else:
            x = F.relu(self.conv(x, edge_index)) + self.res(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class StepModel(nn.Module):
    def __init__(self, pre_model, nfeat, dropout, type):
        super(StepModel, self).__init__()

        self.pre_model = pre_model
        self.resblock = ResBlock(nfeat, dropout, type)
        self.f1_score = self.pre_model.f1_score

    def forward(self, x, edge_index):
        x = self.resblock(x, edge_index)
        x = self.pre_model(x, edge_index)
        return x

    def update(self, data, epochs, learning_rate, weight_decay, patience):
        x, y, edge_index = data.x, data.y, data.edge_index
        best_f1, p_cnt = 0, 0
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(epochs):
            t = time.time()

            self.train()
            optimizer.zero_grad()
            pred = self.forward(x, edge_index)
            loss_train = F.nll_loss(pred[data.train_mask], y[data.train_mask])
            f1_train = self.f1_score(pred[data.train_mask], y[data.train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            pred = self.forward(x, edge_index)
            loss_val = F.nll_loss(pred[data.val_mask], y[data.val_mask])
            f1_val = self.f1_score(pred[data.val_mask], y[data.val_mask])

            print(f"Epoch: {epoch:04d}",
                f"loss_train: {loss_train.item():.4f}",
                f"f1_train: {f1_train.item():.4f}",
                f"loss_val: {loss_val.item():.4f}",
                f"f1_val: {f1_val.item():.4f}",
                f"time: {time.time() - t:.4f}s")

            if f1_val > best_f1:
                best_f1 = f1_val
                p_cnt = 0
            else:
                p_cnt += 1
            if p_cnt >= patience:
                break

    def test(self, data):
        x, y, edge_index = data.x, data.y, data.edge_index
        self.eval()
        pred = self.forward(x, edge_index)
        loss_test = F.nll_loss(pred[data.test_mask], y[data.test_mask])
        f1_test = self.f1_score(pred[data.test_mask], y[data.test_mask])

        print(f"Test set results:",
            f"loss: {loss_test.item():.4f}",
            f"f1: {f1_test.item():.4f}")