import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchmetrics import F1Score


class ResGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth, dropout, type):
        super(ResGCN, self).__init__()

        self.depth = depth
        self.dropout = dropout
        self.type = type
        self.convs = nn.ModuleList()
        self.res = nn.ModuleList()
        self.f1_score = F1Score(task="multiclass", num_classes=nclass, average="micro")

        if self.depth == 1:
            self.convs.append(GCNConv(nfeat, nclass))
            self.res.append(self.residual(nfeat, nclass))
        
        elif "none" in self.type:
            self.convs.append(GCNConv(nfeat, nhid))
            for _ in range(self.depth - 2):
                self.convs.append(GCNConv(nhid, nhid))
            self.convs.append(GCNConv(nhid, nclass))
        
        elif "seq" in self.type:
            self.convs.append(GCNConv(nfeat, nhid))
            self.res.append(self.residual(nfeat, nhid))
            for _ in range(self.depth - 2):
                self.convs.append(GCNConv(nhid, nhid))
            self.convs.append(GCNConv(nhid, nclass))
            self.res.append(self.residual(nhid, nclass))
        
        elif "lin" in self.type:
            self.convs.append(GCNConv(nfeat, nhid))
            self.res.append(self.residual(nfeat, nhid))
            for _ in range(self.depth - 2):
                self.convs.append(GCNConv(nhid, nhid))
                self.res.append(self.residual(nhid, nhid))
            self.convs.append(GCNConv(nhid, nclass))
            self.res.append(self.residual(nhid, nclass))
        
        elif "div" in self.type:
            self.convs.append(GCNConv(nfeat, nhid))
            self.res.append(self.residual(nfeat, nhid))
            for _ in range(self.depth - 2):
                self.convs.append(GCNConv(nhid, nhid))
                self.res.append(self.residual(nfeat, nhid))
            self.convs.append(GCNConv(nhid, nclass))
            self.res.append(self.residual(nfeat, nclass))

        elif "conv" in self.type:
            self.convs.append(GCNConv(nfeat, nhid))
            self.res.append(self.residual(nfeat, nclass))
            for _ in range(self.depth - 2):
                self.convs.append(GCNConv(nhid, nhid))
                self.res.append(self.residual(nhid, nclass))
            self.convs.append(GCNConv(nhid, nclass))
            self.res.append(self.residual(nhid, nclass))

    def residual(self, input, output):
        return GCNConv(input, output) if "graph" in self.type else nn.Linear(input, output)

    def forward(self, x, edge_index):
        if self.depth == 1 and "none" in self.type:
            x = self.convs[0](x, edge_index)
            return F.log_softmax(x, dim=1)
        elif self.depth == 1:
            x = self.convs[0](x, edge_index) + self.forward_residual(0, x, edge_index)
            return F.log_softmax(x, dim=1)
        elif "none" in self.type:
            return self.forward_none(x, edge_index)
        elif "seq" in self.type:
            return self.forward_seq(x, edge_index)
        elif "lin" in self.type:
            return self.forward_lin(x, edge_index)
        elif "div" in self.type:
            return self.forward_div(x, edge_index)
        elif "conv" in self.type:
            return self.forward_conv(x, edge_index)

    def forward_none(self, x, edge_index):
        for i in range(self.depth - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_seq(self, x, edge_index):
        x = self.convs[0](x, edge_index) + self.forward_residual(0, x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(1, self.depth - 1):
            x = self.convs[i](x, edge_index) + x
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) + self.forward_residual(-1, x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_lin(self, x, edge_index):
        x = self.convs[0](x, edge_index) + self.forward_residual(0, x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(1, self.depth - 1):
            x = self.convs[i](x, edge_index) + self.forward_residual(i, x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) + self.forward_residual(-1, x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_div(self, x, edge_index):
        raw = x
        x = self.convs[0](x, edge_index) + self.forward_residual(0, raw, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(1, self.depth - 1):
            x = self.convs[i](x, edge_index) + self.forward_residual(i, raw, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) + self.forward_residual(-1, raw, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_conv(self, x, edge_index):
        x_list = [self.forward_residual(0, x, edge_index)]
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(1, self.depth - 1):
            x_list.append(self.forward_residual(i, x, edge_index))
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x_list.append(self.forward_residual(-1, x, edge_index))
        x = self.convs[-1](x, edge_index)
        x = x + sum(x_list)
        return F.log_softmax(x, dim=1)

    def forward_residual(self, i, x, edge_index):
        return self.res[i](x, edge_index) if "graph" in self.type else self.res[i](x)

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