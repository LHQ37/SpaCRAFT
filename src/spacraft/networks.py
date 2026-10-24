import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GATlayer(nn.Module):

    def __init__(self, input_dim, output_dim, is_last=False,
                 attention_dropout=0.0):
        super().__init__()

        self.is_last = is_last
        self.attention_dropout = float(attention_dropout)

        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
        if not self.is_last:
            self.v0 = nn.Parameter(torch.Tensor(output_dim, 1))
            self.v1 = nn.Parameter(torch.Tensor(output_dim, 1))
            nn.init.xavier_uniform_(self.v0)
            nn.init.xavier_uniform_(self.v1)

        self.alphas = None

    def forward(self, node_features, graph, tied_alphas=None):

        X = torch.matmul(node_features, self.W)

        if not self.is_last:
            # compute attention coefficients
            if tied_alphas is not None:
                alphas = tied_alphas
            else:
                f0 = torch.matmul(X, self.v0)
                f1 = torch.matmul(X, self.v1)
                E = graph * (f0 + f1.T)
                alphas = (torch.sigmoid(E) - 0.5).to_sparse()
                alphas = torch.sparse.softmax(alphas, dim=1)
                alphas = alphas.to_dense()
                if self.attention_dropout:
                    alphas = F.dropout(
                        alphas, self.attention_dropout, self.training)
                self.alphas = alphas

            X = torch.matmul(alphas, X)
            X = F.elu(X, alpha=1.0)

        return X


class GATEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim, graph_neigh=None,
                 feature_dropout=0.0, attention_dropout=0.0,
                 normalization="none"):
        
        super().__init__()
        
        # self.enable_contrastive = enable_contrastive
        self.graph_neigh = graph_neigh
        self.feature_dropout = float(feature_dropout)
        self.normalization = normalization.lower()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = GATlayer(
                input_dim, hidden_dims[0],
                attention_dropout=attention_dropout)
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = GATlayer(
                        hidden_dims[i-1], hidden_dims[i],
                        attention_dropout=attention_dropout)
                    self.layer_list.append(layer)

            layer = GATlayer(hidden_dims[-1], latent_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = GATlayer(input_dim, latent_dim, is_last=True)
            self.layer_list.append(layer)
        dimensions = list(hidden_dims) + [latent_dim]
        self.normalization_list = nn.ModuleList([
            nn.LayerNorm(value) if self.normalization == "layernorm"
            else nn.Identity() for value in dimensions])
        
    def forward(self, node_features, graph):
        """
        参数:
            node_features: 原始节点特征 [n_nodes, input_dim]
            graph: 图邻接矩阵 [n_nodes, n_nodes]
        返回:
            Z: 原始图的潜在表示 [n_nodes, latent_dim]

        说明:
            最终模型中的 enable_contrastive 开关只在 SpaCRAFTModel 中控制
            是否计算 InfoNCE 风格的簇级对比损失。GATEncoder 本身只负责
            输出节点嵌入，不再包含 GraphST 风格的实例级对比分支。
        """
        Z = (
            F.dropout(node_features, self.feature_dropout, self.training)
            if self.feature_dropout else node_features)
        Z = self.layer_list[0](Z, graph)
        Z = self.normalization_list[0](Z)
        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                if self.feature_dropout:
                    Z = F.dropout(Z, self.feature_dropout, self.training)
                Z = self.layer_list[i](Z, graph)
                Z = self.normalization_list[i](Z)
        return Z


class Discriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim_list):
        super().__init__()

        self.num_layer = len(hidden_dim_list) + 1

        if self.num_layer != 1:
            self.linear_list = nn.ModuleList()

            linear = nn.Linear(input_dim, hidden_dim_list[0])
            nn.init.xavier_uniform_(linear.weight)
            self.linear_list.append(linear)

            self.src_layer = nn.Linear(hidden_dim_list[-1], 1)
            nn.init.xavier_uniform_(self.src_layer.weight)

        else:
            self.src_layer = nn.Linear(input_dim, 1)
            nn.init.xavier_uniform_(self.src_layer.weight)

        if self.num_layer >= 3:
            for i in range(1, len(hidden_dim_list)):
                linear = nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i])
                nn.init.xavier_uniform_(linear.weight)
                self.linear_list.append(linear)

    def forward(self, X):
        if self.num_layer != 1:
            h = self.linear_list[0](X)
            h = torch.softmax(h, dim=1)

            if self.num_layer >= 3:
                for i in range(1, self.num_layer - 1):
                    h = self.linear_list[i](h)
                    h = torch.softmax(h, dim=1)

            src = self.src_layer(h)

        else:
            src = self.src_layer(X)
        return src



class MLPSingleLayer(nn.Module):

    def __init__(self, input_dim, output_dim, is_last=False):
        super().__init__()

        self.is_last = is_last

        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, node_features):

        X = self.linear(node_features)

        if not self.is_last:

            X = F.elu(X, alpha=1.0)

        return X


class MLPEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = MLPSingleLayer(input_dim, hidden_dims[0])
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = MLPSingleLayer(hidden_dims[i - 1], hidden_dims[i])
                    self.layer_list.append(layer)

            layer = MLPSingleLayer(hidden_dims[-1], latent_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = MLPSingleLayer(input_dim, latent_dim, is_last=True)
            self.layer_list.append(layer)

    def forward(self, node_features):

        Z = self.layer_list[0](node_features)

        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                Z = self.layer_list[i](Z)

        return Z


class MLPDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()

        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()

        if self.num_layer != 1:

            layer = MLPSingleLayer(latent_dim, hidden_dims[0])
            self.layer_list.append(layer)

            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    layer = MLPSingleLayer(hidden_dims[i - 1], hidden_dims[i])
                    self.layer_list.append(layer)

            layer = MLPSingleLayer(hidden_dims[-1], output_dim, is_last=True)
            self.layer_list.append(layer)

        else:
            layer = MLPSingleLayer(latent_dim, output_dim, is_last=True)
            self.layer_list.append(layer)

    def forward(self, node_features):

        Y = self.layer_list[0](node_features)

        if self.num_layer > 1:
            for i in range(1, len(self.layer_list)):
                Y = self.layer_list[i](Y)

        return Y
