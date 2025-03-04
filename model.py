import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList

from torch_geometric.nn import TransformerConv, AttentionalAggregation
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GATConv

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool 

class GraphTransformerModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, n_heads=1, dropout=0.2):
        super(GraphTransformerModel, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.edge_dim = edge_dim

        self.conv_layers = ModuleList([
            TransformerConv(
                in_channels if i == 0 else self.embedding_size,  
                self.embedding_size,
                heads=self.n_heads,
                dropout=self.dropout_rate,
                edge_dim=self.edge_dim,
                beta=True
            ) for i in range(self.num_layers)
        ])

        self.transf_layers = ModuleList([
            Linear(self.embedding_size * self.n_heads, self.embedding_size) for _ in range(self.num_layers)
        ])
        self.bn_layers = ModuleList([
            BatchNorm1d(self.embedding_size) for _ in range(self.num_layers)
        ])

        self.attention_pool = AttentionalAggregation(gate_nn=Linear(self.embedding_size, 1))
        self.linears = ModuleList([
            Linear(self.embedding_size, hidden_channels // 2),
            Linear(hidden_channels // 2, out_channels)
        ])

    def forward(self, x, edge_index, edge_attr, batch_index):
        local_representation = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            local_representation.append(x)

        x = sum(local_representation)
        x = self.attention_pool(x, batch_index)

        for linear in self.linears[:-1]:
            x = torch.relu(linear(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linears[-1](x)  

        return x

class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, dropout=0.2):
        super(GINModel, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.edge_dim = edge_dim

        self.conv_layers = ModuleList([GINEConv(Linear(in_channels, self.embedding_size), edge_dim=self.edge_dim)])

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GINEConv(Linear(self.embedding_size, self.embedding_size), edge_dim=self.edge_dim))

        self.linear1 = Linear(self.embedding_size * 2, hidden_channels // 2)
        self.linear2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)  
            x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear2(x) 
        return x

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, dropout=0.2):
        super(GCNModel, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.conv_layers = ModuleList([GCNConv(in_channels, self.embedding_size)])

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GCNConv(self.embedding_size, self.embedding_size))

        self.edge_linear = Linear(edge_dim, self.embedding_size)

        self.linear1 = Linear(self.embedding_size * 2, hidden_channels // 2)
        self.linear2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_features = self.edge_linear(edge_attr)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = x + torch.mean(edge_features, dim=0)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)


        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear2(x) 
        return x


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, dropout=0.2):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.edge_dim = edge_dim

        self.conv_layers = ModuleList([GATConv(in_channels, self.embedding_size, edge_dim=self.edge_dim)])

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GATConv(self.embedding_size, self.embedding_size, edge_dim=self.edge_dim))

        self.linear1 = Linear(self.embedding_size * 2, hidden_channels // 2)
        self.linear2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)  
            x = F.relu(x)

        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear2(x)  
        return x