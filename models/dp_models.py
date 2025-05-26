import torch
from dgl.nn.pytorch.conv import GraphConv, GATConv
from torch_geometric.nn import GraphNorm, global_mean_pool, global_max_pool
from torch.nn import ReLU, GELU
import dgl
import dgl.function as fn
import torch.nn.functional as F
import torch_geometric.nn as nn

class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, end_channels, output_channels, n_sequences, device, task_type):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences, hidden_dim, device)
        self.layer3 = MLPLayer(hidden_dim, hidden_dim, device)
        self.layer4 = MLPLayer(hidden_dim, end_channels, device)
        self.layer2 = MLPLayer(end_channels, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, features, edges=None):
        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer2(x)
        if self.task_type == 'classification':
            x = self.s
        return x

class Sep_GRU_GNN(torch.nn.Module):
    def __init__(
        self,
        gru_hidden=64,
        gnn_hidden_list=[32, 64],
        lin_channels=64,
        end_channels=64,
        out_channels=1,
        n_sequences=1,
        task_type='classification',
        device=None,
        act_func='relu',
        static_idx=None,
        temporal_idx=None,
        num_lstm_layers=1,
        use_layernorm=False,
        dropout=0.03,
    ):
        super(Sep_GRU_GNN, self).__init__()

        self.gru_hidden = gru_hidden
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.is_graph_or_node = False
        self.device = device

        # LSTM
        input_size = len(temporal_idx)
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Multi-layer GCN
        self.gnn_layers = torch.nn.ModuleList()
        in_feats = len(static_idx)
        for out_feats in gnn_hidden_list:
            self.gnn_layers.append(GraphConv(in_feats, out_feats))
            in_feats = out_feats  # pour la prochaine couche

        self.gnn_output_dim = gnn_hidden_list[-1]

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        # Output linear layer
        print(f'Spatial {in_feats}')
        print(f'Temporal {input_size}')
        print(f'Sum {gru_hidden} + {self.gnn_output_dim}')
        self.linear1 = torch.nn.Linear(gru_hidden + self.gnn_output_dim, lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_hidden + self.gnn_output_dim).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_hidden + self.gnn_output_dim).to(device)
            
        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def separate_variables(self, x):
        # x: (B, X, T)
        is_static = (x == x[:, :, 0:1]).all(dim=2)
        static_mask = is_static.all(dim=0)
        static_idx = torch.where(static_mask)[0]
        temporal_idx = torch.where(~static_mask)[0]

        x_static = x[:, static_idx, 0].unsqueeze(-1)  # (B, S, 1)
        x_temporal = x[:, temporal_idx, :]            # (B, D, T)

        return x_static, x_temporal, static_idx, temporal_idx
    
    def forward(self, x, graph):
        # x: (B, X, T)
        B, X, T = x.shape

        # Séparation statique/temporelle
        if self.static_idx is None:
            x_static, x_temporal, static_idx, temporal_idx = self.separate_variables(x)
        else:
            x_static = x[:, self.static_idx, 0].unsqueeze(-1)  # (B, S, 1)
            x_temporal = x[:, self.temporal_idx, :]            # (B, D, T)

        # --- LSTM ---
        D = x_temporal.shape[1]
        if D == 0:
            lstm_out = torch.zeros(B, self.lstm_hidden, device=x.device)
        else:
            x_lstm_input = x_temporal.permute(0, 2, 1)  # (B, T, D)
            lstm_out, _ = self.gru(x_lstm_input)       # (B, T, H)
            lstm_out = lstm_out[:, -1, :]               # (B, H)
            
        # --- GCN ---
        S = x_static.shape[1]
        if S == 0:
            gnn_out = torch.zeros(B, self.gnn_output_dim, device=x.device)
        else:
            h = x_static.squeeze(-1)  # (B, S)
            for layer in self.gnn_layers:
                h = layer(graph, h)
                h = torch.relu(h)
            gnn_out = h               # (B, out_dim)

        # --- Fusion ---
        x = torch.cat([lstm_out, gnn_out], dim=1)  # (B, total)
        x = self.dropout(x)
        x = self.norm(x)
        
        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        return output

class GRU(torch.nn.Module):
    def __init__(self, in_channels, gru_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.0, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False):
        super(GRU, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.gru_size = gru_size
        
        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_size).to(device)

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        self.linear1 = torch.nn.Linear(gru_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, X, edge_index=None, graphs=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        # Reshape to (batch, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.gru_size).to(self.device)

        # GRU forward
        x, _ = self.gru(x, h0)

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, x
        else:
            return output

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, lstm_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.03, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False):
        super(LSTM, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.lstm_size = lstm_size

        # LSTM block
        self.lstm = torch.nn.LSTM(
            input_size=in_channels,
            hidden_size=self.lstm_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(self.lstm_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(self.lstm_size).to(device)

        # Dropout after LSTM
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output layer
        self.linear1 = torch.nn.Linear(self.lstm_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def forward(self, X, edge_index=None, graphs=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        # (batch_size, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)

        # LSTM forward
        x, _ = self.lstm(x, (h0, c0))

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output layer
        self.linear1 = torch.nn.Linear(channels[-1], lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()
        
        self.return_hidden = False 

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, x, edges=None):
        # Couche d'entrée

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
class GraphCast(torch.nn.Module):
    def __init__(self,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels = 64,
        lin_channels = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        out_channels = 4,
        task_type = 'classification',
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences = 1,
        act_func='ReLU',
        is_graph_or_node=False):
        super(GraphCast, self).__init__()

        # See graphCast architecture in https://github.com/SeasFire/firecastnet/tree/main/seasfire/backbones/graphcast
        
        self.net = GraphCastNet(
            input_dim_grid_nodes,
            input_dim_mesh_nodes,
            input_dim_edges,
            output_dim_grid_nodes,
            processor_layers,
            hidden_layers,
            hidden_dim,
            aggregation,
            norm_type,
            do_concat_trick,
            has_time_dim)
        
        # Output layer
        self.linear1 = torch.nn.Linear(output_dim_grid_nodes, lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)
        
        self.is_graph_or_node = is_graph_or_node == 'graph'
        
        self.act_func = getattr(torch.nn, act_func)()
        
        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def forward(self, X, graph, graph2mesh, mesh2graph):
        #X = X.view(X.shape[0], -1)
        #print(X.device)
        #print(X.shape)
        X = X.permute(2, 0, 1)
        x = self.net(X, graph, graph2mesh, mesh2graph)[-1]
        
        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        return output