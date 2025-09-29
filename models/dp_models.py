import torch
    
class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, end_channels, output_channels, n_sequences, device, task_type, return_hidden=False):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences, hidden_dim[0], device)
        self.layer3 = MLPLayer(hidden_dim[0], hidden_dim[1], device)
        self.layer4 = MLPLayer(hidden_dim[1], end_channels, device)
        self.layer2 = MLPLayer(end_channels, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden

    def forward(self, features, edges=None):
        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        hidden = x
        logits = self.layer2(x)
        if self.task_type == 'classification':
            output = self.soft(logits)
        else:
            output = logits
            
        return output, logits, hidden

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
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

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
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False, return_hidden=False):
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
        
        self.return_hidden = return_hidden

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
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

class GraphCastGRU(torch.nn.Module):
    def __init__(
        self,
        *,
        # --- GRU specific parameters ---
        in_channels: int = 16,
        num_gru_layers: int = 1,
        # --- GraphCast parameters (unchanged) ---
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels: int = 64,
        lin_channels: int = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "BatchNorm",
        out_channels: int = 4,
        task_type: str = "classification",
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences: int = 1,
        act_func: str = "ReLU",
        is_graph_or_node: bool = False,
        return_hidden: bool = False,
    ):
        super().__init__()

        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=input_dim_grid_nodes,
            num_layers=num_gru_layers,
            dropout=0.03 if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gru_size = input_dim_grid_nodes
        self.num_gru_layers = num_gru_layers
        self.norm = torch.nn.BatchNorm1d(self.gru_size)
        self.dropout = torch.nn.Dropout(0.03)
        
        self.net = GraphCastNet( #https://github.com/seasfire/firecastnet
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
            has_time_dim,
        )

        self.linear1 = torch.nn.Linear(output_dim_grid_nodes, lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)

        self.is_graph_or_node = is_graph_or_node == "graph"

        self.act_func = getattr(torch.nn, act_func)()
        self.return_hidden = return_hidden

        if task_type == "classification":
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == "binary":
            self.output_activation = torch.nn.Sigmoid()
        else:  # regression or custom
            self.output_activation = torch.nn.Identity()

    def forward(self, X, graph, graph2mesh, mesh2graph):
        # Bring node dimension next to batch for GRU: (batch * n_nodes, seq_len, in_channels)
        B, C_in, T = X.shape
        X_for_gru = X.permute(0, 2, 1)
        h0 = torch.zeros(self.num_gru_layers, B, self.gru_size).to(X.device)

        gru_out, _ = self.gru(X_for_gru, h0)  # shape: (B*N, T, hidden)
        # Keep the last hidden state for each sequence
        gru_last = self.norm(gru_out[:, -1, :])
        gru_last = self.dropout(gru_last)  # (B*N, hidden == input_dim_grid_nodes)
        
        X_graphcast = gru_last[None,: ,:]

        # GraphCast processing
        x = self.net(X_graphcast, graph, graph2mesh, mesh2graph)[-1]

        # Head
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
