"""Direction-aware Actor-Critic used only by the A3C training entrypoint."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import to_dense_batch
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "A3C-GCN training requires torch and torch_geometric. "
        "Install builds compatible with the selected Python/CUDA runtime."
    ) from exc


class DirectionalA3CGCNActor(nn.Module):
    def __init__(self, input_dim=11, embedding_dim=64):
        super().__init__()
        self.in_conv_1 = GCNConv(input_dim, embedding_dim)
        self.in_conv_2 = GCNConv(embedding_dim, embedding_dim)
        self.out_conv_1 = GCNConv(input_dim, embedding_dim)
        self.out_conv_2 = GCNConv(embedding_dim, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, data):
        edge_index = data.edge_index
        reverse_edge_index = edge_index.flip(0)
        incoming = F.leaky_relu(self.in_conv_1(data.x, edge_index))
        incoming = F.leaky_relu(self.in_conv_2(incoming, edge_index))
        outgoing = F.leaky_relu(self.out_conv_1(data.x, reverse_edge_index))
        outgoing = F.leaky_relu(self.out_conv_2(outgoing, reverse_edge_index))
        logits = self.head(torch.cat((incoming, outgoing), dim=-1)).squeeze(-1)
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 1
        return logits.reshape(batch_size, -1)


class DirectionalA3CGCNCritic(nn.Module):
    def __init__(self, input_dim=11, embedding_dim=64):
        super().__init__()
        self.conv_1 = GCNConv(input_dim, embedding_dim)
        self.conv_2 = GCNConv(embedding_dim, embedding_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, data):
        x = F.leaky_relu(self.conv_1(data.x, data.edge_index))
        x = F.leaky_relu(self.conv_2(x, data.edge_index))
        dense, mask = to_dense_batch(x, data.batch)
        graph_embedding = (dense * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return self.value_head(graph_embedding)


class DirectionalActorCritic(nn.Module):
    def __init__(self, input_dim=11, embedding_dim=64):
        super().__init__()
        self.actor = DirectionalA3CGCNActor(input_dim, embedding_dim)
        self.critic = DirectionalA3CGCNCritic(input_dim, embedding_dim)

    def act(self, data):
        return self.actor(data)

    def evaluate(self, data):
        return self.critic(data)
