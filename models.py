import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    1D CNN encoder for tokenized sequences (e.g., SMILES, amino acids).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # output dim = num_filters * len(kernel_sizes)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) integer token IDs
        return: (batch_size, output_dim)
        """
        # (B, L) -> (B, L, E)
        emb = self.embedding(x)
        # (B, L, E) -> (B, E, L) for Conv1d
        emb = emb.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            # (B, F, L') after conv
            c = conv(emb)
            # global max pooling over sequence length
            c = F.relu(c)
            c = F.max_pool1d(c, kernel_size=c.shape[-1])  # (B, F, 1)
            conv_outputs.append(c.squeeze(-1))  # (B, F)

        # concat over filters
        out = torch.cat(conv_outputs, dim=1)  # (B, F * n_kernels)
        out = self.dropout(out)
        return out


class SequencePairRegressor(nn.Module):
    """
    Simple DeepDTA-style model:
    two CNN encoders (e.g., ligand & protein), then MLP regressor.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes=(3, 5, 7),
        hidden_dims=(256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.drug_encoder = CNNEncoder(
            vocab_size=drug_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.target_encoder = CNNEncoder(
            vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        input_dim = self.drug_encoder.output_dim + self.target_encoder.output_dim

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        # final regression head
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, drug_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        d = self.drug_encoder(drug_ids)
        t = self.target_encoder(target_ids)
        x = torch.cat([d, t], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)  # (batch_size,)
