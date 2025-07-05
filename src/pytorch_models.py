import torch
from torch.utils.data import Dataset
from torch import nn


class TabularDataset(Dataset):
    """
    Dataset personalizado para carregar dados tabulares para o PyTorch.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    """
    Define uma arquitetura de rede neural mais profunda e estável,
    utilizando Batch Normalization para melhorar a convergência.
    """

    # --- CORREÇÃO: A assinatura do __init__ foi atualizada ---
    def __init__(self, input_size: int, hidden_size_1: int, hidden_size_2: int, dropout_rate: float):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            # Bloco 1
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),  # Normaliza as saídas da camada linear
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Bloco 2
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Camada de Saída (produz logits)
            nn.Linear(hidden_size_2, 1)
        )

    def forward(self, x):
        """Define a passagem para a frente (forward pass)."""
        return self.layers(x)
