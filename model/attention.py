import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Wk=nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wq=nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wv=nn.Linear(embedding_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        query = self.Wq(embedded)
        key = self.Wk(embedded)
        value = self.Wv(embedded)

        # Attention
        scores = (query@key.transpose(-2, -1))/(query.shape[-1]**0.5)
        lower_matrix = torch.tril(torch.ones_like(scores))
        mask = lower_matrix==0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = self.softmax(scores)
        emb = scores@value
        return torch.round(emb, decimals=4)

