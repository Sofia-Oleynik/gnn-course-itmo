"""submission.py — реализация слоёв для ДЗ 1."""
import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree


class MyGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index):
        # В GCN из PyG по умолчанию добавляются self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        
        # Степени (считаются по целевой вершине)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Нормировка 1/sqrt(d_u * d_v)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Линейное преобразование
        x = self.lin(x)
        
        # Агрегация сообщений
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
        return out


class MySAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index):
        row, col = edge_index
        n = x.size(0)
        
        # Считаем количество соседей
        cnt = torch.zeros(n, device=x.device, dtype=x.dtype)
        cnt.index_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        
        # Суммируем соседей
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row])
        
        # Усредняем (защита от деления на ноль)
        agg = agg / cnt.clamp(min=1).unsqueeze(-1)
        
        # Свой вектор + агрегат соседей
        return self.lin_self(x) + self.lin_neigh(agg)
