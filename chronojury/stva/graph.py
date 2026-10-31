
"""
Spatio-Temporal Verdict Graph Module for ChronoJury.
Implements Graph Attention Networks (GAT) for Intra-Stage Inter-Juror Interaction (ISJI).
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DebateGraph(nn.Module):
    """
    Intra-Stage Inter-Juror Interaction (ISJI) Sub-module via Graph Attention Networks.
    Computes spatial opinion attention weights alpha_{ij}^t and aggregates context-aware
    juror representations within each deliberation stage snapshot G_t^{STVA} (Section 3.2).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, num_heads: int, dropout: float = 0.0):
        super(DebateGraph, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.dropout = dropout

        # Single-layer GAT configuration (e.g., STVA mode)
        if num_layers == 1:
            self.gat_layers.append(
                GATConv(input_dim, output_dim, heads=num_heads, concat=False, add_self_loops=True, dropout=dropout)
            )
        else:
            self.gat_layers.append(
                GATConv(input_dim, hidden_dim, heads=num_heads, concat=True, add_self_loops=True, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, add_self_loops=True, dropout=dropout)
                )
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, add_self_loops=True, dropout=dropout)
            )

    def forward(self, x, edge_index, return_edge_weights: bool = False):
        """
        Forward pass for spatial graph attention aggregation.
        :param x: Initial stage node features matrix [N_nodes, Input_dim]
        :param edge_index: Graph topology edge indices [2, Num_edges]
        :param return_edge_weights: Whether to extract spatial attention weights alpha_{ij}^t for interpretability
        :return: Updated node representations and optional edge attention scalar tensor
        """
        # Pass through intermediate GAT layers
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)  

        # Final GAT layer for output representation
        last_layer = self.gat_layers[-1]
        
        if return_edge_weights:

            out = last_layer(x, edge_index, return_attention_weights=True)
            if isinstance(out, tuple) and len(out) == 2:
                x_out, (edge_idx_out, attn) = out

                if attn is not None and attn.dim() == 2:
                    attn_scalar = attn.mean(dim=1)
                else:
                    attn_scalar = attn

                return F.elu(x_out), edge_idx_out, attn_scalar
            else:

                return F.elu(out), edge_index, None
        else:
            x = last_layer(x, edge_index)
            return F.elu(x)
