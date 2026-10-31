"""Explanation-only gates on the trained temporal attention messages.

The original temporal module still adds its positional encodings. Only its
Transformer encoder's forward method is temporarily replaced, using the same
parameters, residual connections, norms, and feed-forward layers. Gates multiply
attention probabilities *after* softmax without renormalization. They explain
same-node, past-to-later-stage messages, not additional GAT edges.
"""

import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F


_MISSING = object()
_ACTIVE_CONTEXTS = weakref.WeakKeyDictionary()


def iter_temporal_edges(*, layers, nodes, heads, stages):
    """Yield strict-past edges in flattened-mask order, using zero-based indices.

    ``weight_index`` indexes [layer, node, head, target_stage, source_stage].
    A temporal edge always preserves the node identity. Diagonal and future
    positions are fixed and therefore are not explanation edges.
    """
    if any(not isinstance(value, int) or value < 1
           for value in (layers, nodes, heads, stages)):
        raise ValueError('Temporal edge dimensions must be positive integers')
    edge_id = 0
    for layer in range(layers):
        for node in range(nodes):
            for head in range(heads):
                for target in range(1, stages):
                    for source in range(target):
                        yield {
                            'edge_id': edge_id, 'layer': layer, 'node': node,
                            'head': head, 'source_stage': source,
                            'target_stage': target,
                            'weight_index': [layer, node, head, target, source],
                        }
                        edge_id += 1


class TemporalMaskContext:
    """Temporarily expose temporal attention messages to differentiable gates.

    Usage::

        with TemporalMaskContext(temporal, stages=3, nodes=7) as context:
            context.set_weights(gates)  # [layers, nodes, heads, target, source]
            result = temporal(node_sequences)

    ``weights=None`` opens all allowed connections. Only entries where
    ``source < target`` are used; diagonal gates are always one and future
    gates zero. Supplying a differentiable tensor preserves its autograd graph.
    The caller freezes model parameters and puts the model in evaluation mode.
    No parameters, buffers, training flags, or state-dict entries are changed.

    Each context explains one case, so the input batch dimension must equal
    ``nodes``. Concurrent or nested contexts on the same encoder are rejected.
    """

    def __init__(self, temporal, stages, nodes):
        if any(not isinstance(value, int) or value < 1 for value in (stages, nodes)):
            raise ValueError('stages and nodes must be positive integers')
        encoder = getattr(temporal, 'transformer', None)
        if not isinstance(encoder, nn.TransformerEncoder) or not encoder.layers:
            raise TypeError('Expected a temporal module with a TransformerEncoder')
        self.temporal = temporal
        self.encoder = encoder
        self.stages = stages
        self.nodes = nodes
        self.num_layers = len(encoder.layers)
        self.num_heads = encoder.layers[0].self_attn.num_heads
        for layer in encoder.layers:
            attention = layer.self_attn
            if (not isinstance(layer, nn.TransformerEncoderLayer)
                    or not attention.batch_first
                    or not attention._qkv_same_embed_dim
                    or attention.in_proj_weight is None
                    or attention.bias_k is not None or attention.bias_v is not None
                    or attention.add_zero_attn
                    or attention.num_heads != self.num_heads):
                raise ValueError('Unsupported temporal Transformer attention configuration')
        positional = getattr(temporal, 'position_embeddings', None)
        if positional is not None and stages > positional.shape[1]:
            raise ValueError('Stage count exceeds the trained positional capacity')
        self.mask_shape = (self.num_layers, nodes, self.num_heads, stages, stages)
        self._past = torch.ones(stages, stages, dtype=torch.bool).tril(diagonal=-1)
        self._weights = None
        self._active = False
        self._previous_forward = _MISSING

    @property
    def allowed_mask(self):
        """CPU boolean selector for the learnable strict-past entries."""
        return self._past.expand(self.mask_shape)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self.set_weights(weights)

    def set_weights(self, weights):
        """Set probability gates without copying or detaching their gradients."""
        if weights is not None:
            if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != self.mask_shape:
                raise ValueError(f'Temporal weights must have shape {self.mask_shape}')
            if not weights.is_floating_point():
                raise TypeError('Temporal weights must be floating-point probabilities')
            if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
                raise ValueError('Temporal weights must be finite probabilities in [0, 1]')
        self._weights = weights
        return self

    def iter_edges(self):
        return iter_temporal_edges(layers=self.num_layers, nodes=self.nodes,
                                   heads=self.num_heads, stages=self.stages)

    def __enter__(self):
        if self._active or self.encoder in _ACTIVE_CONTEXTS:
            raise RuntimeError('A temporal mask context is already active on this encoder')
        if any(module.training for module in self.temporal.modules()):
            raise ValueError('Temporal explanations require the model in evaluation mode')
        self._previous_forward = self.encoder.__dict__.get('forward', _MISSING)
        self.encoder.forward = self._forward
        _ACTIVE_CONTEXTS[self.encoder] = self
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._active:
            try:
                if self._previous_forward is _MISSING:
                    del self.encoder.forward
                else:
                    self.encoder.forward = self._previous_forward
            finally:
                _ACTIVE_CONTEXTS.pop(self.encoder, None)
                self._active = False
                self._previous_forward = _MISSING
        return False

    def _attention(self, x, layer, layer_index, mask):
        attention = layer.self_attn
        nodes, stages, width = x.shape
        heads = attention.num_heads
        head_width = width // heads
        query, key, value = F.linear(x, attention.in_proj_weight,
                                    attention.in_proj_bias).chunk(3, dim=-1)
        query, key, value = (item.reshape(nodes, stages, heads, head_width).transpose(1, 2)
                             for item in (query, key, value))
        scores = torch.matmul(query / math.sqrt(head_width), key.transpose(-2, -1))
        if mask is not None:
            mask = mask.to(device=x.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask.to(dtype=scores.dtype)
        future = torch.ones(stages, stages, dtype=torch.bool, device=x.device).triu(diagonal=1)
        probabilities = scores.masked_fill(future, float('-inf')).softmax(dim=-1)
        probabilities = F.dropout(probabilities, p=attention.dropout, training=attention.training)
        if self._weights is not None:
            weights = self._weights[layer_index].to(device=x.device, dtype=probabilities.dtype)
            past = self._past.to(device=x.device)
            diagonal = torch.eye(stages, dtype=probabilities.dtype, device=x.device)
            gates = torch.where(past, weights, diagonal)
            probabilities = probabilities * gates
        messages = torch.matmul(probabilities, value).transpose(1, 2).reshape(nodes, stages, width)
        return layer.dropout1(attention.out_proj(messages))

    def _forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        # Execute the same layer equations explicitly: PyTorch's fused encoder
        # path would otherwise bypass a patched MultiheadAttention.forward.
        if src.dim() != 3 or tuple(src.shape[:2]) != (self.nodes, self.stages):
            raise ValueError(f'Expected temporal input [nodes={self.nodes}, stages={self.stages}, width]')
        if src_key_padding_mask is not None:
            raise ValueError('Padding is unsupported for fixed-size case explanations')
        if mask is not None and tuple(mask.shape) != (self.stages, self.stages):
            raise ValueError('Expected a two-dimensional temporal attention mask')
        x = src
        for index, layer in enumerate(self.encoder.layers):
            if layer.norm_first:
                x = x + self._attention(layer.norm1(x), layer, index, mask)
                residual = layer.norm2(x)
                x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(residual)))))
            else:
                x = layer.norm1(x + self._attention(x, layer, index, mask))
                x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x
