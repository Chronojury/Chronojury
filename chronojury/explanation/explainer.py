"""Explain frozen STVA predictions using spatial and causal temporal messages."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from ..stva import config
from ..stva.classifier import DebateClassifier
from ..stva.model import shared_model_manager
from ..stva.trainer import load_case_edges
from .report import build_report, opinions_digest, save_report, add_summary
from .temporal import TemporalMaskContext


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _seed_mask_rng(seed, device):
    # fork_rng restores CPU and the selected CUDA device. Do not seed other
    # CUDA generators, which a CPU explanation (or a single GPU) does not own.
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.default_generators[index].manual_seed(seed)


class VerdictGraph(nn.Module):
    """Expose the existing complete forward pass to PyG's graph explainer."""

    def __init__(self, manager, classifier):
        super().__init__()
        self.manager = manager
        self.structural = manager.stva_structural
        self.temporal = manager.stva_temporal
        self.projection = manager.gat_input_projection
        self.classifier = classifier

    def forward(self, x, edge_index, stages):
        nodes = self.manager.num_nodes
        # The expanded graph has an identical topology at each stage. The
        # existing forward_batch recreates these same edges in stage order.
        base_edges = edge_index[:, edge_index[0] < nodes]
        features = x.reshape(1, stages, nodes, self.manager.input_dim)
        return self.classifier(self.manager.forward_batch(features, base_edges))


def load_model(checkpoint_path, device, temporal_heads=2):
    """Restore architecture and weights without creating a training optimizer.

    Older checkpoints do not record temporal head count; their training default
    is two. --temporal-heads supports checkpoints trained with another count.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('ablation_mode', 'complete') != 'complete':
        raise ValueError('Explanations require a complete STVA checkpoint')
    model_config = deepcopy(config.GNN_MODEL_CONFIG)
    structural = model_config['stva']['structural']
    weights = checkpoint['gat_state_dict']
    layers = len([key for key in weights if key.endswith('.att_src')])
    first = weights['gat_layers.0.att_src']
    last = weights[f'gat_layers.{layers - 1}.att_src']
    input_width = checkpoint.get('input_dim', weights['gat_layers.0.lin.weight'].shape[1])
    structural.update(input_dim=input_width, num_layers=layers,
                      num_heads=first.shape[1], hidden_dim=first.shape[2], output_dim=last.shape[2])
    projection = checkpoint.get('gat_input_projection_state_dict')
    model_config['stva']['gat_input_projection'] = {
        'enabled': projection is not None,
        'output_dim': projection['0.weight'].shape[0] if projection is not None else input_width,
    }
    temporal = checkpoint.get('stva_temporal_state_dict', checkpoint.get('attention_state_dict'))
    if temporal is None:
        raise ValueError('The checkpoint is missing temporal weights')
    temporal_layers = len([key for key in temporal if key.endswith('self_attn.in_proj_weight')])
    model_config['stva']['temporal'].update(input_dim=last.shape[2],
                                           num_layers=temporal_layers, num_heads=temporal_heads)
    agents = checkpoint.get('num_agents', 6)
    model_config.update(num_agents=agents, graph_topology=[[agents, i] for i in range(agents)])
    config.GNN_MODEL_CONFIG.clear()
    config.GNN_MODEL_CONFIG.update(model_config)
    config.ABLATION_MODE = 'complete'
    shared_model_manager.initialize(device=device, force_reload=True)
    shared_model_manager.load_checkpoint_weights(checkpoint)
    classifier = DebateClassifier(**checkpoint['model_config']).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model = VerdictGraph(shared_model_manager, classifier).to(device).eval()
    model.requires_grad_(False)
    return model


def explain_case(model, features, edges, *, epochs=250, lr=0.05,
                 size_coefficient=0.001, entropy_coefficient=2.0, seed=42,
                 temporal_masks=True):
    """Optimize message masks against frozen predictions, without true labels."""
    stages, nodes, width = features.shape
    if nodes != model.manager.num_nodes or width != model.manager.input_dim:
        raise ValueError('Feature nodes/width do not match the checkpoint')
    if not 1 <= stages <= model.temporal.position_embeddings.shape[1]:
        raise ValueError('Stage count exceeds the trained temporal positional capacity')
    if not torch.isfinite(features).all():
        raise ValueError('Features contain non-finite values')
    if edges.numel() == 0 or edges.min() < 0 or edges.max() >= nodes:
        raise ValueError('Invalid case graph')
    if (edges[0] == edges[1]).any():
        raise ValueError('Self-loops are supplied internally by GAT, not the following network')
    if epochs < 1 or lr <= 0 or size_coefficient < 0 or entropy_coefficient < 0:
        raise ValueError('Invalid mask optimization settings')
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Explanations require an evaluation-mode, frozen model')
    expanded_edges = torch.cat([edges + stage * nodes for stage in range(stages)], dim=1)
    x = features.detach().reshape(stages * nodes, width)
    with torch.no_grad():
        original = model(x, expanded_edges, stages)
        try:
            set_masks(model.structural, x.new_ones(expanded_edges.shape[1]),
                      expanded_edges, apply_sigmoid=False)
            all_edges = model(x, expanded_edges, stages)
        finally:
            clear_masks(model.structural)
        torch.testing.assert_close(original, all_edges, rtol=1e-5, atol=1e-6)

    if temporal_masks:
        return _explain_joint(model, x, expanded_edges, stages, original,
                              epochs=epochs, lr=lr, size_coefficient=size_coefficient,
                              entropy_coefficient=entropy_coefficient, seed=seed)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr, edge_size=size_coefficient,
                               edge_ent=entropy_coefficient),
        explanation_type='model', node_mask_type=None, edge_mask_type='object',
        model_config={'mode': 'multiclass_classification', 'task_level': 'graph', 'return_type': 'raw'},
    )
    devices = [features.device.index] if features.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            _seed_mask_rng(seed, features.device)
            explanation = explainer(x, expanded_edges, stages=stages)
        with torch.no_grad():
            masked = explainer.get_masked_prediction(
                x, expanded_edges, edge_mask=explanation.edge_mask, stages=stages)
    finally:
        clear_masks(model.structural)
    mask = explanation.edge_mask.detach().reshape(stages, -1).cpu()
    predicted = int(original.argmax(dim=-1))
    diagnostics = {
        'original_logits': original[0].tolist(),
        'original_probabilities': original.softmax(-1)[0].tolist(),
        'masked_probabilities': masked.softmax(-1)[0].tolist(),
        'masked_predicted_class': int(masked.argmax(dim=-1)),
        'prediction_preserved': int(masked.argmax(dim=-1)) == predicted,
        'all_ones_max_logit_error': float((original - all_edges).abs().max()),
    }
    return mask, predicted, diagnostics


def _explain_joint(model, x, expanded_edges, stages, original, *, epochs, lr,
                   size_coefficient, entropy_coefficient, seed):
    """GNNExplainer fidelity + sparsity + entropy over both message families.

    Spatial gates multiply GAT messages; temporal gates multiply attention
    messages after softmax. Both target the same frozen model prediction.
    Only prediction-connected gates receive regularization, as in PyG's
    GNNExplainer. Fixed self-messages and residual connections stay available.
    """
    devices = [x.device.index] if x.is_cuda else []
    nodes = model.manager.num_nodes
    predicted = int(original.argmax(-1))
    target = original.argmax(-1)
    with TemporalMaskContext(model.temporal, stages=stages, nodes=nodes) as temporal:
        allowed = temporal.allowed_mask.to(device=x.device)

        def dense(values):
            weights = x.new_ones(temporal.mask_shape)
            weights[allowed] = values
            return weights

        def predict(spatial_weights, temporal_weights, *, apply_sigmoid=False):
            temporal.set_weights(temporal_weights)
            try:
                set_masks(model.structural, spatial_weights, expanded_edges, apply_sigmoid=apply_sigmoid)
                return model(x, expanded_edges, stages)
            finally:
                clear_masks(model.structural)

        with torch.no_grad():
            all_edges = predict(x.new_ones(expanded_edges.shape[1]), dense(x.new_ones(int(allowed.sum()))))
        torch.testing.assert_close(original, all_edges, rtol=1e-5, atol=1e-6)

        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            _seed_mask_rng(seed, x.device)
            spatial_logits = nn.Parameter(torch.randn(expanded_edges.shape[1], device=x.device)
                                           * (2.0 / x.shape[0]) ** 0.5)
            temporal_logits = nn.Parameter(torch.randn(int(allowed.sum()), device=x.device)
                                            * (2.0 / stages) ** 0.5)
            optimizer = torch.optim.Adam([spatial_logits, temporal_logits], lr=lr)
            active_spatial = active_temporal = None
            for step in range(epochs):
                optimizer.zero_grad()
                spatial_weights, temporal_values = spatial_logits.sigmoid(), temporal_logits.sigmoid()
                masked = predict(spatial_logits, dense(temporal_values), apply_sigmoid=True)
                loss = F.cross_entropy(masked, target)
                if active_spatial is not None:
                    retained = torch.cat([spatial_weights[active_spatial], temporal_values[active_temporal]])
                    if retained.numel():
                        entropy = -retained * (retained + 1e-15).log() - (1 - retained) * (1 - retained + 1e-15).log()
                        loss = loss + size_coefficient * retained.sum() + entropy_coefficient * entropy.mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite explanation loss')
                loss.backward()
                if step == 0:
                    if spatial_logits.grad is None:
                        raise ValueError('Spatial masks are disconnected from the prediction')
                    active_spatial = spatial_logits.grad.detach() != 0
                    active_temporal = (temporal_logits.grad.detach() != 0 if temporal_logits.grad is not None
                                       else torch.zeros_like(temporal_logits, dtype=torch.bool))
                optimizer.step()
            spatial_weights = (spatial_logits.sigmoid() * active_spatial).detach()
            temporal_values = (temporal_logits.sigmoid() * active_temporal).detach()

        with torch.no_grad():
            masked = predict(spatial_weights, dense(temporal_values))
            spatial_selected = (spatial_weights > 0.5).to(x.dtype)
            temporal_selected = (temporal_values > 0.5).to(x.dtype)
            configurations = {
                'retain_joint': (spatial_selected, temporal_selected),
                'remove_joint': (1 - spatial_selected, 1 - temporal_selected),
                'retain_temporal': (torch.ones_like(spatial_selected), temporal_selected),
                'remove_temporal': (torch.ones_like(spatial_selected), 1 - temporal_selected),
                'without_cross_stage_messages': (torch.ones_like(spatial_selected), torch.zeros_like(temporal_selected)),
            }
            checks = {}
            for name, (spatial_selection, temporal_selection) in configurations.items():
                logits = predict(spatial_selection, dense(temporal_selection))
                checks[name] = {'predicted_class': int(logits.argmax(-1)), 'logits': logits[0].tolist(),
                                'prediction_preserved': int(logits.argmax(-1)) == predicted}

        temporal_records = []
        for index, (layer, node, head, target_stage, source_stage) in enumerate(allowed.nonzero().tolist()):
            temporal_records.append({'edge_id': index, 'layer': layer + 1, 'head': head,
                                     'node_id': node, 'source_stage': source_stage + 1,
                                     'target_stage': target_stage + 1,
                                     'mask_weight': float(temporal_values[index]),
                                     'active': bool(active_temporal[index])})
        metadata = {'num_layers': temporal.mask_shape[0], 'num_heads': temporal.mask_shape[2],
                    'spatial_layers': len(model.structural.gat_layers),
                    'diagonal': 'fixed at one; future-stage connections fixed at zero',
                    'mask_location': 'attention messages after softmax, without renormalization',
                    'direction': 'earlier source stage to later target stage, within the same node',
                    'inactive': 'zero prediction gradient in the initial fidelity step; excluded from ranking',
                    'selection_threshold': 0.5}
    with torch.no_grad():
        restored = model(x, expanded_edges, stages)
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    diagnostics = {
        'original_logits': original[0].tolist(),
        'original_probabilities': original.softmax(-1)[0].tolist(),
        'masked_probabilities': masked.softmax(-1)[0].tolist(),
        'masked_predicted_class': int(masked.argmax(-1)),
        'prediction_preserved': int(masked.argmax(-1)) == predicted,
        'all_ones_max_logit_error': float((original - all_edges).abs().max()),
        'restored_max_logit_error': float((original - restored).abs().max()),
        'mask_checks': checks,
        'selected_spatial_edges': int(spatial_selected.sum()),
        'selected_temporal_head_edges': int(temporal_selected.sum()),
        'temporal_edges': temporal_records, 'temporal_metadata': metadata,
    }
    return spatial_weights.reshape(stages, -1).cpu(), predicted, diagnostics


def explanation_payload(case_id, mask, edges, predicted, diagnostics):
    stages, count = mask.shape
    nodes = int(edges.max()) + 1
    importance = torch.zeros(stages, nodes)
    rounds = []
    for stage in range(stages):
        stage_edges = []
        for index, (src, dst) in enumerate(edges.t().tolist()):
            weight = float(mask[stage, index])
            importance[stage, src] += weight
            importance[stage, dst] += weight
            stage_edges.append(dict(edge_id=index, src=src, dst=dst, mask_weight=weight))
        importance[stage] /= importance[stage].max().clamp_min(1e-12)
        rounds.append({'round': stage + 1, 'edges_count': count,
                       'edges': sorted(stage_edges, key=lambda e: (-e['mask_weight'], e['edge_id']))})
    flat = sorted([dict(round=item['round'], **edge) for item in rounds for edge in item['edges']],
                  key=lambda e: (-e['mask_weight'], e['round'], e['edge_id']))
    payload = {
        'case_id': str(case_id), 'predicted_class': predicted,
        'method': 'gnnexplainer_stage_edge_masks', 'num_agents': nodes - 1, 'num_stages': stages,
        'node_order': [f'Agent_{i}' for i in range(nodes - 1)] + ['Case'],
        'stage_node_importance': importance.tolist(),
        'all_spatio_temporal_edges_by_round': rounds,
        'all_spatio_temporal_edges_sorted_flat': flat, 'top_4_edges_summary': flat[:4],
        'diagnostics': {key: value for key, value in diagnostics.items()
                        if key not in ('temporal_edges', 'temporal_metadata')},
    }
    if 'temporal_edges' in diagnostics:
        payload.update(method='gnnexplainer_spatial_temporal_masks',
                       temporal_edges=diagnostics['temporal_edges'],
                       temporal_metadata=diagnostics['temporal_metadata'])
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--network-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case-id', help='Explain one case instead of the entire directory')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temporal-heads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--size-coefficient', type=float, default=0.001)
    parser.add_argument('--entropy-coefficient', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spatial-only', action='store_true', help='Explain only intra-stage graph edges')
    parser.add_argument('--path-length', type=int, default=3)
    parser.add_argument('--summary', action='store_true', help='Add an optional LLM summary of the selected rationales')
    parser.add_argument('--summary-model', help='Model name at the configured OpenAI-compatible endpoint')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.lr <= 0 or args.size_coefficient < 0 or args.entropy_coefficient < 0
            or args.temporal_heads < 1 or args.path_length < 1 or (args.limit is not None and args.limit < 1)):
        parser.error('Invalid explainer optimization, path length, or limit')
    files = sorted(args.features_dir.glob('*_for_gnn.json'))
    if args.case_id is not None:
        files = [path for path in files if path.name.split('_', 1)[0] == args.case_id]
        if len(files) != 1:
            parser.error('--case-id must identify exactly one feature file')
    if not files:
        parser.error('No feature files found')
    ids = [path.name.split('_', 1)[0] for path in files]
    if len(ids) != len(set(ids)):
        parser.error('Multiple feature files for the same case')
    files = files[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and not args.summary:
        pending = []
        for path in files:
            case_id = path.name.split('_', 1)[0]
            destination = args.output_dir / f'{case_id}_critical_path_report.md'
            if destination.exists():
                print(f'{case_id}: report already exists; use --overwrite to regenerate', flush=True)
            else:
                pending.append(path)
        files = pending
    if not files:
        return
    model = load_model(args.checkpoint, args.device, args.temporal_heads)
    if args.path_length > model.manager.num_agents:
        parser.error('--path-length cannot exceed the number of jurors')
    checkpoint_hash = file_digest(args.checkpoint)
    settings = {key: getattr(args, key) for key in
                ('epochs', 'lr', 'size_coefficient', 'entropy_coefficient', 'seed')}
    settings['temporal_masks'] = not args.spatial_only
    for path in files:
        case_id = path.name.split('_', 1)[0]
        data = json.loads(path.read_text())
        if str(data['case_data']['id']) != case_id:
            raise ValueError(f'Case ID mismatch in {path.name}')
        network = args.network_dir / f'{case_id}_following_networks.json'
        following = json.loads(network.read_text())
        if isinstance(following, dict) and set(following) != {
                str(i) for i in range(model.manager.num_agents)}:
            raise ValueError('Following network juror IDs do not match the checkpoint')
        provenance = {'checkpoint_sha256': checkpoint_hash, 'features_sha256': file_digest(path),
                      'network_sha256': file_digest(network), 'settings': settings,
                      'temporal_heads': args.temporal_heads,
                      'explanation_scope': 'spatial' if args.spatial_only else 'spatial_temporal',
                      'self_loops': 'fixed at one; excluded from path search',
                      'opinions_sha256': opinions_digest(data, model.manager.num_agents)}
        features = torch.tensor(data['round_node_features'], dtype=torch.float32, device=args.device)
        if features.dim() == 4 and features.shape[0] == 1:
            features = features.squeeze(0)
        edges = load_case_edges(case_id, args.network_dir, config.GNN_MODEL_CONFIG['graph_topology'])
        edges = torch.tensor(edges, dtype=torch.long, device=args.device).t().contiguous()
        mask, predicted, diagnostics = explain_case(model, features, edges, **settings)
        payload = explanation_payload(case_id, mask, edges.cpu(), predicted, diagnostics)
        payload['provenance'] = provenance
        report = build_report(payload, data, args.path_length)
        save_report(report, args.output_dir)
        if args.summary:
            add_summary(report, args.summary_model)
            save_report(report, args.output_dir)
        print(f'{case_id}: {case_id}_critical_path_report.md saved', flush=True)


if __name__ == '__main__':
    main()
