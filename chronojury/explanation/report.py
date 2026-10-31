"""Join spatial paths and active temporal links to the exact stage texts.

Each spatial stage uses one simple directed path, preferring the requested edge
count and then the greatest mean mask weight. Ties use the original edge IDs.
Temporal links retain their layer and head coordinates instead of being joined
to spatial paths or links from other layers.
"""
import argparse
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path


def ordered_opinions(discussion, agents):
    rounds = discussion['round_results']
    if not rounds or [r['round_idx'] for r in rounds] != list(range(len(rounds))):
        raise ValueError('Discussion stages must be consecutive and start at zero')
    names = [f'Agent_{i}' for i in range(agents)]
    opinions = []
    for item in rounds:
        entries = item['agent_conclusions']
        replies = {entry['agent']: entry['reply'] for entry in entries}
        if len(entries) != agents or set(replies) != set(names):
            raise ValueError('Discussion juror IDs do not match the explained graph')
        if any(not isinstance(replies[name], str) or not replies[name].strip() for name in names):
            raise ValueError('A selected discussion contains an empty opinion')
        opinions.append([replies[name] for name in names])
    return opinions


def opinions_digest(discussion, agents):
    opinions = [[text.strip() for text in stage] for stage in ordered_opinions(discussion, agents)]
    return hashlib.sha256(json.dumps(opinions, ensure_ascii=False).encode('utf-8')).hexdigest()


def critical_path(edges, num_nodes, length=3):
    """Find an existing directed path; never manufacture links between top edges."""
    if not 1 <= length < num_nodes:
        raise ValueError('Path length must be between 1 and the number of jurors')
    adjacency = {}
    seen = set()
    for edge in edges:
        src, dst, weight = edge['src'], edge['dst'], edge['mask_weight']
        if (not isinstance(src, int) or not isinstance(dst, int)
                or not 0 <= src < num_nodes or not 0 <= dst < num_nodes or src == dst
                or not math.isfinite(weight) or not 0 <= weight <= 1
                or (src, dst) in seen):
            raise ValueError('Invalid or duplicate directed explainer edge')
        seen.add((src, dst))
        adjacency.setdefault(src, []).append(edge)
    best, best_score = [], (0, 0.0, ())

    def visit(nodes, path):
        nonlocal best, best_score
        if path:
            score = (len(path), sum(e['mask_weight'] for e in path) / len(path),
                     tuple(-e['edge_id'] for e in path))
            if score > best_score:
                best, best_score = path, score
        if len(path) == length:
            return
        for edge in adjacency.get(nodes[-1], []):
            if edge['dst'] not in nodes:
                visit(nodes + [edge['dst']], path + [edge])

    for start in sorted(adjacency):
        visit([start], [])
    return best


def temporal_report(explainer, opinions, texts, names):
    """Aggregate connected temporal gates, retaining every original head record.

    A temporal link connects one node to itself at a later stage in one layer.
    ``active`` is supplied by the explainer's gradient-connectivity check; gates
    outside that check are never ranked or presented as explanation evidence.
    An active Case link requires recorded case summaries in round_node_texts.
    """
    metadata = explainer.get('temporal_metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Joint explanations require temporal_metadata')
    for key in ('num_layers', 'num_heads', 'spatial_layers'):
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            raise ValueError(f'Invalid temporal metadata: {key}')
    for key in ('diagonal', 'mask_location'):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f'Missing temporal metadata: {key}')
    raw_edges = explainer.get('temporal_edges')
    if not isinstance(raw_edges, list):
        raise ValueError('Joint explanations require a temporal_edges list')
    stages, nodes = len(opinions), len(names)
    coordinates, ids, groups = set(), set(), {}
    for edge in raw_edges:
        if not isinstance(edge, dict):
            raise ValueError('Invalid temporal explainer edge')
        bounds = {
            'layer': (1, metadata['num_layers']),
            'head': (0, metadata['num_heads'] - 1),
            'node_id': (0, nodes - 1),
            'source_stage': (1, stages),
            'target_stage': (1, stages),
        }
        for key, (lower, upper) in bounds.items():
            if type(edge.get(key)) is not int or not lower <= edge[key] <= upper:
                raise ValueError(f'Invalid temporal edge index: {key}')
        edge_id = edge.get('edge_id')
        if type(edge_id) is not int or edge_id < 0 or edge_id in ids:
            raise ValueError('Invalid or duplicate temporal edge ID')
        ids.add(edge_id)
        if edge['source_stage'] >= edge['target_stage']:
            raise ValueError('Temporal edges must point strictly from past to future')
        weight = edge.get('mask_weight')
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or not 0 <= weight <= 1):
            raise ValueError('Temporal mask weights must be finite and in [0, 1]')
        if type(edge.get('active')) is not bool:
            raise ValueError('Temporal edges require a boolean active flag')
        coordinate = tuple(edge[key] for key in
                           ('layer', 'head', 'node_id', 'source_stage', 'target_stage'))
        if coordinate in coordinates:
            raise ValueError('Duplicate temporal layer/head/node/stage coordinates')
        coordinates.add(coordinate)
        if edge['active']:
            key = tuple(edge[key] for key in ('layer', 'node_id', 'source_stage', 'target_stage'))
            groups.setdefault(key, []).append(dict(edge))

    def node_text(stage, node):
        if node != nodes - 1:
            return opinions[stage - 1][node]
        if texts is None or not isinstance(texts[stage - 1][node], str) or not texts[stage - 1][node].strip():
            raise ValueError('Active temporal Case links require recorded case summaries')
        return texts[stage - 1][node]

    links = []
    for (layer, node, source, target), head_edges in sorted(groups.items()):
        head_edges.sort(key=lambda edge: edge['head'])
        links.append({
            'layer': layer, 'node_id': node, 'node': names[node],
            'source_stage': source, 'target_stage': target,
            'mask_weight': math.fsum(edge['mask_weight'] for edge in head_edges) / len(head_edges),
            'active_heads': [edge['head'] for edge in head_edges], 'head_edges': head_edges,
            'source_text': node_text(source, node), 'target_text': node_text(target, node),
        })
    return {'temporal_metadata': dict(metadata),
            'temporal_edges': [dict(edge) for edge in raw_edges], 'temporal_links': links}


def build_report(explainer, discussion, path_length=3):
    if explainer.get('method') not in ('gnnexplainer_stage_edge_masks', 'gnnexplainer_spatial_temporal_masks'):
        raise ValueError('Legacy masks lack verified model/graph provenance; regenerate with chronojury.explanation.explainer')
    case_id = str(explainer['case_id'])
    if case_id != str(discussion['case_data']['id']):
        raise ValueError('Explainer and discussion case IDs differ')
    agents = explainer['num_agents']
    names = [f'Agent_{i}' for i in range(agents)] + ['Case']
    if explainer['node_order'] != names:
        raise ValueError('Unexpected node ordering')
    opinions = ordered_opinions(discussion, agents)
    if opinions_digest(discussion, agents) != explainer['provenance']['opinions_sha256']:
        raise ValueError('Discussion opinions differ from the explained input')
    edge_rounds = explainer['all_spatio_temporal_edges_by_round']
    if len(opinions) != len(edge_rounds) or len(opinions) != explainer['num_stages']:
        raise ValueError('Explainer and discussion stage counts differ')
    texts = discussion.get('round_node_texts')
    if texts is not None:
        if len(texts) != len(opinions) or any(
                len(row) != agents + 1 or [text.strip() for text in row[:-1]] !=
                [opinion.strip() for opinion in stage] for row, stage in zip(texts, opinions)):
            raise ValueError('Encoded node texts do not match stage-wise SWER opinions')
    stages = []
    for index, item in enumerate(edge_rounds):
        if item['round'] != index + 1 or item['edges_count'] != len(item['edges']):
            raise ValueError('Invalid stage index or edge count')
        path = critical_path(item['edges'], agents + 1, path_length)
        nodes = [path[0]['src']] + [edge['dst'] for edge in path] if path else []
        stage = {
            'stage': index + 1, 'path_edges': path,
            'path_score': sum(e['mask_weight'] for e in path) / len(path) if path else None,
            'juror_rationales': [{'node_id': node, 'agent': names[node], 'opinion': opinions[index][node]}
                                for node in nodes if node != agents],
        }
        if agents in nodes and texts is not None:
            stage['case_summary'] = texts[index][-1]
        stages.append(stage)
    report = {'case_id': case_id, 'predicted_class': explainer['predicted_class'],
              'class_labels': {'0': 'buyer', '1': 'seller'}, 'node_order': names,
              'method': 'per_stage_max_mean_mask_simple_directed_path', 'path_length': path_length,
              'stages': stages, 'diagnostics': explainer['diagnostics'], 'provenance': explainer['provenance']}
    if explainer['method'] == 'gnnexplainer_spatial_temporal_masks':
        report['explanation_method'] = explainer['method']
        report.update(temporal_report(explainer, opinions, texts, names))
    return report


def temporal_markdown(report):
    """Show temporal links directly, without inventing cross-layer paths."""
    lines = ['## 跨阶段信息传递', '',
             '时间边连接同一节点的较早阶段与较晚阶段，不连接不同评审员。'
             '下表仅列出对当前裁决有梯度连接的时间消息，门值按参与的注意力头取均值；'
             '未参与裁决的门值不作为解释证据。层编号从 1 开始，头编号从 0 开始。', '',
             '时间消息传递空间聚合后的节点表示；评审员表示可包含邻居和案件信息，'
             '后续时序层还会结合前层的时序信息。来源与目标原文用于理解节点的语义背景，'
             '门值不对应某一句裁决，也不证明它造成了评审员的立场变化。', '']
    links = report['temporal_links']
    if not links:
        return lines + ['没有可展示的有效时间消息。', '']
    lines += ['| 层／参与的头 | 时间消息 | 平均门值 | 来源阶段原文 | 目标阶段原文 |',
              '| --- | --- | --- | --- | --- |']

    def cell(text):
        return escape(text, quote=False).replace('|', '\\|').replace('\n', '<br>')

    for link in links:
        heads = '、'.join(map(str, link['active_heads']))
        source = f"{link['node']}（阶段 {link['source_stage']}）"
        target = f"{link['node']}（阶段 {link['target_stage']}）"
        lines.append(f"| 第 {link['layer']} 层／头 {heads} | {source} → {target} | "
                     f"{link['mask_weight']:.4f} | {cell(link['source_text'])} | {cell(link['target_text'])} |")
    return lines + ['']


def to_markdown(report):
    verdict = '支持买家' if report['predicted_class'] == 0 else '支持卖家'
    title = '空间与时序解释' if 'temporal_links' in report else '关键路径解释'
    lines = [f"# 案例 {report['case_id']}：{title}", '', f'模型裁决：{verdict}。', '',
             '以下按阶段列出关键路径及对应智能体的原始意见。边分数为 GNNExplainer 掩码权重。', '']
    if not report['diagnostics']['prediction_preserved']:
        lines += ['当前掩码后的预测与原裁决不同，这组掩码未保持原模型预测。', '']
    if 'temporal_links' in report:
        lines += ['空间路径与时间消息分别展示；门值表示本次解释保留信息的程度，不是概率。', '']
        lines += temporal_markdown(report)
    for stage in report['stages']:
        lines += [f"## 阶段 {stage['stage']}", '']
        for edge in stage['path_edges']:
            src, dst = (report['node_order'][edge[key]] for key in ('src', 'dst'))
            lines.append(f"- {src} → {dst}，解释权重 {edge['mask_weight']:.4f}")
        lines.append('')
        if 'case_summary' in stage:
            lines += ['### 本阶段案件节点', '', stage['case_summary'], '']
        for item in stage['juror_rationales']:
            lines += [f"### {item['agent']} 的原始意见", '', item['opinion'], '']
    if 'model_summary' in report:
        lines += ['## 辅助总结', '', report['model_summary']['text'], '']
    return '\n'.join(lines).rstrip() + '\n'


def make_summary_prompt(report):
    evidence = {key: report[key] for key in ('case_id', 'predicted_class', 'class_labels', 'node_order', 'stages')}
    temporal_instruction = ''
    if 'temporal_links' in report:
        evidence['temporal_metadata'] = report['temporal_metadata']
        evidence['temporal_links'] = report['temporal_links']
        evidence['prediction_preserved'] = report['diagnostics']['prediction_preserved']
        temporal_instruction = ('请同时概括有效时间消息，明确同一节点的来源阶段、目标阶段和所在层；'
                                '时间消息仅连接同一节点，不得编造不同评审员之间的时间边，'
                                '不得把不同层的边或阶段内空间路径拼成未经提供的传播链。'
                                '时间消息作用于经过空间聚合的节点表示，原文只提供语义背景，'
                                '不得声称门值控制某一句裁决或造成评审员的意见变化。'
                                '若 prediction_preserved 为 false，须说明这组掩码未保持原裁决。')
    return ('请根据给定关键路径及对应阶段的原始意见，写一段中文解释总结。'
            '概括关键智能体的裁决倾向及其随阶段的变化，保留智能体和阶段编号以便追溯。'
            '不得补充未提供的案件事实，不得改变模型裁决，不得把解释权重当成因果证明。'
            '如意见互相矛盾，请如实说明。' + temporal_instruction + '只输出总结正文。\n'
            + json.dumps(evidence, ensure_ascii=False))


def add_summary(report, model=None):
    from ..llm import chat_completion
    from ..swer.config import REASONING_CONFIG
    model = model or os.environ.get('CHRONOJURY_MODEL', REASONING_CONFIG['model'])
    response = chat_completion(model=model, messages=[{'role': 'user', 'content': make_summary_prompt(report)}],
                               temperature=0.2, max_tokens=1000)
    content = response['choices'][0]['message']['content']
    if not isinstance(content, str) or not content.strip():
        raise ValueError('The summary model returned no text; the Markdown report remains saved')
    report['model_summary'] = {'model': model, 'text': content.strip()}


def save_report(report, output_dir):
    """Write the readable Markdown report without changing other artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{report['case_id']}_critical_path_report.md"
    destination.write_text(to_markdown(report), encoding='utf-8')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--explainer', type=Path)
    source.add_argument('--explainer-dir', type=Path)
    parser.add_argument('--discussion', type=Path)
    parser.add_argument('--discussion-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--path-length', type=int, default=3)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--summary-model')
    args = parser.parse_args(argv)
    if (bool(args.explainer) != bool(args.discussion)
            or bool(args.explainer_dir) != bool(args.discussion_dir)):
        parser.error('Pair --explainer with --discussion, or --explainer-dir with --discussion-dir')
    paths = [args.explainer] if args.explainer else sorted(args.explainer_dir.glob('*_explainer.json'))
    if not paths:
        parser.error('No explainer files found')
    for path in paths:
        explainer = json.loads(path.read_text())
        discussions = [args.discussion] if args.discussion else list(
            args.discussion_dir.glob(f"{explainer['case_id']}_*_for_gnn.json"))
        if len(discussions) != 1:
            raise ValueError(f'{path.name}: expected exactly one matching discussion file')
        report = build_report(explainer, json.loads(discussions[0].read_text()), args.path_length)
        save_report(report, args.output_dir)
        if args.summary:
            add_summary(report, args.summary_model)
            save_report(report, args.output_dir)
        print(f"Saved report for case {report['case_id']}")


if __name__ == '__main__':
    main()
