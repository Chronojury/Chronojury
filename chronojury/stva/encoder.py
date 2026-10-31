"""Encode temporal verdict nodes with local Chinese RoBERTa or Qwen checkpoints.

The two original pooling rules are preserved: token zero (CLS) for RoBERTa,
and attention-mask-weighted mean pooling for Qwen. Both backends consume the
same ordered juror verdict nodes and one stage-specific case node in each round.
"""
import argparse
import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def load_encoder(model_path, backend, device=None):
    """Load existing local weights only; never download a checkpoint."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local encoder directory does not exist: {model_path}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if backend == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
    elif backend == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError("The Qwen tokenizer must provide a pad token or EOS token")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        raise ValueError(f"Unknown encoding backend: {backend}")
    return model.to(device).eval(), tokenizer, device


def encode_text(model, tokenizer, text, device, backend, max_length=512):
    """Return the original backend-specific text embedding without projection."""
    inputs = tokenizer(
        text if text is not None else "",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
        if backend == "roberta":
            embedding = hidden[:, 0, :]
        elif backend == "qwen":
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown encoding backend: {backend}")
    return embedding.squeeze(0).float().cpu().tolist()


def encode_text_batch(model, tokenizer, texts, device, backend, max_length=512):
    """Batch the same truncated texts and preserve the original pooling rule."""
    inputs = tokenizer([text if text is not None else "" for text in texts],
                       padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt")
    inputs = {key: value.pin_memory().to(device, non_blocking=True) if device.type == 'cuda'
              else value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs, **({'use_cache': False} if backend == 'qwen' else {})).last_hidden_state
        if backend == 'roberta':
            embedding = hidden[:, 0, :]
        else:
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return embedding.float().cpu().tolist()


def choose_batch_size(model, tokenizer, device, backend, max_length, maximum):
    """Probe actual forward-pass memory without changing weights or saved features."""
    if device.type != 'cuda':
        raise ValueError('Automatic encoder batch sizing requires a CUDA device')
    free, _ = torch.cuda.mem_get_info(device)
    budget = torch.cuda.memory_allocated(device) + int(free * .85)
    token = tokenizer('案例', return_tensors='pt')['input_ids'][0, 0].item()
    best, measurements, size = 1, [], 1
    while size <= maximum:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        inputs = None
        try:
            inputs = {'input_ids': torch.full((size, max_length), token, device=device, dtype=torch.long),
                      'attention_mask': torch.ones((size, max_length), device=device, dtype=torch.long)}
            start = time.monotonic()
            with torch.no_grad():
                output = model(**inputs, **({'use_cache': False} if backend == 'qwen' else {}))
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            measurements.append({'batch_size': size, 'peak_bytes': peak,
                                 'seconds': time.monotonic() - start})
            print(f'Encoder batch probe: {measurements[-1]}', flush=True)
            del output
            if peak > budget:
                break
            best = size
        except RuntimeError as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or 'canUse32BitIndexMath' in str(error):
                break
            raise
        finally:
            del inputs
            torch.cuda.empty_cache()
        size *= 2
    return best, measurements


def extract_case_id(file_path, data):
    match = re.match(r"^(\d+)_", Path(file_path).name)
    if match:
        return match.group(1)
    return str(data["case_data"]["id"])


def load_stage_case_text(case_id, stage_index, summaries_dir):
    case_id = str(case_id)
    case_name = f"CASE_{int(case_id):06d}" if case_id.isdigit() else f"CASE_{case_id}"
    path = Path(summaries_dir) / f"{case_name}_Stage{stage_index}_Summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))["node_texts"]
    return (
        f"买家核心主张: {summary.get('buyer_core_claim', '')}\n\n"
        f"卖家核心主张: {summary.get('seller_core_claim', '')}\n\n"
        f"争议焦点: {summary.get('dispute_focus', '')}"
    )


def build_round_node_texts(data, case_id, summaries_dir):
    """Use raw self verdicts; neighbor interaction text is never appended to nodes."""
    rounds = data["round_results"]
    if not rounds or [item["round_idx"] for item in rounds] != list(range(len(rounds))):
        raise ValueError("round_results must contain consecutive round indices starting at zero")
    agent_order = tuple(f"Agent_{index}" for index in range(len(rounds[0]['agent_conclusions'])))
    if len(agent_order) < 2:
        raise ValueError('Each round must contain at least two jurors')
    result = []
    for item in rounds:
        conclusions = item["agent_conclusions"]
        replies = {entry["agent"]: entry["reply"] for entry in conclusions}
        if len(conclusions) != len(agent_order) or set(replies) != set(agent_order):
            raise ValueError('Each round must contain the same consecutive juror IDs exactly once')
        texts = [replies[name].strip() for name in agent_order]
        texts.append(load_stage_case_text(case_id, item["round_idx"] + 1, summaries_dir))
        result.append(texts)
    return result


def encode_result(data, case_id, summaries_dir, model, tokenizer, device, backend, max_length=512):
    """Add temporal features and retain the existing final-round feature fields."""
    texts = build_round_node_texts(data, case_id, summaries_dir)
    features = [
        [encode_text(model, tokenizer, text, device, backend, max_length) for text in one_round]
        for one_round in texts
    ]
    result = dict(data)
    result["round_node_texts"] = texts
    result["round_node_features"] = features
    result["node_texts"] = texts[-1]
    result["node_features"] = features[-1]
    return result


def encode_results(results_dir, summaries_dir, model_path, backend, *, device=None, max_length=512,
                   output_dir=None, batch_size=1, auto_batch_size=False, max_batch_size=1024,
                   prefetch_batches=2, shard_index=0, num_shards=1, skip_existing=False):
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not Path(summaries_dir).is_dir():
        raise FileNotFoundError(f"Stage summary directory does not exist: {summaries_dir}")
    files = sorted(results_dir.rglob("*_for_gnn.json"))[shard_index::num_shards]
    if not files:
        raise FileNotFoundError(f"No *_for_gnn.json results found in {results_dir}")
    destination = Path(output_dir) if output_dir is not None else results_dir
    if skip_existing:
        files = [p for p in files if not (destination / p.relative_to(results_dir)).exists()]
    if not files:
        return []
    model, tokenizer, device = load_encoder(model_path, backend, device)
    measurements = []
    if auto_batch_size:
        batch_size, measurements = choose_batch_size(model, tokenizer, device, backend, max_length, max_batch_size)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f'encoding_shard_{shard_index}.json').write_text(json.dumps({
        'backend': backend, 'batch_size': batch_size, 'max_length': max_length,
        'shard_index': shard_index, 'num_shards': num_shards, 'cases': len(files),
        'memory_probes': measurements}, indent=2))
    print(f'Encoding shard {shard_index}/{num_shards}: {len(files)} cases, text batch size {batch_size}', flush=True)
    written = []
    first = json.loads(files[0].read_text(encoding='utf-8'))
    stages = len(first['round_results'])
    nodes = len(first['round_results'][0]['agent_conclusions']) + 1
    case_batch_size = max(1, batch_size // (stages * nodes))
    groups = iter([files[i:i + case_batch_size] for i in range(0, len(files), case_batch_size)])
    def prepare(group):
        records = []
        for path in group:
            data = json.loads(path.read_text(encoding='utf-8'))
            summary_root = Path(summaries_dir) / path.relative_to(results_dir).parent
            texts = build_round_node_texts(data, extract_case_id(path, data), summary_root)
            records.append((path, data, texts))
        return records
    with ThreadPoolExecutor(max_workers=1) as reader, tqdm(total=len(files), desc=f'Encoding ({backend})') as progress:
        pending = deque()
        for _ in range(prefetch_batches + 1):
            group = next(groups, None)
            if group is not None: pending.append(reader.submit(prepare, group))
        while pending:
            records = pending.popleft().result()
            group = next(groups, None)
            if group is not None: pending.append(reader.submit(prepare, group))
            texts = [text for _, _, rounds in records for one_round in rounds for text in one_round]
            features = []
            for i in range(0, len(texts), batch_size):
                if batch_size == 1:
                    features.append(encode_text(model, tokenizer, texts[i], device, backend, max_length))
                else:
                    features.extend(encode_text_batch(model, tokenizer, texts[i:i + batch_size], device, backend, max_length))
            offset = 0
            for path, data, rounds in records:
                round_features = []
                for one_round in rounds:
                    round_features.append(features[offset:offset + len(one_round)])
                    offset += len(one_round)
                result = dict(data, round_node_texts=rounds, round_node_features=round_features,
                              node_texts=rounds[-1], node_features=round_features[-1])
                output = destination / path.relative_to(results_dir)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(output)
                written.append(output)
                progress.update(1)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("roberta", "qwen"))
    parser.add_argument("--model-path", required=True, type=Path, help="Existing local encoder checkpoint")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--summaries-dir", required=True, type=Path)
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0; defaults to automatic selection")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, help="Write encoded copies here; default: update results in place")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--auto-batch-size', action='store_true')
    parser.add_argument('--max-batch-size', type=int, default=1024)
    parser.add_argument('--prefetch-batches', type=int, default=2)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if min(args.batch_size, args.max_batch_size, args.num_shards) < 1 or args.prefetch_batches < 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Invalid batch size, prefetch count or shard index')
    outputs = encode_results(
        args.results_dir, args.summaries_dir, args.model_path, args.backend,
        device=args.device, max_length=args.max_length,
        output_dir=args.output_dir,
        batch_size=args.batch_size, auto_batch_size=args.auto_batch_size, max_batch_size=args.max_batch_size,
        prefetch_batches=args.prefetch_batches, shard_index=args.shard_index, num_shards=args.num_shards,
        skip_existing=args.skip_existing,
    )
    print(f"Encoded {len(outputs)} result files")


if __name__ == "__main__":
    main()
