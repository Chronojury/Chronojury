"""Produce a separate case-information summary from claims available at each stage."""
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List
import argparse
import concurrent.futures
from ..llm import chat_completion
from ..runtime import case_lock
from ..tracing import trace_case, trace_scope, record_event
from .config import REASONING_CONFIG, SUMMARY_CONFIG
from .json_output import load_model_json

# =========================================================================
# 🚀 1. Global Relative Path Configurations
# =========================================================================
DATASET_DIR = Path(SUMMARY_CONFIG["data_dir"])
SEGMENT_DIR = Path(SUMMARY_CONFIG["segment_dir"])
OUTPUT_DIR = Path(SUMMARY_CONFIG["output_dir"])

# =========================================================================
# 🚀 2. ChronoJury Stage-Guided Summary Pipeline
# =========================================================================
class ChronoJuryStageSummaryPipeline:
    def __init__(self, model_name=None, api_base=None, api_key=None, *, temperature=None):
        self.request_options = {}
        if api_base:
            self.request_options["api_base"] = api_base
        if api_key:
            self.request_options["api_key"] = api_key
        self.model_name = model_name or os.getenv("CHRONOJURY_MODEL", REASONING_CONFIG["model"])
        self.temperature = SUMMARY_CONFIG["temperature"] if temperature is None else temperature

    def parse_case_to_standard_format(self, raw_case: Dict, case_dir_name: str) -> Dict:
        raw_id = raw_case.get('id', case_dir_name)
        try:
            case_id_num = int(raw_id)
            case_id_formatted = f"CASE_{case_id_num:06d}"
        except (ValueError, TypeError):
            case_id_num = raw_id
            case_id_formatted = f"CASE_{case_dir_name}"

        case_id_str = str(case_dir_name) 
        
        title = raw_case.get('title', '')
        product = title.split('是否')[0] if '是否' in title else '商品'
        dispute = raw_case.get('title', '').replace('是否存在', '').replace('的情况？期待你来揭示真相！', '')
        
        buyer_claim = raw_case.get('buyer_claim', '')
        seller_claim = raw_case.get('seller_claim', '')
        
        evidences_raw = raw_case.get('case_details', {}).get('evidences', [])
        if not evidences_raw and isinstance(raw_case.get('evidences'), list):
            evidences_raw = raw_case.get('evidences', [])

        all_evidences = []
        for idx, ev in enumerate(evidences_raw):
            identity = ev.get('identity', '')
            content = ev.get('content', {}) or ev
            text = content.get('text', '') if isinstance(content, dict) else ''
            
            all_evidences.append({
                "evidence_index": idx,
                "identity": identity,
                "text": text
            })

        return {
            "case_id_str": case_id_str,
            "case_id_formatted": case_id_formatted,
            "original_id": case_id_num,
            "title": title,
            "product": product,
            "dispute": dispute,
            "buyer_claim": buyer_claim,
            "seller_claim": seller_claim,
            "all_evidences": all_evidences
        }

    def generate(self, prompt: str) -> str:
        """Call the configured OpenAI-compatible endpoint with the original sampling settings."""
        response = chat_completion(
            **self.request_options,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            timeout=SUMMARY_CONFIG["timeout"],
        )
        return response.choices[0].message.content

    def extract_stage_evidences(self, all_evidences: List[Dict], stage_indices: List[int]) -> tuple:
        """Select the claims available up to a stage, using its cumulative indices."""
        buyer_texts = []
        seller_texts = []
        
        ev_map = {ev["evidence_index"]: ev for ev in all_evidences}
        
        for idx in stage_indices:
            if idx in ev_map:
                ev = ev_map[idx]
                identity = ev.get("identity", "")
                text = ev.get("text", "").strip()
                if text:
                    if identity == "买家":
                        buyer_texts.append(f"[证据{idx}]: {text}")
                    elif identity == "卖家":
                        seller_texts.append(f"[证据{idx}]: {text}")
                    else:
                        buyer_texts.append(f"[证据{idx}({identity})]: {text}")

        buyer_desc = "\n".join(buyer_texts) if buyer_texts else "本阶段无新增买家质证描述"
        seller_desc = "\n".join(seller_texts) if seller_texts else "本阶段无新增卖家质证描述"
        return buyer_desc, seller_desc

    def summarize_stage_case_info(self, case_data: Dict, stage_idx: int, stage_indices: List[int]) -> Dict:
        """Apply F_CaseInfo(C_<=t); later-stage claims and other summaries are excluded."""
        buyer_desc, seller_desc = self.extract_stage_evidences(case_data['all_evidences'], stage_indices)
        c_id = case_data.get('case_id_formatted', 'N/A')

        # Retained prompt structure with stage context adaptation
        prompt = f"""根据案件本阶段的文本，理解买卖双方的诉求和纠纷原因。

案件信息：
- 案件编号：{c_id} (当前处于阶段 {stage_idx + 1})
- 商品：{case_data.get('product', 'N/A')}
- 争议点：{case_data.get('dispute', 'N/A')}
- 买家主张：{case_data.get('buyer_claim', 'N/A')}
- 卖家主张：{case_data.get('seller_claim', 'N/A')}

截至第 {stage_idx + 1} 阶段，买方已呈现的全部质证陈述：
{buyer_desc}

截至第 {stage_idx + 1} 阶段，卖方已呈现的全部质证陈述：
{seller_desc}

请结合双方截至本阶段的全部言辞与交锋，分析：
1. 买家当前的核心诉求是什么？
2. 卖家当前的核心诉求是什么？
3. 双方当前争议的焦点是什么？

每个问题的回答不超过50个字，总共的回答不超过150字.
请以唯一的 JSON 格式输出，严禁带有 Markdown 代码包裹符：
{{
  "buyer_core_claim": "买家核心诉求",
  "seller_core_claim": "卖家核心诉求",
  "dispute_focus": "双方争议的焦点"
}}"""

        response = self.generate(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match is None:
            raise ValueError(f"Case {c_id}, stage {stage_idx + 1}: model response contains no JSON object")
        parsed_res = load_model_json(json_match.group())
        return {
            "buyer_core_claim": parsed_res.get("buyer_core_claim", "").strip(),
            "seller_core_claim": parsed_res.get("seller_core_claim", "").strip(),
            "dispute_focus": parsed_res.get("dispute_focus", "").strip(),
        }

# =========================================================================
# 🚀 3. Single Case Processing Task (For Thread Pool)
# =========================================================================
@trace_case("case_summary")
def process_case_task(case_dir: Path, pipeline: ChronoJuryStageSummaryPipeline):
    with case_lock(OUTPUT_DIR, case_dir.name):
        _process_case_task(case_dir, pipeline)


def _process_case_task(case_dir: Path, pipeline: ChronoJuryStageSummaryPipeline):
    case_file = case_dir / "data.json"
    case_dir_name = case_dir.name  
    
    with open(case_file, "r", encoding="utf-8") as f:
        raw_case = json.load(f)

    standard_case = pipeline.parse_case_to_standard_format(raw_case, case_dir_name)
    case_id_str = standard_case["case_id_str"]
    c_id_formatted = standard_case["case_id_formatted"]
    
    segment_file = SEGMENT_DIR / f"{case_id_str}_segment.json"
    
    with open(segment_file, "r", encoding="utf-8") as seg_f:
        boundaries = json.load(seg_f)["boundaries"]

    print(f"\n▶ Start folders: {case_dir_name} ({c_id_formatted}) | Segments: {len(boundaries)}")

    for stage_idx, stage_indices in enumerate(boundaries):

        output_file = OUTPUT_DIR / f"{c_id_formatted}_Stage{stage_idx + 1}_Summary.json"
        
        if output_file.exists():
            print(f"   ⏭️ [{c_id_formatted}] Stage {stage_idx + 1} 已存在，跳过处理。")
            continue

        # Build each stage's case summary from claims available up to that
        # stage, excluding future claims.
        cumulative_indices = []
        for past_idx in range(stage_idx + 1):
            cumulative_indices.extend(boundaries[past_idx])
        
        cumulative_indices = sorted(list(set(cumulative_indices)))
        print(f"   [{c_id_formatted}] Stage {stage_idx + 1}/{len(boundaries)}: process {stage_indices}...")
        
        with trace_scope(stage=stage_idx + 1):
            summary_result = pipeline.summarize_stage_case_info(standard_case, stage_idx, cumulative_indices)
            record_event("stage_summary", evidence_indices=cumulative_indices, summary=summary_result)
        
        final_save_payload = {
            "case_id": c_id_formatted,
            "folder_name": case_dir_name,
            "original_id": standard_case["original_id"],
            "stage_index": stage_idx + 1,
            "current_stage_evidence_indices": stage_indices,      # 当前阶段新增
            "cumulative_evidence_indices": cumulative_indices,     # 累积审视的证据
            "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "standardized_meta": {
                "product": standard_case["product"],
                "dispute": standard_case["dispute"]
            },
            "node_texts": {
                "buyer_core_claim": summary_result["buyer_core_claim"],
                "seller_core_claim": summary_result["seller_core_claim"],
                "dispute_focus": summary_result["dispute_focus"]
            }
        }
        
        temporary = output_file.with_suffix('.tmp')
        with temporary.open("w", encoding="utf-8") as out_f:
            json.dump(final_save_payload, out_f, ensure_ascii=False, indent=4)
        temporary.replace(output_file)
            
        print(f"   💾 [{c_id_formatted}] save {stage_idx + 1} summary -> {output_file.name}")

# =========================================================================
# 🚀 4. Execution Pipeline (Multi-threading)
# =========================================================================
def main(argv=None):
    global DATASET_DIR, SEGMENT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="ChronoJury Stage Summary Extraction with ThreadPool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int,
                        default=SUMMARY_CONFIG["max_workers"], help="Concurrent cases")
    parser.add_argument("--start", type=int, default=1,
                        help="1-based position in numeric case-ID order")
    parser.add_argument("--end", type=int, help="Inclusive position in numeric case-ID order")
    parser.add_argument('--data-dir', type=Path, default=DATASET_DIR)
    parser.add_argument('--segment-dir', type=Path, default=SEGMENT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--model')
    parser.add_argument('--api-base')
    parser.add_argument('--temperature', type=float, default=SUMMARY_CONFIG["temperature"],
                        help="Stage-summary sampling temperature")
    args = parser.parse_args(argv)
    if args.start < 1 or (args.end is not None and args.end < args.start) or args.max_workers < 1:
        parser.error("Invalid case range or worker count")
    if not 0 <= args.temperature <= 2:
        parser.error("Invalid sampling temperature")
    DATASET_DIR, SEGMENT_DIR, OUTPUT_DIR = args.data_dir, args.segment_dir, args.output_dir

    pipeline = ChronoJuryStageSummaryPipeline(model_name=args.model, api_base=args.api_base,
                                            temperature=args.temperature)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"❌ Do not exist: {DATASET_DIR}")
        exit(1)

    case_dirs = sorted(
        (d for d in DATASET_DIR.iterdir()
         if d.is_dir() and d.name.isdigit() and (d / "data.json").exists()),
        key=lambda path: int(path.name),
    )[args.start - 1:args.end]

    print(f"📖 scan {len(case_dirs)} folders in {DATASET_DIR}")
    print(f"⚡ Running with ThreadPoolExecutor (max_workers={args.max_workers})...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_case_task, case_dir, pipeline) for case_dir in case_dirs]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("\n🏁 Done")

if __name__ == "__main__":
    main()
