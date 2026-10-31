import time
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError
from . import config
from .json_output import load_model_json
from .prompts import STAGE_PROMPTS
from ..llm import chat_completion
from ..tracing import trace_operation, record_event
import backoff  # 仅保留在查询函数中使用
import base64
import json
import cv2
import os
from typing import Optional, List, Dict
from pathlib import Path
from uuid import uuid4
import copy
import re



GLOBAL_TOKEN_STATS = {
    'in_tokens': 0,
    'out_tokens': 0,
    'total_tokens': 0,
    'calls': 0
}

def build_agent(cfg, model_name: str, T: float, sleep: float, max_completion_tokens: int | None = None):
    agent = Agent(model_name, cfg.name, T, sleep, max_completion_tokens=max_completion_tokens)
    agent.set_meta_prompt(cfg.meta_prompt)
    return agent

class Agent:

    def __init__(self, model_name: str, name: str,
                 temperature: float, sleep_time: float = 0,
                 max_completion_tokens: int | None = None) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.max_completion_tokens = (config.BATCH_CONFIG["max_completion_tokens"]
                                      if max_completion_tokens is None else max_completion_tokens)
        self.system_prompt = ""
        self.token_history = []  # ✅ 改为实例变量

        self.token_stats = {
            'in_tokens': 0,
            'out_tokens': 0,
            'total_tokens': 0,
            'calls': 0
        }

    def _limit_tokens(self, max_tokens: int) -> int:
        """限制token数量在合理范围内"""
        return max(1, min(max_tokens, self.max_completion_tokens))

    def _make_openai_request(self, messages: list, max_tokens: int, temperature: float) -> str:
        """发送OpenAI请求并处理响应"""

        resp = chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = None

        usage = resp.get('usage') if isinstance(resp, dict) else getattr(resp, 'usage', None)

        if isinstance(usage, dict):
            prompt_t = int(usage.get('prompt_tokens', 0) or 0)
            comp_t = int(usage.get('completion_tokens', 0) or 0)
            total_t = int(usage.get('total_tokens', prompt_t + comp_t) or (prompt_t + comp_t))
        else:
            prompt_t = int(getattr(usage, 'prompt_tokens', 0) or 0)
            comp_t = int(getattr(usage, 'completion_tokens', 0) or 0)
            total_t = int(getattr(usage, 'total_tokens', prompt_t + comp_t) or (prompt_t + comp_t))

        self.token_stats['in_tokens'] += prompt_t
        self.token_stats['out_tokens'] += comp_t
        self.token_stats['total_tokens'] += total_t
        self.token_stats['calls'] += 1

        GLOBAL_TOKEN_STATS['in_tokens'] += prompt_t
        GLOBAL_TOKEN_STATS['out_tokens'] += comp_t
        GLOBAL_TOKEN_STATS['total_tokens'] += total_t
        GLOBAL_TOKEN_STATS['calls'] += 1

        return resp["choices"][0]["message"]["content"]

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError),
        max_tries=20
    )
    def query(self, messages: list, max_tokens: int, temperature: float) -> str:
        time.sleep(self.sleep_time)
        limited_tokens = self._limit_tokens(max_tokens)
        return self._make_openai_request(messages, limited_tokens, temperature)

    def set_meta_prompt(self, prompt: str):
        self.system_prompt = prompt

    @trace_operation("segmentation")
    def segment_evidences(self, case_id: str, statements_summary: list, num_stages: int = 3, max_retries: int = 5) -> list:
        """
        SWER: Stage-Guided Evidence Segmentation (SGES)
        根据阶段指令 (Π) 与文本证据序列，识别 T-1 个转换边界，将证据分割为连续且不重叠的子链。
        带有最多 5 次重试机制，若 5 次全部失败则抛出 RuntimeError 触发案例跳过。
        """
        actual_indices = sorted([
            item["evidence_index"] if isinstance(item, dict) and "evidence_index" in item else idx
            for idx, item in enumerate(statements_summary)
        ])
        if len(actual_indices) < num_stages:
            return [[index] for index in actual_indices] + [[] for _ in range(num_stages - len(actual_indices))]
        indices_str = ", ".join(map(str, actual_indices))

        stage_instructions = """
- Stage 1 (主张厘清阶段 / Claim Clarification): 
  焦点在于交易背景与初始事实构建。优先归论商品原始发布描述、买家初始退款诉求、卖家首次拒绝理由以及基础订单履约事实。
- Stage 2 (争议核验阶段 / Evidence Contestation): 
  焦点在于争议焦点的交叉质证与细节核验。优先归论双方聊天记录、商品质检报告/细查图视频、瑕疵隐瞒/虚构言论质询及抗辩过程。
- Stage 3 (责任归纳阶段 / Responsibility Consolidation): 
  焦点在于平台规则对齐与终审责任判定。优先归论履约结果总结、平台规则适用性说明、补充协商结论及终审性的主张汇总。
        """
        prompt = f"""你是一个专业的纠纷证据结构化分析助手。
你的任务是将下面给出的按时间顺序排列的 {len(statements_summary)} 条纠纷证据序列，严格动态分割为刚好 {num_stages} 个连续、不重叠的阶段子链。

【阶段切分指导原则】
{stage_instructions}

【待划分的全部证据索引】
[{indices_str}]

【证据序列清单】
{json.dumps(statements_summary, ensure_ascii=False, indent=2)}

【任务要求】
1. 必须按照实际提供的待划分证据索引集合按顺序严格划分为3个连续递增的非空子区间，绝不允许只生成2个或4个，必须生成3个。
2. 注意：严格不要照抄示例！输出的子区间必须将上述可用索引集合中的所有索引完整覆盖、不重叠、无遗漏。
3. 绝对严禁缩写：每个子列表必须列出所有具体的索引数字！
   ❌ 严禁输出闭区间缩写，例如严禁写成 [2, 9]！
   ✅ 必须完整枚举区间内的所有索引，例如必须写成 [2, 3, 4, 5, 6, 7, 8, 9]！
4. 确保每个阶段至少包含 1 条证据，总计正好划分成 {num_stages} 个子列表。
5. 只返回标准的 JSON 格式，严格不要包含思维过程、Markdown 标记或多余文字。

输出格式示例:
{{
    "boundaries": [
        [0, 1],
        [2, 3, 4],
        [5, 6, 7]
    ]
}}
"""
        messages = [{"role": "user", "content": prompt}]
        retry_delay = 2.0

        expected_indices = sorted([
            item["evidence_index"] if isinstance(item, dict) and "evidence_index" in item else idx
            for idx, item in enumerate(statements_summary)
        ])

        for attempt in range(1, max_retries + 1):
            response_text = ""
            try:
                response_text = self.query(messages, **config.STAGE_REQUEST_CONFIG["segmentation"])
                if not response_text:
                    raise ValueError("模型返回为空")

                raw_clean = response_text.strip()

                raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
                raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean, flags=re.MULTILINE)
                raw_clean = re.sub(r"```\s*$", "", raw_clean, flags=re.MULTILINE).strip()

                if '"boundaries": [' in raw_clean and not '"boundaries": [[' in raw_clean:
                    raw_clean = re.sub(r'"boundaries"\s*:\s*\[\s*(\d+)', r'"boundaries": [[\1', raw_clean)

                while raw_clean.endswith("]}") and raw_clean.count("{") < raw_clean.count("}"):
                    raw_clean = raw_clean[:-1].strip()

                json_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
                if not json_match:
                    raise ValueError(f"模型未返回 JSON 结构: {response_text[:150]}")

                parsed_json_str = json_match.group().strip()
                while parsed_json_str.endswith("]}") and parsed_json_str.count("{") < parsed_json_str.count("}"):
                    parsed_json_str = parsed_json_str[:-1].strip()

                try:
                    res = json.loads(parsed_json_str)
                except json.JSONDecodeError:
                    array_match = re.search(r'"boundaries"\s*:\s*(\[\s*\[.*?\]\s*\])', raw_clean, re.DOTALL)
                    if array_match:
                        res = {"boundaries": json.loads(array_match.group(1))}
                    else:
                        raise

                boundaries = res.get("boundaries", [])

                ALLOW_AUTO_PAD_EMPTY_STAGE = True 

                if ALLOW_AUTO_PAD_EMPTY_STAGE:
                    if isinstance(boundaries, list) and len(boundaries) == (num_stages - 1):
                        print(f"      💡 [容错提示] 检测到模型仅生成了 {len(boundaries)} 个阶段，已自动追加空阶段补齐至 {num_stages} 阶段。")
                        boundaries.append([])

                if isinstance(boundaries, list) and len(boundaries) == num_stages:
                    all_indices = [idx for chunk in boundaries if isinstance(chunk, list) for idx in chunk]
                    if sorted(all_indices) == expected_indices:
                        return boundaries
                    else:
                        raise ValueError(f"返回索引完备性校验失败: {boundaries}")
                else:
                    raise ValueError(f"返回段数不符合要求 (期待 {num_stages} 段): {boundaries}")

            except Exception as e:
                print(f"\n🔍 [Debug SGES 原始输出]:\n{response_text}\n{'-'*50}")
                print(f"      ⚠️ [SGES 动态分割] 第 {attempt}/{max_retries} 次尝试失败: {e}")
                if attempt < max_retries:
                    if response_text:
                        messages = [messages[0],
                            {"role": "assistant", "content": response_text},
                            {"role": "user", "content": (
                                f"上一次结果未通过校验：{e}\n"
                                f"合法证据索引只有 [{indices_str}]。每个索引必须恰好出现一次，"
                                f"不能缺失、重复或添加其他索引。请按照原要求重新生成 {num_stages} 个"
                                "连续且不重叠的阶段子链，只返回完整的 JSON。"
                            )}]
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"💥 [案例跳过] SGES 动态证据分割 API 连续 {max_retries} 次调用彻底失败。")

        raise RuntimeError("💥 [案例跳过] SGES 动态证据分割未能成功完成。")

    def _get_scoped_evidences(self, case_data: dict) -> list:
        """读取当前阶段可见证据，并保留原始证据索引。"""
        case_details = case_data.get("case_details", {}) if isinstance(case_data, dict) else {}
        if isinstance(case_details, dict) and isinstance(case_details.get("evidences"), list):
            all_evidences = case_details["evidences"]
        elif isinstance(case_data, dict) and isinstance(case_data.get("evidences"), list):
            all_evidences = case_data["evidences"]
        else:
            all_evidences = []

        evidence_indices = case_data.get("_evidence_indices") if isinstance(case_data, dict) else None
        if evidence_indices is None:
            evidence_indices = list(range(len(all_evidences)))

        scoped_evidences = []
        for original_idx in evidence_indices:
            if not isinstance(original_idx, int) or original_idx < 0 or original_idx >= len(all_evidences):
                continue
            ev_copy = copy.deepcopy(all_evidences[original_idx])
            ev_copy["_original_evidence_index"] = original_idx
            scoped_evidences.append(ev_copy)
        return scoped_evidences

    @trace_operation("evidence_exploration")
    def _cot_active_exploration_loop(
        self, phase: str, side: str, case_data: dict, *,
        retrieval_memory: list, perception_memory: list
    ) -> dict:
        """结合自身与邻居历史检索、感知，用感知缺口指导下一次检索。"""

        evidences = self._get_scoped_evidences(case_data)
        case_id = str(case_data.get("id", "unknown_id"))

        clean_history_ctx = self._prepare_memory_context(retrieval_memory)

        final_evidence_content = {
            "text": "",
            "images": [],
            "image_descriptions": []
        }

        for idx, ev in enumerate(evidences):
            raw_identity = ev.get("identity", "")

            if raw_identity not in ["买家", "卖家"]:
                continue

            content_dict = ev.get("content", {}) or ev
            raw_txt = content_dict.get("text", "").strip() if content_dict.get("text") else ""
            if raw_txt:
                final_evidence_content["text"] += f"【{raw_identity}举证】: {raw_txt}\n"

        analyzed_evidence_ids: set = set()

        perceived_insights = []
        retrieval_feedback = None
        all_accumulated_images = []

        max_iterations = 3
        current_iteration = 1

        while current_iteration <= max_iterations:

            res_tuple = self._cot_select_evidence_interleaved(
                clean_history_ctx,    # 1. clean_history
                evidences,            # 2. evidences
                side,                 # 3. side
                phase,                # 4. phase
                analyzed_evidence_ids,# 5. analyzed_ids
                retrieval_memory,      # 6. shared_memory
                case_id,               # 7. case_id
                current_iteration,      # 8. current_iteration (新增：用于Iteration区分)
                perceived_insights,
                retrieval_feedback=retrieval_feedback,
            )

            if isinstance(res_tuple, (tuple, list)) and len(res_tuple) == 2:
                selected_ev, ev_index = res_tuple
            else:
                selected_ev, ev_index = None, None

            if ev_index is None or ev_index in analyzed_evidence_ids:
                print("💡 [提示] 无更多可供选择的新物证，提前终止探索。")
                break

            analyzed_evidence_ids.add(ev_index)

            content_media = selected_ev.get("content", {}) or selected_ev
            if content_media.get("text"):
                evidence_raw_text = content_media.get("text", "").strip()
            else:
                evidence_raw_text = ""

            current_ev_identity = str(selected_ev.get("identity", "")).strip()
            if current_ev_identity == "买方":
                current_ev_identity = "买家"
            if current_ev_identity == "卖方":
                current_ev_identity = "卖家"

            print(f"🔎 [{phase}/{side}] 第{current_iteration}轮选择证据索引: {ev_index}, 证据身份: {current_ev_identity}")

            is_sufficient, new_insight, selected_base64_list, retrieval_feedback = self._cot_perceive_visual_details_interleaved(
                selected_evidence=selected_ev,
                raw_text=evidence_raw_text,
                current_iteration=current_iteration,
                perceived_insights=perceived_insights,
                side=side,
                case_id=case_id,
                phase=phase,
                shared_memory=perception_memory,
                retrieval_feedback=retrieval_feedback,
            )

            if new_insight:
                formatted_insight = f"[针对【{current_ev_identity}举证】(物证{ev_index})的感知线索]: {new_insight}"
                perceived_insights.append(formatted_insight)
                final_evidence_content["image_descriptions"].append(formatted_insight)

            if selected_base64_list:
                all_accumulated_images.extend(selected_base64_list)

            if is_sufficient:
                print("      感知线索已充足，仍按每阶段三次检索规则执行。")
            else:
                print(f"      感知不足：{retrieval_feedback['insufficiency_reason']}")
                print(f"      后续检索目标：{retrieval_feedback['next_retrieval_goal']}")
            current_iteration += 1

        return {
            "text": final_evidence_content.get("text", ""),
            "images": copy.deepcopy(all_accumulated_images),
            "image_descriptions": copy.deepcopy(final_evidence_content.get("image_descriptions", []))
        }

    @trace_operation("retrieval")
    def _cot_select_evidence_interleaved(self, clean_history: list, evidences: list, side: str, phase: str, analyzed_ids: set, shared_memory: list, case_id: str = "unknown_id", current_iteration: int = 1, perceived_insights: list = None, retrieval_feedback: dict = None) -> tuple:

        labeled_evidences_pool = []
        for idx, ev in enumerate(evidences):
            evidence_idx = ev.get("_original_evidence_index", idx)
            raw_identity = str(ev.get("identity", "未知")).strip()
            content_dict = ev.get("content", {}) or {}
            if content_dict.get("text"):
                raw_text = content_dict.get("text", "").strip()
            else:
                raw_text =  ""

            image_count = len(content_dict.get("image_paths", [])) if content_dict.get("image_paths") else 0
            video_count = len(content_dict.get("video_paths", [])) if content_dict.get("video_paths") else 0

            if image_count == 0 and video_count == 0:
                continue

            if image_count > 0 and video_count > 0:
                media_label = "【图+视频综合物证】"
            elif image_count > 0:
                media_label = "【纯图物证】"
            else:
                media_label = "【纯视频物证】"

            labeled_evidences_pool.append({
                "evidence_index": evidence_idx,
                "raw_ev_reference": ev,
                "identity": {"买方": "买家", "卖方": "卖家"}.get(raw_identity, raw_identity),
                "label_info": {
                    "evidence_index": evidence_idx,
                    "证据所属": f"【{raw_identity}物证】",
                    "文本证据": raw_text if raw_text else "无文字说明，完全依赖多模态视觉呈现",
                    "多模态标签": media_label,
                    "包含媒体文件数量": f"图片 {image_count} 张, 视频 {video_count} 个"
                }
            })

        available_evidences = [
            item for item in labeled_evidences_pool
            if item["evidence_index"] not in analyzed_ids
        ]
        if not available_evidences:
            return None, None

        # 每个阶段依次检索买家、卖家、自由证据；指定方无未审阅候选时回退。
        required_identity = {1: "买家", 2: "卖家"}.get(current_iteration)
        selection_scope = "从全部未审阅候选中自由选择。"
        if required_identity:
            party_evidences = [
                item for item in available_evidences
                if item["identity"] == required_identity
            ]
            if party_evidences:
                available_evidences = party_evidences
                selection_scope = f"只从{required_identity}的未审阅候选中选择。"
            else:
                selection_scope = f"{required_identity}无未审阅候选，回退为自由选择。"

        available_manifest = [
            {**item["label_info"], "审阅状态标签": "【未审阅，本轮可选】"}
            for item in available_evidences
        ]

        phase_str = str(phase).lower()
        reviewed_list = list(analyzed_ids) if analyzed_ids else []

        if "1" in phase_str or "round_1" in phase_str:
            tactical_guidance = """你是第一轮讨论的证据筛选助手。
                任务：从初始提交段的买卖双方多模态证据中，选择一条最能帮助理解案件基础事实的证据。
                选择标准：
                1. 优先选择能说明交易商品、双方核心争议或初始承诺的证据；
                2. 保持绝对中立，严禁为了偏袒某一方而忽略关键信息；
                3. 避免选择信息增量很低或只重复案情的证据；
                4. 只返回一个最值得视觉分析的 evidence_index。"""

        elif "2" in phase_str or "round_2" in phase_str:
            reviewed_list = list(analyzed_ids) if analyzed_ids else []
            tactical_guidance = f"""你是第二轮讨论的案件证据筛选助手。
                任务：从中间段的买卖双方多模态证据中，选择一条最有助于验证或反驳当前争议点的证据。
                选择标准：
                1. 优先选择能直接检验关键争议点真伪的客观证据；
                2. 优先选择聊天记录、商品描述、退款/拒绝理由、图片或视频中的可核查细节；
                3. 避免选择只能重复既有结论、信息增量很低的证据；
                4. 不要扩展无证据支持的新理由；
                5. 只返回一个最值得视觉分析的 evidence_index。
                已审阅证据索引：{reviewed_list}"""

        else:
            tactical_guidance = """你是第三轮讨论的证据筛选助手。
                任务：从最后段的买卖双方多模态证据中，选择一条最有助于形成最终判断的证据。
                选择标准：
                1. 优先选择能补充最终事实链、责任归属或履约情况的证据；
                2. 如果证据与前文重复，降低优先级；
                3. 不要扩展无证据支持的新理由；
                4. 只返回一个最值得视觉分析的 evidence_index。"""

        accumulated_insights_str = "无（当前为本轮第1次选择）"
        if perceived_insights and len(perceived_insights) > 0:
            accumulated_insights_str = "\n".join(f"- {item}" for item in perceived_insights)

        prompt = f"""你是中立的证据筛选助手。
                你的任务是从候选证据中选择一条最值得进一步视觉分析的多模态证据。每个阶段的第1次检索选择买家证据，第2次检索选择卖家证据，第3次自由选择。指定方没有未审阅候选时才回退为自由选择。
                当前为本阶段第{current_iteration}次检索：{selection_scope}只能返回下方可选清单中的 evidence_index。
                            
                【阶段任务】
                {tactical_guidance}

                【本轮前几次检索中已发掘的关键线索】
                {accumulated_insights_str}

                【上一次感知的不足与后续检索目标】
                {json.dumps(retrieval_feedback, ensure_ascii=False, indent=2) if retrieval_feedback else "无（本阶段尚未进行感知）"}
                若有反馈，优先选择能核验上述缺口、补充所需信息的证据。缺口和检索目标不是已观察事实。仍须遵守本次提交方限制，只能从下方未审阅候选中选择，不得假定候选一定能证明某个结论。
                            
                【历史辩论内容】
                {json.dumps(clean_history, ensure_ascii=False, indent=2)}
                    
                【本轮可选证据清单】
                {json.dumps(available_manifest, ensure_ascii=False, indent=2)}
                    
                请只输出 JSON，不要输出解释、Markdown 或多余文字：
                {{
                "selected_index": int,
                "strategic_reason": "不超过30字，说明选择原因"
                }}
                """
        messages = [
            {"role": "user", "content": prompt}
        ]

        max_retries = 5
        retry_delay = 2.0
        resp = None

        valid_evidences = {
            item["evidence_index"]: item["raw_ev_reference"]
            for item in available_evidences
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = chat_completion(
                    model=self.model_name,
                    messages=messages,
                    **config.STAGE_REQUEST_CONFIG["selection"],
                )
                resp = response["choices"][0]["message"]["content"]
                if not resp:
                    raise ValueError("模型返回为空")

                json_match = re.search(r'\{.*\}', resp, re.DOTALL)
                if not json_match:
                    raise ValueError(f"模型未返回 JSON: {resp[:200]}")

                # Only the explicit integer is consumed here. Keep the original reply
                # in the trace even if the unused reason contains unescaped quotes.
                selections = re.findall(r'"selected_index"\s*:\s*(-?\d+)(?=\s*[,}])', json_match.group())
                if len(selections) != 1:
                    raise ValueError("模型必须返回唯一的整数 selected_index")
                chosen_idx = int(selections[0])

                if chosen_idx not in valid_evidences:
                    raise ValueError(f"所选证据索引不在本轮候选中: {chosen_idx}")
                return valid_evidences[chosen_idx], chosen_idx

            except Exception as e:
                print(f"      ⚠️ 步骤一 API 调用异常 [第 {attempt}/{max_retries} 次尝试失败]: {e}")
                if attempt < max_retries:
                    print(f"      ⏳ 正在等待 {retry_delay} 秒后重新重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Exception:", e)
                    raise RuntimeError(f"步骤一连续 {max_retries} 次调用或证据选择失败。") from e

    @trace_operation("perception")
    def _cot_perceive_visual_details_interleaved(self, selected_evidence: dict, raw_text: str, current_iteration: int, perceived_insights: list, side: str, case_id: str, phase: str, shared_memory: list = None, retrieval_feedback: dict = None) -> tuple:
        """【交织重构-步骤二】复刻 8 帧核心分析提示词，高密度画面价值筛选与充分性裁决"""

        content_media = selected_evidence.get("content", {}) or {}
        image_paths = content_media.get("image_paths", []) or []
        video_paths = content_media.get("video_paths", []) or []

        visual_payload_pool = []
        media_mapping_manifest = []

        for idx, img_p in enumerate(image_paths):
            norm_p = img_p.replace("\\", "/")

            if not os.path.exists(norm_p):
                raise ValueError(f"静态图片不存在: {norm_p}")

            base64_img = self.encode_image(norm_p)
            if not base64_img:
                raise ValueError(f"静态图片编码失败: {norm_p}")

            image_format = norm_p.lower().split('.')[-1]
            image_format = 'jpeg' if image_format == 'jpg' else image_format
            visual_payload_pool.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{base64_img}"}
            })
            media_mapping_manifest.append({
                "payload_index": len(visual_payload_pool) - 1,
                "media_type": "image",
                "media_index": idx,
                "origin_path": norm_p,
                "timestamp": "0"
            })

        if len(video_paths) > 0:
            for v_idx, vid_p in enumerate(video_paths):
                norm_vid_p = vid_p.replace("\\", "/")
                video_payloads, video_mappings = self._build_video_payloads(
                    video_path=norm_vid_p,
                    media_index=v_idx,
                    case_id=case_id,
                    start_payload_index=len(visual_payload_pool),
                    max_frames=config.MAX_VIDEO_FRAMES
                )
                visual_payload_pool.extend(video_payloads)
                media_mapping_manifest.extend(video_mappings)

        num_images = len(image_paths)
        num_videos = len(video_paths)

        visual_info = []
        if num_images > 0:
            visual_info.append(f"{num_images}张图片")
        if num_videos > 0:
            visual_info.append(f"{num_videos}个视频（每个视频已提取8帧关键帧）")
        visual_desc = "、".join(visual_info)

        clean_history = self._prepare_memory_context(shared_memory)
        stage_index = {"round_1": 0, "round_2": 1, "round_3": 2}[phase.lower()]
        stage_instruction = STAGE_PROMPTS[stage_index]

        prompt = f"""你正在分析一条纠纷证据的视觉内容，并执行关键画面筛选。

            【当前阶段指导】:
            {stage_instruction}
            上述指导用于确定本次事实提取的重点。本步骤只提取与当前阶段相关的可核验事实及其对双方主张的影响，不在此输出最终裁决，不为符合阶段目标而推测画面中不存在的事实。

            【自己与关注邻居在此前阶段的历史意见】:
            {json.dumps(clean_history, ensure_ascii=False, indent=2)}

            【本阶段此前检索已发现的多模态线索】:
            {json.dumps(perceived_insights, ensure_ascii=False, indent=2)}

            【上一次感知提出的缺口与检索目标】:
            {json.dumps(retrieval_feedback, ensure_ascii=False, indent=2) if retrieval_feedback else "无（本阶段首次感知）"}

            当前分析的证据文本描述：{raw_text}

            本次的视觉内容：{visual_desc}

            【任务要求】
            1. 只记录画面中能直接看到的事实，不要猜测。
            2. 优先提取与当前争议焦点最相关、最能还原事实真相的客观画面。
            3. 同时客观判断该事实对买家/卖家分别是支持、削弱或中性。
            4. 结合本次证据、本阶段已有线索和历史意见，判断现有感知是否足以支撑本阶段裁决。历史意见仅供参考，不能作为本次画面中已观察到的事实。
            5. 若仍有关键事实无法确认、证据冲突尚未解决或画面不清，is_sufficient 填布尔值 false。insufficiency_reason 必须说明当前还缺少什么以及为何不足，next_retrieval_goal 必须说明下一次需要核验的具体问题和所需证据类型，不能只写“需要更多证据”。
            6. 重新检查上次提出的缺口是否已由当前证据解决，只保留仍未解决的缺口及新的检索目标。若现有感知已充足，is_sufficient 填布尔值 true，insufficiency_reason 和 next_retrieval_goal 填空字符串。
            7. summary 和 visual_findings 只描述已提取的事实线索，不要把尚缺的证据或后续检索目标写成已证实的事实。
            8. 每张画面前附有媒体定位标签。media_type、media_index 和 timestamp 必须使用该画面的标签值，图片和视频分别从 0 编号。

            请以唯一的 JSON 格式输出：
            {{
            "visual_findings": [
                {{
                "media_type": "image/video",
                "media_index": int,
                "timestamp": "image图填0，视频填MM:SS",
                "fact": "不超过35字",
                "buyer_effect": "支持/削弱/中性",
                "seller_effect": "支持/削弱/中性"
                }}
            ],
            "summary": "不超过60字",
            "is_sufficient": false,
            "insufficiency_reason": "仍未解决的证据缺口及不足原因，充足时为空",
            "next_retrieval_goal": "下一次需核验的具体问题和所需证据类型，充足时为空"
            }}
        """
        multimodal_content = [{"type": "text", "text": prompt}]
        for mapping in media_mapping_manifest:
            multimodal_content.append({
                "type": "text",
                "text": (
                    f"media_type={mapping['media_type']}, "
                    f"media_index={mapping['media_index']}, "
                    f"timestamp={mapping['timestamp']}"
                )
            })
            multimodal_content.append(visual_payload_pool[mapping["payload_index"]])

        messages = [
            {"role": "user", "content": multimodal_content}

        ]

        max_retries = 5
        retry_delay = 2.0

        for attempt in range(1, max_retries + 1):
            resp = None
            try:
                response = chat_completion(
                    model=self.model_name,
                    messages=messages,
                    **config.STAGE_REQUEST_CONFIG["perception"],
                )
                resp = response.choices[0].message.content
                record_event("media_mapping", mapping=media_mapping_manifest, payloads=visual_payload_pool)
                return self._parse_visual_response(resp, media_mapping_manifest, visual_payload_pool)

            except Exception as e:
                print(f"      ⚠️ 步骤二 API/解析异常 [第 {attempt}/{max_retries} 次尝试失败]: {e}")
                if attempt < max_retries:
                    print(f"      ⏳ 正在等待 {retry_delay} 秒后重新重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

                else:
                    raise RuntimeError(f"步骤二多模态感知连续 {max_retries} 次请求或结果校验失败：{e}") from e

        raise RuntimeError("🚨 [案例跳过] 步骤二未能成功完成多模态视觉筛选判定。")

    def _parse_visual_response(self, resp: str, media_mapping_manifest: list, visual_payload_pool: list) -> tuple:
        """解析事实与画面，并单独返回充分性判断和后续检索反馈。"""
        if not resp:
            raise ValueError("步骤二大模型返回为空")

        json_match = re.search(r'\{.*\}', resp, re.DOTALL)
        if not json_match:
            raise ValueError(f"步骤二大模型未返回 JSON: {resp[:200]}")

        res = load_model_json(json_match.group())
        is_sufficient = res.get("is_sufficient")
        if not isinstance(is_sufficient, bool):
            raise ValueError("is_sufficient 必须是 JSON 布尔值")
        retrieval_feedback = {
            key: res.get(key, "")
            for key in ("insufficiency_reason", "next_retrieval_goal")
        }
        if any(not isinstance(value, str) for value in retrieval_feedback.values()):
            raise ValueError("感知不足原因和后续检索目标必须是字符串")
        if not is_sufficient and any(not value.strip() for value in retrieval_feedback.values()):
            raise ValueError("感知不足时必须说明不足原因和后续检索目标")
        filtered_findings_summary = []
        selected_base64_pool = []

        findings = res.get("visual_findings", []) or []
        for find in findings:
            m_type = find.get("media_type", "")
            m_idx = find.get("media_index")
            m_ts = find.get("timestamp")
            observed_fact = (find.get("fact") or find.get("observed_fact") or "").strip()
            buyer_effect = find.get("buyer_effect", "").strip()
            seller_effect = find.get("seller_effect", "").strip()

            if observed_fact:
                effect_text = f"买家:{buyer_effect or '中性'}; 卖家:{seller_effect or '中性'}"
                filtered_findings_summary.append(f"事实:{observed_fact}；影响:{effect_text}")

            matched_payload_idx = self._match_visual_payload_index(
                media_mapping_manifest,
                media_type=m_type,
                media_index=m_idx,
                timestamp=m_ts
            )
            if matched_payload_idx is not None:
                if matched_payload_idx < 0 or matched_payload_idx >= len(visual_payload_pool):
                    raise ValueError(f"模型返回的画面索引越界: {matched_payload_idx}")

                item_payload = visual_payload_pool[matched_payload_idx]
                if item_payload not in selected_base64_pool:
                    selected_base64_pool.append(item_payload)
                    matched_mapping = next(
                        (m for m in media_mapping_manifest if m.get("payload_index") == matched_payload_idx),
                        {}
                    )
                    origin_path = matched_mapping.get("origin_path", "unknown")
                    print(f" 已匹配 {m_type} (媒体索引:{m_idx}/时间戳:{m_ts}/payload:{matched_payload_idx}) -> {origin_path}")

        summary_text = res.get("summary")

        insight_parts = []
        if summary_text:
            insight_parts.append(f"摘要:{str(summary_text).strip()}")
        if filtered_findings_summary:
            insight_parts.append("关键事实:" + " | ".join(filtered_findings_summary))
        combined_insight_str = "\n".join(insight_parts)

        return is_sufficient, combined_insight_str, selected_base64_pool, retrieval_feedback

    def _match_visual_payload_index(
        self,
        media_mapping_manifest: list,
        media_type: str,
        media_index,
        timestamp: str
    ) -> Optional[int]:
        """按媒体编号定位图片，按视频编号和精确时间戳定位视频帧。"""
        media_type = str(media_type).strip()
        index_text = str(media_index).strip()
        if not index_text.isdecimal():
            return None
        media_index = int(index_text)

        if media_type == "video":
            timestamp = str(timestamp).strip()
            if timestamp == "0":
                timestamp = "00:00"
            parts = re.fullmatch(r"(\d+):([0-5]?\d)", timestamp)
            if not parts:
                return None
            timestamp = f"{int(parts[1]):02d}:{int(parts[2]):02d}"

        for mapping in media_mapping_manifest:
            if str(mapping.get("media_type", "")).strip() != media_type:
                continue
            if mapping.get("media_index") != media_index:
                continue
            if media_type == "image":
                return mapping.get("payload_index")
            if media_type == "video" and mapping.get("timestamp") == timestamp:
                return mapping.get("payload_index")

        return None

    def _get_uniform_video_timestamps(self, video_path: str, max_frames: int = config.MAX_VIDEO_FRAMES) -> list:
        """根据视频真实时长均匀生成 MM:SS 抽帧时间戳。"""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps <= 0 or frame_count <= 0:
                raise ValueError(f"无法获取视频时长信息: {video_path}")

            duration_seconds = frame_count / fps
            if duration_seconds <= 0:
                raise ValueError(f"视频时长无效: {video_path}")

            sample_count = min(max_frames, max(1, int(duration_seconds) + 1))
            if sample_count == 1:
                sample_seconds = [0]
            else:
                last_second = max(0, duration_seconds - 1.0)
                sample_seconds = [
                    round(i * last_second / (sample_count - 1), 2)
                    for i in range(sample_count)
                ]

            timestamps = []
            for sec in sample_seconds:
                total_seconds = int(sec)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                timestamp = f"{minutes:02d}:{seconds:02d}"
                if timestamp not in timestamps:
                    timestamps.append(timestamp)

            return timestamps
        finally:
            if cap is not None:
                cap.release()

    def _build_video_payloads(
        self,
        video_path: str,
        media_index: int,
        case_id: str,
        start_payload_index: int,
        max_frames: int = config.MAX_VIDEO_FRAMES
    ) -> tuple:
        """均匀抽取视频帧，并构建多模态 payload 与索引映射。"""
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")

        sample_timestamps = self._get_uniform_video_timestamps(video_path, max_frames=max_frames)
        if not sample_timestamps:
            raise ValueError(f"视频无法生成抽帧时间戳: {video_path}")

        video_payloads = []
        video_mappings = []
        for ts in sample_timestamps:
            frame_p = self._extract_video_frame(video_path, ts, case_id=case_id)
            if not frame_p or not os.path.exists(frame_p):
                print(f"    ⚠️  跳过无法抽取的视频帧: {video_path} @ {ts}")
                continue

            base64_frame = self.encode_image(frame_p)
            if not base64_frame:
                print(f"    ⚠️  跳过无法编码的视频帧: {frame_p}")
                continue

            payload_index = start_payload_index + len(video_payloads)
            video_payloads.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
            })
            video_mappings.append({
                "payload_index": payload_index,
                "media_type": "video",
                "media_index": media_index,
                "origin_path": frame_p,
                "timestamp": ts
            })

        if not video_payloads:
            raise ValueError(f"视频未成功抽取任何可用帧: {video_path}")

        return video_payloads, video_mappings

    def _extract_video_frame(self, video_path: str, timestamp: str, case_id: str = "unknown") -> Optional[str]:
        """
        从视频中提取指定时间戳的帧并保存为图片。

        Args:
            video_path: 视频文件路径。
            timestamp: 时间戳字符串，格式为 "MM:SS"。

        Returns:
            保存的帧图片路径，如果失败则返回 None。
        """
        try:
            parts = timestamp.split(':')
            if len(parts) != 2:
                print(f"    ⚠️  无效的时间戳格式: {timestamp}")
                return None

            minutes = int(parts[0])
            seconds = int(parts[1])
            time_in_seconds = minutes * 60 + seconds

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    ⚠️  无法打开视频文件: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print(f"    ⚠️  无法获取视频FPS，将使用默认值30: {video_path}")
                fps = 30

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_number = int(time_in_seconds * fps)
            if frame_count > 0:
                frame_number = min(frame_number, max(frame_count - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"    ⚠️  无法在时间 {timestamp} 提取帧: {video_path}")
                return None

            target_dir = Path('frame') / case_id
            target_dir.mkdir(parents=True, exist_ok=True)

            frame_filename = target_dir / f"frame_{uuid4().hex}.jpg"
            if not cv2.imwrite(str(frame_filename), frame):
                return None

            print(f"      → 已提取视频帧: {frame_filename} (来自 {os.path.basename(video_path)} @ {timestamp})")
            return str(frame_filename)

        except Exception as e:
            print(f"    ⚠️  提取视频帧时发生错误: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return None

    def encode_image(self, image_path: str) -> str:
        """
        将本地图片文件编码为 base64 字符串
        
        Args:
            image_path: 图片文件的本地路径
            
        Returns:
            base64 编码字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _prepare_memory_context(self, shared_memory: list) -> list:
        """Preserve the scoped self/neighbor messages without summarizing or dropping roles."""
        return copy.deepcopy(shared_memory or [])

    def ask(self, shared_memory: list, prompt: str, temperature: float = None):
        memory_ctx = self._prepare_memory_context(shared_memory)
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + memory_ctx
            + [{"role": "user", "content": prompt}]
        )

        max_tokens, token_list, total_sum = self._calculate_max_tokens(messages)
        effective_temperature = temperature if temperature is not None else self.temperature

        response = self.query(messages, max_tokens, effective_temperature)

        return response, token_list, total_sum

    def ask_with_content(self, shared_memory: List[Dict], content: List[Dict], *, temperature: float, max_retries: int = 3) -> str:
        """发送结构化内容，保留任务请求参数和用量统计。"""

        cleaned_memory = self._prepare_memory_context(shared_memory)

        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + cleaned_memory
            + [{"role": "user", "content": copy.deepcopy(content)}]
        )

        max_tokens, token_list, total_sum = self._calculate_max_tokens(messages)

        for attempt in range(max_retries):
            try:

                response = chat_completion(
                    model=self.model_name,
                    messages=messages,  # 原始数据，完全未受打印代码影响
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **config.STAGE_REQUEST_CONFIG["multimodal_verdict"],
                )

                usage = response.get('usage') if isinstance(response, dict) else getattr(response, 'usage', None)
                if isinstance(usage, dict):
                    prompt_t = int(usage.get('prompt_tokens', 0) or 0)
                    comp_t = int(usage.get('completion_tokens', 0) or 0)
                    total_t = int(usage.get('total_tokens', prompt_t + comp_t) or (prompt_t + comp_t))
                else:
                    prompt_t = int(getattr(usage, 'prompt_tokens', 0) or 0)
                    comp_t = int(getattr(usage, 'completion_tokens', 0) or 0)
                    total_t = int(getattr(usage, 'total_tokens', prompt_t + comp_t) or (prompt_t + comp_t))

                self.token_stats['in_tokens'] += prompt_t
                self.token_stats['out_tokens'] += comp_t
                self.token_stats['total_tokens'] += total_t
                self.token_stats['calls'] += 1

                GLOBAL_TOKEN_STATS['in_tokens'] += prompt_t
                GLOBAL_TOKEN_STATS['out_tokens'] += comp_t
                GLOBAL_TOKEN_STATS['total_tokens'] += total_t
                GLOBAL_TOKEN_STATS['calls'] += 1

                self.last_call_usage = self.token_stats.copy()
                self.last_call_usage['model'] = self.model_name

                print(f"  ➡️ Prompt Tokens: {prompt_t}, Completion Tokens: {comp_t}, Total Tokens: {total_t}")
                print(f"  📊 Agent [{self.name}] 累计 Token - 输入: {self.token_stats['in_tokens']}, 输出: {self.token_stats['out_tokens']}, 总计: {self.token_stats['total_tokens']}")
                print(f"  🌍 [Batch Total] 累计 Token - 输入: {GLOBAL_TOKEN_STATS['in_tokens']}, 输出: {GLOBAL_TOKEN_STATS['out_tokens']}, 总计: {GLOBAL_TOKEN_STATS['total_tokens']}")

                result = response.choices[0].message.content

                return result, token_list, total_sum

            except (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _calculate_max_tokens(self, messages: list) -> int:
        """手动计算消息的 token 数量，支持多模态内容，并输出总 token 数"""
        import re

        def calculate_tokens(content: str) -> int:
            """
            手动计算字符串的 token 数量。
            - 英文单词、中文字符、标点符号、空格均算作 1 个 token。
            - Base64 数据每 4 个字符算作 1 个 token。
            """
            if content.startswith("data:image/") or content.startswith("data:video/"):
                if "," in content:
                    base64_data = content.split(",", 1)[1]
                    estimated = len(base64_data) // 4
                    return min(estimated, 4096)
                return 0

            token_pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+|[^\w\s]|\s")
            tokens = token_pattern.findall(content)
            return len(tokens)

        total_tokens = 0

        try:
            for message in messages:
                content = message.get("content", "")

                if isinstance(content, str):
                    total_tokens += calculate_tokens(content)

                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                total_tokens += calculate_tokens(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                total_tokens += calculate_tokens(image_url)
                            elif item.get("type") == "video":
                                video_url = item.get("video", {}).get("url", "")
                                total_tokens += calculate_tokens(video_url)

                else:
                    total_tokens += calculate_tokens(str(content))

            self.token_history.append(total_tokens)
            current_sum = sum(self.token_history)

        except Exception as e:
            print(f"⚠️ Token计算出错: {e}")
            total_tokens = 0  # 使用保守的默认值
            self.token_history.append(total_tokens)
            current_sum = sum(self.token_history)

        max_context = config.MODEL_CONTEXT_LENGTHS.get(self.model_name, config.DEFAULT_CONTEXT_LENGTH)
        available_tokens = max_context - total_tokens

        MIN_AVAILABLE_TOKENS = 256

        if available_tokens < MIN_AVAILABLE_TOKENS:
            result = min(self.max_completion_tokens, MIN_AVAILABLE_TOKENS)
        else:
            result = min(available_tokens, self.max_completion_tokens)

        return result, self.token_history.copy(), current_sum

    def get_token_history(self):
        """获取当前实例的token记录列表"""
        return self.token_history.copy()

    def clear_token_history(self):
        """清空当前实例的token历史记录"""
        self.token_history.clear()

    def get_token_statistics(self):
        """获取当前实例的token使用统计信息"""
        if not self.token_history:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "average_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "token_list": []
            }

        total_sum = sum(self.token_history)

        return {
            "total_requests": len(self.token_history),
            "total_tokens": total_sum,
            "average_tokens": total_sum / len(self.token_history),
            "min_tokens": min(self.token_history),
            "max_tokens": max(self.token_history),
            "token_list": self.token_history.copy()
        }
