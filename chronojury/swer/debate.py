import os
from pathlib import Path
from typing import List, Dict, Any
import time
import json
import copy
import random
from . import config
from .config import (
    DEBATE_AGENTS, FREE_ROUNDS, FOLLOWING_NETWORK_OUT_DEGREE, FOLLOWING_NETWORKS_DIR,
)
from .prompts import RoleConfig, ROUND_DISCUSSION_TEMPLATE, STAGE_PROMPTS
from .agent import build_agent, Agent
from ..tracing import trace_scope, record_event

class Debate:

    def __init__(self, model_name: str, T: float, sleep: float, case_data: dict, output_dir=None,
                 max_completion_tokens: int | None = None):

        self.model_name, self.T, self.sleep = model_name, T, sleep
        self.output_dir = output_dir
        self.max_completion_tokens = (config.BATCH_CONFIG["max_completion_tokens"]
                                      if max_completion_tokens is None else max_completion_tokens)
        self.shared: List[Dict] = []
        self.transcript: List[Dict] = []
        self.round_history: List[Dict[str, Any]] = []
        self.interaction_history: List[Dict[str, Any]] = []
        self.following_network: Dict[str, List[int]] = {}
        self.agents = self._init_agents(case_data)
        self.case_data = case_data

        self.debate_token_history = []  # 整个辩论的token历史
        self.debate_total_tokens = 0    # 整个辩论的token总计

    def _init_agents(self, case_data: dict) -> Dict[str, Agent]:
        """初始化所有角色代理"""
        cfgs = self._create_role_configs(case_data)  # 传递 case_data
        return {
            c.name: build_agent(c, self.model_name, self.T, self.sleep, self.max_completion_tokens)
            for c in cfgs
        }

    def _create_role_configs(self, case_data: dict) -> List[RoleConfig]:
        """创建角色配置列表"""
        cfgs = []
        cfgs.extend(self._create_debate_configs(DEBATE_AGENTS))
        return cfgs

    def _create_debate_configs(self, agent_names: List[str]) -> List[RoleConfig]:
        """创建辩论角色配置"""

        stance = (
            f"你是一位经验丰富、逻辑严谨、绝对中立的二手交易纠纷裁决专家。你正在参加一个有{len(agent_names)}个专家的讨论，目的是裁决给定案例买家和卖家谁的诉求更合理。"
            "你需要保持客观、理性的态度，根据交易经过、沟通记录、双方提供的证据、图片等，对案件进行全面、中立的剖析。不要偏袒任何一方，仅基于客观事实和视觉证据进行发言，说明你的看法。你的分析内容不得超过200字！\n"
        )

        return [
            RoleConfig(
                name=agent_name,
                side="",
                duty="Juror",
                meta_prompt=(
                    f"You are a speaker in a discussion.\n"
                    f"{stance}"
                )
            )
            for agent_name in agent_names
        ]

    def _collect_token_data(self, token_list: list, total_sum: int):
        """收集token数据到辩论级别统计"""
        if token_list:
            current_request_tokens = token_list[-1]  # 获取当前请求的token数
            self.debate_token_history.append(current_request_tokens)
            self.debate_total_tokens = sum(self.debate_token_history)

    def run(self, *, case_data: str, case_path: Path):
        """运行完整的辩论流程"""

        assert case_data, "case_data 不能为空"
        self.case_data = case_data
        self.case_path = case_path
        self._cached_segmented_indices = None

        self._load_following_network()
        record_event("following_network", network=self.following_network)
        self._run_debate_phases(case_data)

        node_texts = self._build_node_texts_and_edges()

        saved_path = self._save_for_gnn(case_data, node_texts)

        return {
            "node_texts": node_texts,
            "edges": [],                                 # 🔒 强制锁空，不保存边关系
            "updated_node_features": None,               # 🔒 强制关闭特征编码
            "interactive_representation": None,          # 🔒 强制关闭交互向量
            "edge_index_after_gat": None,
            "edge_attention_after_gat": None,
            "classification_result": None,
            "saved_path": saved_path
        }

    def _record(self, role: str, prompt: str, reply: str):
        """记录对话到共享上下文和转录本"""
        self.shared.extend([
            {"role": "user", "content": f"{role}: {prompt}"},
            {"role": "assistant", "content": f"{role}: {reply}"}
        ])

        if self.transcript and self.transcript[-1]["speaker"] == role:
            self.transcript[-1]["text"] += "\n\n" + reply
        else:
            self.transcript.append({"speaker": role, "text": reply})

    def _ask(self, role: str, prompt: str, evidence_content: Dict = None, shared_memory: List[Dict] = None) -> str:
        agent = self.agents[role]
        memory_source = [] if shared_memory is None else shared_memory
        if evidence_content and (evidence_content.get("images") or evidence_content.get("videos")):
            # 保留含视觉线索时的请求设置，裁决输入只发送文本。
            content = [{"type": "text", "text": prompt}]
            reply, token_list, total_sum = agent.ask_with_content(
                memory_source, content, temperature=self.T, max_retries=3
            )
        else:
            reply, token_list, total_sum = agent.ask(memory_source, prompt, temperature=self.T)
        if reply is None:
            raise RuntimeError(f"No verdict returned by {role}")
        self._collect_token_data(token_list, total_sum)
        self._record(role, prompt, reply)
        return reply

    def get_all_evidences(self) -> List[Dict]:
        """获取当前案例的完整证据列表，兼容不同数据结构。"""

        if isinstance(self.case_data, dict):
            case_details = self.case_data.get("case_details", {})
            if isinstance(case_details, dict) and isinstance(case_details.get("evidences"), list):
                return case_details["evidences"]

            if isinstance(self.case_data.get("evidences"), list):
                return self.case_data["evidences"]

        return []

    def get_round_evidence_indices(self, round_idx: int) -> List[int]:
        """
        SWER - SGES 动态证据分割机制：
        使用大模型根据阶段指令 Π 识别证据时间分割边界，并将结果保存至 segment/ 路径下。
        无兜底策略，失败重试 5 次后抛出异常跳过案例。
        """
        evidences = self.get_all_evidences()
        evidence_count = len(evidences)
        if evidence_count == 0:
            return []

        if round_idx < 0 or round_idx >= FREE_ROUNDS:
            raise ValueError(f"round_idx must be between 0 and {FREE_ROUNDS - 1}, got {round_idx}")

        case_id = "unknown"
        if hasattr(self, "case_path") and self.case_path:
            case_id = self.case_path.parent.name
        elif isinstance(self.case_data, dict):
            case_id = str(self.case_data.get("_case_id", self.case_data.get("id", "unknown")))

        segment_dir = Path("segment")
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_file_path = segment_dir / f"{case_id}_segment.json"

        if not hasattr(self, "_cached_segmented_indices"):
            self._cached_segmented_indices = None

        if self._cached_segmented_indices is None:
            if segment_file_path.exists():
                try:
                    with open(segment_file_path, "r", encoding="utf-8") as f:
                        saved_data = json.load(f)
                        self._cached_segmented_indices = saved_data.get("boundaries", [])
                    print(f"✨ 已成功读取案例 {case_id} 的动态分割文件: {segment_file_path}")
                except Exception as e:
                    print(f"⚠️ 读取动态分割文件失败 ({e})，将重新调用模型生成...")

            if self._cached_segmented_indices is None:
                statements_summary = []
                for idx, ev in enumerate(evidences):
                    raw_identity = ev.get("identity", "未知")
                    content_dict = ev.get("content", {}) or ev
                    raw_txt = content_dict.get("text", "").strip() if content_dict.get("text") else "无文字描述"

                    if raw_identity not in ["买家", "卖家"]:
                        continue  # 跳过小二/平台等非买卖双方的证据，不参与时间切分

                    statements_summary.append({
                        "evidence_index": idx,
                        "provider": raw_identity,
                        "statement": raw_txt[:200]
                    })

                first_agent = next(iter(self.agents.values()))

                dynamic_boundaries = first_agent.segment_evidences(
                    case_id=case_id,
                    statements_summary=statements_summary,
                    num_stages=FREE_ROUNDS,
                    max_retries=5
                )

                self._cached_segmented_indices = dynamic_boundaries

                try:
                    save_payload = {
                        "case_id": case_id,
                        "evidence_count": evidence_count,
                        "num_stages": FREE_ROUNDS,
                        "boundaries": self._cached_segmented_indices
                    }
                    with open(segment_file_path, "w", encoding="utf-8") as f:
                        json.dump(save_payload, f, ensure_ascii=False, indent=2)
                    print(f"💾 [SWER-SGES] 案例 {case_id} 动态分割结果已保存至: {segment_file_path}")
                except Exception as e:
                    print(f"⚠️ 保存动态分割文件到 {segment_file_path} 失败: {e}")

        if 0 <= round_idx < len(self._cached_segmented_indices):
            selected_indices = self._cached_segmented_indices[round_idx]
            print(f"[evidence_split] round={round_idx + 1}/{FREE_ROUNDS}, selected_indices={selected_indices}")
            return selected_indices

        return []

    def build_scoped_case_data(self, round_idx: int) -> Dict:
        """给当前轮次构造只暴露其对应时间段证据的 case_data。"""

        scoped_case_data = copy.deepcopy(self.case_data)
        scoped_case_data["_evidence_indices"] = self.get_round_evidence_indices(round_idx)
        scoped_case_data["_round_idx"] = round_idx
        return scoped_case_data

    def _build_round_memory(self, round_idx: int, agent_name: str, *, include_neighbors: bool = True) -> List[Dict]:
        """Keep prior self verdicts and the raw messages delivered along outgoing follows."""
        memory = []
        for past_round in self.round_history:
            past_idx = past_round["round_idx"]
            if past_idx >= round_idx:
                continue
            own_reply = next(
                (entry["reply"] for entry in past_round["entries"] if entry["agent"] == agent_name),
                "",
            )
            if own_reply:
                memory.append({"role": "assistant", "content": f"第{past_idx + 1}轮自己的裁决：\n{own_reply}"})
            if include_neighbors:
                for delivery in self.interaction_history:
                    if delivery["round_idx"] != past_idx:
                        continue
                    for item in delivery["interactions"]:
                        if item["agent"] == agent_name and item["reply"]:
                            memory.append({
                                "role": "user",
                                "content": f"第{past_idx + 1}轮关注邻居的原始发言：\n{item['reply']}",
                            })
        return memory

    def _run_debate_phases(self, case_data: str):
        """执行N轮讨论"""

        self.round_history = []
        self.interaction_history = []
        self.transcript = []

        agent_names = list(self.agents.keys())
        for round_idx in range(FREE_ROUNDS):
            round_case_data = self.build_scoped_case_data(round_idx)
            round_entries = []
            print(f"\n📢 开始第 {round_idx + 1}/{FREE_ROUNDS} 轮讨论")

            for turn, sp in enumerate(agent_names, 1):
                side = sp.split('_')[0]
                agent = self.agents[sp]
                agent_round_memory = self._build_round_memory(round_idx, sp)
                with trace_scope(stage=round_idx + 1, agent=sp):
                    record_event("agent_context", memory=agent_round_memory,
                                 evidence_indices=round_case_data.get("_evidence_indices", []))
                    evidence_content = agent._cot_active_exploration_loop(
                        phase=f"Round_{round_idx + 1}",
                        side=side,
                        case_data=round_case_data,
                        retrieval_memory=agent_round_memory,
                        perception_memory=agent_round_memory
                    )
                    prompt = self._build_round_prompt(sp, round_idx, evidence_content)
                    with trace_scope(operation="verdict"):
                        reply = self._ask(sp, prompt, evidence_content, shared_memory=agent_round_memory)
                    record_event("verdict", reply=reply, evidence_content=evidence_content)
                round_entries.append({
                    "agent": sp,
                    "side": side,
                    "turn": turn,
                    "evidence_indices": round_case_data.get("_evidence_indices", []),
                    "prompt": prompt,
                    "reply": reply,
                    "evidence_content": evidence_content,
                })

            self.round_history.append({
                "round_idx": round_idx,
                "evidence_indices": round_case_data.get("_evidence_indices", []),
                "entries": round_entries,
            })
            self._record_follow_messages(round_idx)

    def _build_round_prompt(self, speaker: str, round_idx: int, evidence_content: Dict = None) -> str:
        """构建三轮讨论提示词，直接复用 prompts.py 中的模板。"""
        round_name = f"第{round_idx + 1}轮"
        stage_instruction = STAGE_PROMPTS.get(round_idx, STAGE_PROMPTS[0])
        evidence_text = ""
        if evidence_content:
            if evidence_content.get("text"):
                evidence_text += f"\n【本轮证据文本】\n{evidence_content['text']}"
            if evidence_content.get("image_descriptions"):
                evidence_text += "\n【本轮提取线索】\n" + "\n".join(evidence_content["image_descriptions"])
        base_prompt = ROUND_DISCUSSION_TEMPLATE.format(round_name=round_name, stage_instruction=stage_instruction)
        full_prompt = base_prompt + evidence_text
        return full_prompt

    def _load_following_network(self):
        """Load directed follows: node i receives the opinions of nodes listed at network[i]."""
        case_id = self.case_path.parent.name if getattr(self, "case_path", None) else str(
            self.case_data.get("_case_id", self.case_data.get("id", "unknown"))
        )
        net_dir = Path(FOLLOWING_NETWORKS_DIR)
        net_dir.mkdir(parents=True, exist_ok=True)
        net_file_path = net_dir / f"{case_id}_following_networks.json"
        agent_count = len(self.agents)
        if net_file_path.exists():
            network = json.loads(net_file_path.read_text(encoding="utf-8"))
            network = {str(key): [int(value) for value in values] for key, values in network.items()}
        else:
            network = {}
            for agent_id in range(agent_count):
                others = [idx for idx in range(agent_count) if idx != agent_id]
                network[str(agent_id)] = random.sample(others, min(FOLLOWING_NETWORK_OUT_DEGREE, len(others)))
            net_file_path.write_text(json.dumps(network, ensure_ascii=False, indent=2), encoding="utf-8")

        if set(network) != {str(index) for index in range(agent_count)}:
            raise ValueError("Following network must contain exactly one entry per agent")
        for index, neighbors in network.items():
            if len(neighbors) != len(set(neighbors)) or any(
                neighbor == int(index) or not 0 <= neighbor < agent_count for neighbor in neighbors
            ):
                raise ValueError(f"Invalid directed neighbors for agent {index}: {neighbors}")
        self.following_network = network
        return network

    def _record_follow_messages(self, round_idx: int):
        """Deliver completed-round neighbor replies verbatim; no LLM call or summary."""
        if not self.following_network:
            self._load_following_network()
        current_round = next(item for item in self.round_history if item["round_idx"] == round_idx)
        replies = {entry["agent"]: entry["reply"] for entry in current_round["entries"]}
        interactions = []
        for agent_name in self.agents:
            agent_index = DEBATE_AGENTS.index(agent_name)
            neighbors = list(self.following_network[str(agent_index)])
            received = [
                f"{DEBATE_AGENTS[index]}: {replies[DEBATE_AGENTS[index]]}"
                for index in neighbors if replies.get(DEBATE_AGENTS[index])
            ]
            if received:
                interactions.append({
                    "agent": agent_name,
                    "round_idx": round_idx,
                    "neighbors": neighbors,
                    "reply": "\n\n".join(received),
                })
        self.interaction_history.append({"round_idx": round_idx, "interactions": interactions})
        record_event("neighbor_delivery", stage=round_idx + 1, interactions=interactions)

    def _build_node_texts_and_edges(self):
        """Return final raw verdicts in the fixed node order used by the classifier."""
        final_entries = self.round_history[-1]["entries"] if self.round_history else []
        replies = {entry["agent"]: entry["reply"] for entry in final_entries}
        return [replies.get(agent, "") for agent in DEBATE_AGENTS]

    def _save_for_gnn(self, case_data: dict, node_texts: list, edges: list = None, *args, **kwargs):
        """保存用于 GNN 的极简纯文本节点以及案例数据和辩论记录（已物理删除特征编码与边关系）"""
        import json
        from pathlib import Path

        save_dir = getattr(self, "output_dir", None) or config.SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")

        case_id = "unknown_case"
        if hasattr(self, 'case_path') and self.case_path:
            case_id = self.case_path.parent.name
            case_stem = self.case_path.stem
            filename_base = f"{case_id}_{case_stem}"
        else:
            file_path = case_data.get("_file_path", "") if isinstance(case_data, dict) else ""
            if file_path:
                path_obj = Path(file_path)
                case_id = path_obj.parent.name
                filename_base = f"{case_id}_{path_obj.stem}"
            else:
                filename_base = "unknown_case"

        out = Path(save_dir) / f"{filename_base}_{timestamp}_for_gnn.json"

        round_results = []
        for round_item in self.round_history:
            round_results.append({
                "round_idx": round_item.get("round_idx"),
                "agent_conclusions": [
                    {
                        "agent": entry.get("agent"),
                        "reply": entry.get("reply", "")
                    }
                    for entry in round_item.get("entries", [])
                ],
                "interactions": [
                    {
                        "agent": item.get("agent"),
                        "neighbors": item.get("neighbors", []),
                        "reply": item.get("reply", "")
                    }
                    for item in next((x.get("interactions", []) for x in self.interaction_history if x.get("round_idx") == round_item.get("round_idx")), [])
                ]
            })

        save_data = {
            "case_data": case_data,
            "node_texts": node_texts,
            "round_results": round_results,
        }

        temporary = Path(str(out) + '.tmp')
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        temporary.replace(out)

        print(f"💾 已保存三轮讨论结果到 GNN 格式文件: {out}")
        return str(out)

