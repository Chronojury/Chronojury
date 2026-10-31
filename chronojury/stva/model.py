"""
Shared Model Manager for ChronoJury's Spatio-Temporal Verdict Aggregation (STVA) Module.
Singleton pattern implementation to manage shared graph neural network components across agent deliberation instances.
"""
import torch
import torch.nn as nn
from copy import deepcopy
from .graph import DebateGraph
from . import config
from .config import GNN_MODEL_CONFIG, PATH_CONFIG


class TemporalSelfAttention(nn.Module):
    """
    Cross-Stage Opinion Evolution (CSOE) Sub-module.
    Applies causal multi-head self-attention along the stage dimension (T_rounds)
    to model the dynamic opinion trajectories of jurors over time (Section 3.2).
    """
    def __init__(self, input_dim: int, num_heads: int = 2, num_layers: int = 1, dropout: float = 0.1):
        super(TemporalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        
        # Learnable stage positional encodings P in R^{T x d_out} to distinguish temporal logic
        self.position_embeddings = nn.Parameter(torch.zeros(1, 3, input_dim))
        
        # Causal Temporal Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Causal mask M_causal blocking future stages to prevent future information leakage
        # Triu mask shape: [3, 3] -> [[0, -inf, -inf], [0, 0, -inf], [0, 0, 0]]
        mask = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)
        
        # Initialize positional embeddings
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Cross-Stage Opinion Evolution.
        :param x: Node sequence tensor across stages [N_nodes, T_rounds, Hidden_dim]
        :return: Spatio-temporal representations J^{ST} [N_nodes, T_rounds, Hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Inject stage positional encoding: J^P = J^S + P
        x = x + self.position_embeddings[:, :seq_len, :]
        
        # 2. Compute causal multi-head self-attention along stage dimension with M_causal
        out = self.transformer(x, mask=self.causal_mask[:seq_len, :seq_len])
        
        return out

class SharedModelManager:
    """
    Singleton Manager for STVA neural network modules (GAT for ISJI and Transformer for CSOE).
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SharedModelManager._initialized:
            return
        
        self.text_encoder = None
        self.tokenizer = None
        self.stva_structural = None  
        self.stva_temporal = None    
        self.device = None

        self.stva_projection = None
        self.gat_input_projection = None
        self.ablation_mode = None
        self.stva_config = None
        
        SharedModelManager._initialized = True
    
    def initialize(self, device=PATH_CONFIG['device'], force_reload=False):
        if (not force_reload and self.stva_structural is not None
                and self.ablation_mode == config.ABLATION_MODE
                and self.stva_config == GNN_MODEL_CONFIG['stva']
                and self.num_agents == GNN_MODEL_CONFIG['num_agents']
                and torch.device(self.device) == torch.device(device)):
            print("✅ Using existing SharedModelManager instance.")
            return
        
        print("🔧 Initializing SharedModelManager (STVA Engine)...")
        self.device = device
        self.ablation_mode = config.ABLATION_MODE
        self.stva_config = deepcopy(GNN_MODEL_CONFIG['stva'])
        self.input_dim = self.stva_config['structural']['input_dim']
        self.num_agents = GNN_MODEL_CONFIG['num_agents']
        self.num_nodes = self.num_agents + 1

        if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        self.stva_projection = None

        # 1. Instantiate Spatial Interaction Module: Intra-Stage Inter-Juror Interaction (ISJI) via GAT
        print("🕸️ Initializing Spatial GAT Layer (ISJI)...")
        structural_config = dict(self.stva_config['structural'])
        projection_config = self.stva_config.get('gat_input_projection', {})
        self.gat_input_projection = None
        if self.ablation_mode in ('complete', 'gat_concat') and projection_config.get('enabled', False):
            projected_dim = int(projection_config['output_dim'])
            if projected_dim <= 0:
                raise ValueError('GAT input projection output_dim must be positive')
            self.gat_input_projection = nn.Sequential(
                nn.Linear(structural_config['input_dim'], projected_dim),
                nn.LayerNorm(projected_dim),
                nn.GELU(),
            ).to(self.device)
            structural_config['input_dim'] = projected_dim
            print(f"🔽 GAT input projection initialized: {GNN_MODEL_CONFIG['stva']['structural']['input_dim']} -> {projected_dim}")
        self.stva_structural = DebateGraph(**structural_config).to(self.device)
        print(f"✅ Structural GAT initialized. Parameters: {sum(p.numel() for p in self.stva_structural.parameters()):,}")
        
        # 2. Instantiate Temporal Evolution Module: Cross-Stage Opinion Evolution (CSOE) via Causal Self-Attention
        print("🔗 Initializing Causal Temporal Attention Layer (CSOE)...")
        self.stva_temporal = TemporalSelfAttention(**self.stva_config['temporal']).to(self.device)
        print(f"✅ Temporal Attention initialized. Parameters: {sum(p.numel() for p in self.stva_temporal.parameters()):,}")
        


        # initialize() 中，在创建 stva_temporal 后
        if self.ablation_mode == 'temporal_only':
            input_dim = self.input_dim
            output_dim = self.stva_config['temporal']['input_dim']
            self.stva_projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            ).to(self.device)

        total_params = sum(
            p.numel()
            for module in (
                self.stva_structural,
                self.gat_input_projection,
                self.stva_temporal,
                self.stva_projection,
            )
            if module is not None
            for p in module.parameters()
        )
        print(f"🎯 Total STVA Parameters: {total_params:,}")
        print(f"🎯 Compute Device: {self.device}")
        print("✅ STVA SharedModelManager successfully initialized.\n")
        

    def load_checkpoint_weights(self, checkpoint):
        """Restore all STVA weights; missing learned projections are an error."""
        if checkpoint.get('ablation_mode', self.ablation_mode) != self.ablation_mode:
            raise ValueError(f"Checkpoint mode {checkpoint['ablation_mode']} does not match {self.ablation_mode}")
        if checkpoint.get('input_dim', self.input_dim) != self.input_dim:
            raise ValueError(f"Checkpoint input width {checkpoint['input_dim']} does not match {self.input_dim}")
        if checkpoint.get('num_agents', 6) != self.num_agents:
            raise ValueError('Checkpoint juror count does not match --num-agents')
        if checkpoint.get('temporal_layers', 2) != self.stva_config['temporal']['num_layers']:
            raise ValueError('Checkpoint temporal layer count does not match --temporal-layers')
        self.stva_structural.load_state_dict(checkpoint['gat_state_dict'])
        temporal = checkpoint.get('stva_temporal_state_dict', checkpoint.get('attention_state_dict'))
        if temporal is None:
            raise ValueError("Checkpoint is missing temporal attention weights")
        self.stva_temporal.load_state_dict(temporal)
        for name in ('gat_input_projection', 'stva_projection'):
            module = getattr(self, name)
            weights = checkpoint.get(f'{name}_state_dict')
            if module is not None:
                if weights is None:
                    raise ValueError(f"Checkpoint is missing {name} weights")
                module.load_state_dict(weights)
            elif weights is not None:
                raise ValueError(f"Checkpoint has {name} weights but this configuration disables it")

    def get_all_trainable_parameters(self):
        """Retrieves all trainable parameters across STVA modules."""
        params = []
        if self.stva_structural is not None:
            params.extend(self.stva_structural.parameters())
        if self.gat_input_projection is not None:
            params.extend(self.gat_input_projection.parameters())
        if self.stva_temporal is not None:
            params.extend(self.stva_temporal.parameters())
        if self.stva_projection is not None:
            params.extend(self.stva_projection.parameters())
        return params
    
    def train_mode(self):
        """Sets STVA neural network modules to training mode."""
        if self.stva_structural is not None:
            self.stva_structural.train()
        if self.gat_input_projection is not None:
            self.gat_input_projection.train()
        if self.stva_temporal is not None:
            self.stva_temporal.train()
        if self.stva_projection is not None:
            self.stva_projection.train()
    
    def eval_mode(self):
        """Sets STVA neural network modules to evaluation mode."""
        if self.stva_structural is not None:
            self.stva_structural.eval()
        if self.gat_input_projection is not None:
            self.gat_input_projection.eval()
        if self.stva_temporal is not None:
            self.stva_temporal.eval()
        if self.stva_projection is not None:
            self.stva_projection.eval()
    
    def reset(self):
        self.text_encoder = None
        self.tokenizer = None
        self.stva_structural = None
        self.gat_input_projection = None
        self.stva_temporal = None
        self.stva_projection = None
        self.ablation_mode = None
        self.stva_config = None
        self.device = None
        SharedModelManager._initialized = False
        print("🔄 SharedModelManager reset.")

    def forward_batch(self, text_features, edge_index):
        """Process [B, T, N, D] features with disjoint case graphs.

        edge_index already offsets each case by N nodes. Each stage receives
        its own copy of those edges, and temporal attention treats every
        (case, node) pair as an independent sequence. Parameter names and
        checkpoint layout are shared with the single-case path below.
        """
        if text_features.dim() != 4 or text_features.shape[-1] != self.input_dim:
            raise ValueError(f'Expected [batch, stages, nodes, {self.input_dim}] features')
        batch, stages, nodes, _ = text_features.shape
        if nodes != self.num_nodes:
            raise ValueError(f'Expected {self.num_nodes} nodes ({self.num_agents} jurors and one case node)')
        if self.ablation_mode == 'raw_pool':
            return text_features.mean(dim=2).mean(dim=1)

        if self.ablation_mode == 'temporal_only':
            spatial = self.stva_projection(text_features)
        else:
            projected = (self.gat_input_projection(text_features)
                         if self.gat_input_projection is not None else text_features)
            # Flatten in stage/case/node order; no edge crosses cases or stages.
            stage_edges = torch.cat([edge_index + t * batch * nodes for t in range(stages)], dim=1)
            flat = projected.permute(1, 0, 2, 3).reshape(stages * batch * nodes, -1)
            spatial = self.stva_structural(flat, stage_edges)
            spatial = spatial.reshape(stages, batch, nodes, -1).permute(1, 0, 2, 3)
            if self.ablation_mode == 'gat_concat':
                return spatial.mean(dim=1).reshape(batch, -1)

        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, stages, -1)
        temporal = self.stva_temporal(sequences)
        return temporal[:, -1, :].reshape(batch, nodes, -1).mean(dim=1)

    def forward_train(self, text_features_3d, edge_index):
            """Two-Stage Spatio-Temporal Forward Pass (STVA Module Pipeline) 支持消融实验模式

            :param text_features_3d: Stage-wise node feature matrix [T_rounds,
            N_nodes, Feature_dim] :param edge_index: Heterogeneous spatio-temporal
            graph topology [2, Num_edges] :return: Aggregated verdict representation
            h_final [1, Hidden_dim] & spatial attention weights
            """
            self.last_attention_edge_index = None
            self.stva_structural.to(self.device)
            self.stva_temporal.to(self.device)
            text_features_3d = text_features_3d.to(self.device)
            edge_index = edge_index.to(self.device)

            # 🛡️ 形状对齐防御（针对部分样本多包一层维度的兼容）
            if text_features_3d.dim() == 4 and text_features_3d.shape[0] == 1:
                text_features_3d = text_features_3d.squeeze(0)

            if text_features_3d.dim() != 3 or text_features_3d.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected node features [stages, nodes, {self.input_dim}], "
                    f"got {tuple(text_features_3d.shape)}. Match --input-dim to the saved encoder features."
                )

            model_device = next(self.stva_structural.parameters()).device
            if text_features_3d.shape[1] != self.num_nodes:
                raise ValueError(f'Expected {self.num_nodes} nodes for this juror count')

            # =========================================================================
            # 🔬 消融模式 1: raw_pool (移除空间 GAT 与时间自注意力，仅做均值池化与时序拼接)
            # =========================================================================
            if self.ablation_mode == 'raw_pool':
                # 1. 在节点轴(dim=1)做均值池化 -> 形状变为 [num_rounds, 2560]
                graph_pooled = text_features_3d.mean(dim=1)
                
                # 2. 在轮次轴(dim=0)上做均值池化，融合所有轮次 -> 形状变为 [2560]
                temporal_pooled = graph_pooled.mean(dim=0)
                
                # 3. 增加 batch 维度 -> 形状严格为 [1, 2560]
                h_final = temporal_pooled.unsqueeze(0)
                return h_final, None
            
            gat_features = (
                self.gat_input_projection(text_features_3d)
                if self.gat_input_projection is not None else text_features_3d
            )

            # =========================================================================
            # 🔬 消融模式 2: gat_concat (保留空间 GAT，移除时间自注意力，跨阶段均值后拼接节点)
            # =========================================================================
            if self.ablation_mode == 'gat_concat':
                round_node_reps = []
                last_edge_attn = None
                for t in range(text_features_3d.shape[0]):
                    x_t = gat_features[t, :, :].to(model_device)
                    # 统一修正为 self.stva_structural
                    updated_nodes, attention_edges, edge_attn = self.stva_structural(
                        x_t, edge_index, return_edge_weights=True
                    )
                    self.last_attention_edge_index = attention_edges
                    round_node_reps.append(updated_nodes)  # 每个是 [7, GAT output_dim]
                    last_edge_attn = edge_attn

                # 堆叠并进行时间轴上的均值池化 -> [7, GAT output_dim]
                gat_stacked = torch.stack(round_node_reps, dim=0).mean(dim=0)
                # 横向拼接 7 个节点特征 -> [1, 7 * GAT output_dim]
                h_final = gat_stacked.view(1, -1)
                return h_final, last_edge_attn

            # =========================================================================
            # 🔬 消融模式 3: temporal_only (移除空间 GAT，仅通过线性投影对齐后输入时间自注意力)
            # =========================================================================
            if self.ablation_mode == 'temporal_only':
                # 如果直接输入 1024 维到预期 256 维的 stva_temporal 会报错，故需临时降维或投影
                # 检查是否有投影层，无则动态通过切片/均值或复用线性层投影到 256 维
                feats_aligned = self.stva_projection(text_features_3d)  # [3, 7, 128]

                # 调整维度对齐时序输入格式: [3, 7, 256] -> [7, 3, 256]
                nodes_time_seq = feats_aligned.permute(1, 0, 2)
                # 统一修正为 self.stva_temporal
                temporal_node_feats = self.stva_temporal(nodes_time_seq)
                # 切片提取最后一轮终审观点 -> [7, 256]
                final_node_feats = temporal_node_feats[:, -1, :]
                # 全局图池化聚合为一维案件向量 -> [1, 256]
                h_final = final_node_feats.mean(dim=0, keepdim=True)
                return h_final, None

            # =========================================================================
            # 🚀 完整模式 (Complete Mode): 默认 STVA 空间 GAT + 时间因果自注意力
            # =========================================================================
            # ---------------- STAGE 1: Spatial Interaction (ISJI) ----------------
            round_node_reps = []
            for t in range(text_features_3d.shape[0]):
                x_t = gat_features[t, :, :].to(model_device)
                updated_nodes, attention_edges, edge_attn = self.stva_structural(
                    x_t, edge_index, return_edge_weights=True
                )
                self.last_attention_edge_index = attention_edges
                round_node_reps.append(updated_nodes)

            # ---------------- STAGE 2: Temporal Evolution (CSOE) ----------------
            rounds_stacked = torch.stack(round_node_reps, dim=0)
            nodes_time_seq = rounds_stacked.permute(1, 0, 2)
            temporal_node_feats = self.stva_temporal(nodes_time_seq)
            final_node_feats = temporal_node_feats[:, -1, :]
            h_final = final_node_feats.mean(dim=0, keepdim=True)

            return h_final, edge_attn

# Global Singleton Instance
shared_model_manager = SharedModelManager()
