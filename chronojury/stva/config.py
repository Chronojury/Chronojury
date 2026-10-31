import os
import torch

NUM_AGENTS = int(os.environ.get('CHRONOJURY_NUM_AGENTS', '6'))
if NUM_AGENTS < 2:
    raise ValueError('CHRONOJURY_NUM_AGENTS must be at least 2')

# ==================== 1. Path & Runtime Environment Configurations ====================
PATH_CONFIG = {
    'train_dir': 'features/qwen/train',
    'val_dir': 'features/qwen/val',
    'test_dir': 'features/qwen/test',
    'test_val_dir': 'features/qwen/test',
    'model_save_dir': 'checkpoints/qwen/complete',
    'tensorboard_log_dir': 'checkpoints/logs', 
    'following_network_dir': 'following_networks' if NUM_AGENTS == 6 else f'following_networks_{NUM_AGENTS}',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

ABLATION_MODE = 'complete'

# ==================== 2. Spatio-Temporal Verdict Graph Architecture ====================
GNN_MODEL_CONFIG = {
    'num_agents': NUM_AGENTS,
    # The last node contains case information and sends it to every juror.
    # Serves as the case context guidance for intra-stage opinion aggregation 
    'graph_topology': [[NUM_AGENTS, index] for index in range(NUM_AGENTS)],
    # Spatio-Temporal Verdict Aggregation (STVA) Module Hyperparameters
    'stva': {
        # Spatial Interaction Stage: Intra-Stage Inter-Juror Interaction (ISJI) via GAT
        'structural': {
            'input_dim': 1024,   # Dimension of initial node semantic embeddings
            'hidden_dim': 256,   # Intermediate hidden representation dimension
            'output_dim': 256,   # Output dimension of spatial interaction representations
            'num_layers': 1,     # Single-layer Graph Attention Network (GAT)
            'num_heads': 1,      # Number of spatial attention heads
            'dropout': 0.1
        },
        # Used by complete and gat_concat. Disable to feed the original 4096-dimensional
        # features directly into the GAT.
        'gat_input_projection': {
            'enabled': False,
            'output_dim': 1024,
        },
        # Temporal Evolution Stage: Cross-Stage Opinion Evolution (CSOE) via Causal Self-Attention
        'temporal': {
            'input_dim': 256,    # Input dimension aligned with ISJI output
            'num_heads': 2,      # Number of temporal attention heads
            'num_layers': 2,     # Number of Transformer Causal Self-Attention layers
            'dropout': 0.2
        }
    },
    
    'classifier': {
        'type': 'mlp',           # linear/mlp
        'input_dim': 256,        # Spatio-temporal verdict feature vector dimension
        'hidden_dim': 128,       # Multilayer Perceptrons (MLP) hidden dimension
        'num_classes': 2,
        'dropout_rate': 0.2    
    }
}

if ABLATION_MODE == 'gat_concat':
    GNN_MODEL_CONFIG['classifier']['input_dim'] = (NUM_AGENTS + 1) * GNN_MODEL_CONFIG['stva']['structural']['output_dim']
projection_cfg = GNN_MODEL_CONFIG['stva']['gat_input_projection']
projection_tag = (
    f"linear_gelu{projection_cfg['output_dim']}_projlr2e-6" if projection_cfg['enabled']
    else 'direct4096'
)



# ==================== 3. Optimization & Training Hyperparameters ====================
HYPER_PARAMS = {
    'epochs': 200,
    'batch_size': 32,
    'weight_decay': 3e-3,
    'label_smoothing': 0.0,
    'class_weights': [1.0, 1.0], 
    'validate_every': 1,
    # Multitrack Learning Rate Scheduler Strategy
    'multitrack_lr_config': {
        # Track 1: Structural Attention (ISJI / GAT) Learning Rate Track
        'stva_structural': {
            'init_lr': 1e-5,
            'type': 'exponential',          
            'gamma': 0.97,
            'step': 5,
            'min_lr': 1e-7
        },
        # Keep the extra input projection slower than the GAT.
        'gat_input_projection': {
            'init_lr': 2e-5,
            'type': 'exponential',
            'gamma': 0.97,
            'min_lr': 1e-7,
        },
        # Track 2: Temporal Causal Self-Attention (CSOE) Learning Rate Track
        'stva_temporal': {
            'init_lr': 2e-5,          
            'type': 'exponential',     
            'gamma': 0.97,
            'step': 5,
            'min_lr': 1e-7
        },
        'classifier': {
            'init_lr': 1e-4,           
            'type': 'exponential',
            'step': 5,
            'gamma': 0.97,
            'min_lr': 1e-7                              
        } # step(step_size,gamma),exponential(gamma),cosine(T_max,eta_min),
    }
}

DATA_LOADER_CONFIG = {
    'num_workers': 4,
    'prefetch_factor': 2,
    'pin_memory': True,
}

EXP_TAG = f"9B_qwen-complete-{projection_tag}-1GAT_128_1h_1e-5_0.97_0.1-2TA_2h_2e-5_0.97_0.2-CLS_64_1e-5_0.97_0.2"  # name of tensorboard log folder
if ABLATION_MODE == 'gat_concat':
    EXP_TAG = f"9B_qwen-gat_concat-{projection_tag}-1GAT_128_0.1-CLS_64_0.2"
# ==================== 4. Pipeline Execution Protocols ====================
TRAINING_CONFIG = {
    'mode': 'train',
    'graph_execution': 'batch',  # 'serial' retains the original per-case execution order
    'train_dir': PATH_CONFIG['train_dir'],
    'val_dir': PATH_CONFIG['val_dir'],
    'data_dir': './dataset',
    'model_path': None,
    'epochs': HYPER_PARAMS['epochs'],
    'lr': HYPER_PARAMS['multitrack_lr_config']['stva_structural']['init_lr'], 
    'device': PATH_CONFIG['device'],
    'case_file': None,
    'validate_every': HYPER_PARAMS['validate_every'],
    'resume_from': None, # None or checkpoint path
    'seed': 42,
    'ablation_mode': ABLATION_MODE,
    'exp_tag': EXP_TAG
}

TESTING_CONFIG = {
    'mode': 'test',
    'data_dir': './dataset',
    'train_dir': PATH_CONFIG['test_dir'],
    'val_dir': PATH_CONFIG['test_val_dir'],
    'model_path': '', # Path to the trained model checkpoint for testing
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'ablation_mode': ABLATION_MODE
}

#Config selector
CURRENT_CONFIG = TRAINING_CONFIG
# CURRENT_CONFIG = TESTING_CONFIG

