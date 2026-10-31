"""
Main Training & Evaluation Script for ChronoJury
Handles full pipeline execution including data loading, model training, and test evaluation.
"""
import sys
import os
from pathlib import Path

import argparse
from . import config
from .config import PATH_CONFIG, HYPER_PARAMS, TRAINING_CONFIG, TESTING_CONFIG, CURRENT_CONFIG

import time
import json
import torch
import numpy as np
import random
import logging
from datetime import datetime
from .trainer import TrainingManager
from .classifier import DebateClassificationManager
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)



def set_seed(seed=2026):
    """
    Fixes global random seeds across Python, NumPy, PyTorch, and CuDNN
    to guarantee exact reproducibility of graph neural network training.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

        

def setup_logger(save_dir):
    log_dir = Path('.') / "logs"  
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    experiment_name = os.environ.get("DEBATE_EXPERIMENT_NAME", "")
    log_file = log_dir / f"training_{timestamp}_{experiment_name}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    class PrintToLogger:
        def __init__(self, logger_level, terminal):
            self.logger_level = logger_level
        def write(self, message):
            if message.strip():
                logger.log(self.logger_level, message.strip())
        def flush(self):
            pass
        def isatty(self):
            return False

    self_stdout = sys.stdout
    self_stderr = sys.stderr
    sys.stdout = PrintToLogger(logging.INFO, self_stdout)
    sys.stderr = PrintToLogger(logging.ERROR, self_stderr)

    sys.__stdout__.write(f"📝 Logging system initialized at: {log_file}\n")
    return logger

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train or evaluate ChronoJury STVA on saved graph features")
    parser.add_argument('--mode', choices=('train', 'test'), default=CURRENT_CONFIG['mode'])
    parser.add_argument('--ablation', choices=('complete', 'gat_concat', 'raw_pool', 'temporal_only'))
    parser.add_argument('--input-dim', type=int, help='Saved feature width: 1024 for RoBERTa, 4096 for the original Qwen features')
    parser.add_argument('--gat-input-projection', action=argparse.BooleanOptionalAction,
                        help='Enable or disable the learned projection before GAT')
    parser.add_argument('--experiment-name', help='Label used for this TensorBoard run')
    parser.add_argument('--num-agents', type=int, default=config.GNN_MODEL_CONFIG['num_agents'],
                        help='Number of jurors; one additional node contains case information')
    parser.add_argument('--temporal-layers', type=int,
                        help='Number of causal Transformer layers; defaults to the model config')
    parser.add_argument('--train-dir')
    parser.add_argument('--val-dir')
    parser.add_argument('--test-dir')
    parser.add_argument('--network-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--tensorboard-dir', help='Directory for TensorBoard event files')
    parser.add_argument('--model-path')
    parser.add_argument('--resume-from')
    parser.add_argument('--optimizer-config', type=Path,
                        help='JSON overrides for per-module learning rates and optional warmup')
    parser.add_argument('--reset-lr-schedule', action='store_true',
                        help='Restart the LR schedule on resume while retaining Adam moments')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--graph-execution', choices=('batch', 'serial'),
                        default=config.TRAINING_CONFIG['graph_execution'])
    parser.add_argument('--loader-workers', type=int, default=config.DATA_LOADER_CONFIG['num_workers'])
    parser.add_argument('--prefetch-batches', type=int, default=config.DATA_LOADER_CONFIG['prefetch_factor'])
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int)
    overrides = parser.parse_args(argv)
    if overrides.reset_lr_schedule and not overrides.resume_from:
        parser.error('--reset-lr-schedule requires --resume-from')
    if overrides.optimizer_config and overrides.resume_from and not overrides.reset_lr_schedule:
        parser.error('Use --reset-lr-schedule to apply a new optimizer config when resuming')
    if overrides.optimizer_config:
        optimizer_config = json.loads(overrides.optimizer_config.read_text())
        for key in ('warmup_epochs', 'warmup_start_lr'):
            if key in optimizer_config:
                HYPER_PARAMS[key] = optimizer_config[key]
        for name, values in optimizer_config.get('multitrack_lr_config', {}).items():
            if name not in HYPER_PARAMS['multitrack_lr_config']:
                parser.error(f'Unknown learning-rate track: {name}')
            HYPER_PARAMS['multitrack_lr_config'][name].update(values)
        config.TRAINING_CONFIG['lr'] = HYPER_PARAMS['multitrack_lr_config']['stva_structural']['init_lr']
        if not isinstance(HYPER_PARAMS.get('warmup_epochs', 0), int) or HYPER_PARAMS.get('warmup_epochs', 0) < 0:
            parser.error('warmup_epochs must be a nonnegative integer')
        for values in HYPER_PARAMS['multitrack_lr_config'].values():
            if not 0 < values['min_lr'] <= values['init_lr']:
                parser.error('Learning rates must satisfy 0 < min_lr <= init_lr')
    config.TRAINING_CONFIG['reset_lr_schedule'] = overrides.reset_lr_schedule
    config.TRAINING_CONFIG['graph_execution'] = overrides.graph_execution
    if overrides.loader_workers < 0 or overrides.prefetch_batches < 1:
        parser.error('Loader workers must be nonnegative and prefetch batches must be positive')
    config.DATA_LOADER_CONFIG.update(num_workers=overrides.loader_workers,
                                    prefetch_factor=overrides.prefetch_batches)
    values = dict(TRAINING_CONFIG if overrides.mode == 'train' else TESTING_CONFIG)
    for key, value in vars(overrides).items():
        if value is not None and key not in ('ablation', 'input_dim', 'batch_size', 'save_dir', 'network_dir', 'test_dir'):
            values[key] = value
    if overrides.mode == 'test' and overrides.test_dir is not None:
        values['val_dir'] = overrides.test_dir
    for argument, path_key in (('train_dir', 'train_dir'), ('val_dir', 'val_dir'),
                               ('test_dir', 'test_dir'), ('save_dir', 'model_save_dir'),
                               ('tensorboard_dir', 'tensorboard_log_dir'),
                               ('network_dir', 'following_network_dir')):
        value = getattr(overrides, argument)
        if value is not None:
            PATH_CONFIG[path_key] = value
    if overrides.batch_size is not None:
        HYPER_PARAMS['batch_size'] = overrides.batch_size
    model = config.GNN_MODEL_CONFIG
    if overrides.num_agents < 2:
        parser.error('--num-agents must be at least 2')
    model['num_agents'] = overrides.num_agents
    model['graph_topology'] = [[overrides.num_agents, index] for index in range(overrides.num_agents)]
    if overrides.network_dir is None and overrides.num_agents != config.NUM_AGENTS:
        PATH_CONFIG['following_network_dir'] = (
            'following_networks' if overrides.num_agents == 6 else f'following_networks_{overrides.num_agents}'
        )
    if overrides.temporal_layers is not None:
        if overrides.temporal_layers < 1:
            parser.error('--temporal-layers must be positive')
        model['stva']['temporal']['num_layers'] = overrides.temporal_layers
    structural = model['stva']['structural']
    if overrides.gat_input_projection is not None:
        model['stva']['gat_input_projection']['enabled'] = overrides.gat_input_projection
    if overrides.experiment_name:
        config.CURRENT_CONFIG['exp_tag'] = overrides.experiment_name
    if overrides.input_dim is not None:
        if overrides.input_dim < 1:
            parser.error('--input-dim must be positive')
        structural['input_dim'] = overrides.input_dim
    if overrides.ablation is not None:
        config.ABLATION_MODE = overrides.ablation
        values['ablation_mode'] = overrides.ablation
    if overrides.ablation is not None or overrides.input_dim is not None or overrides.num_agents is not None:
        mode = config.ABLATION_MODE
        model['classifier']['input_dim'] = (
            (model['num_agents'] + 1) * structural['output_dim'] if mode == 'gat_concat'
            else structural['input_dim'] if mode == 'raw_pool'
            else model['stva']['temporal']['input_dim']
        )
    if values['mode'] == 'test' and not values.get('model_path'):
        parser.error('--model-path is required for testing')
    if overrides.batch_size is not None and overrides.batch_size < 2:
        parser.error('--batch-size must be at least 2 for the BatchNorm classifier')
    if values.get('epochs', 1) < 1:
        parser.error('--epochs must be positive')
    return argparse.Namespace(**values)


def main(argv=None):
    """Train or evaluate without changing the archived model defaults."""
    args = parse_args(argv)

    # Set global seed
    current_seed = getattr(args, 'seed', 42)
    set_seed(current_seed)

    save_dir = getattr(args, 'model_save_dir', PATH_CONFIG.get('model_save_dir', 'trained_models'))
    setup_logger(save_dir)
    
    print("🚀 Launching ChronoJury STVA Classification Engine...")
    print(f"📋 Mode: {args.mode}")
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"🔧 Compute Device: {args.device}")
    print(f"🎯 Epochs: {getattr(args, 'epochs', 'N/A')}")
    print(f"📈 Learning Rate: {getattr(args, 'lr', 'N/A')}")

    if hasattr(args, 'model_path') and args.model_path:
        print(f"📥 Model Path: {args.model_path}")
    if hasattr(args, 'case_file') and args.case_file:
        print(f"📄 Case File: {args.case_file}")
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'test':
        run_testing(args)
    else:
        print(f"❌ ❌ Unknown execution mode {args.mode}")
        print("✅ Supported modes: train, test")


def run_training(args):
    """Load each split once, then train the configured STVA components."""
    device = 'cpu' if args.device == 'cuda' and not torch.cuda.is_available() else args.device
    resume_from = getattr(args, 'resume_from', None)
    if resume_from and not Path(resume_from).is_file():
        raise FileNotFoundError(resume_from)
    use_separate_dirs = bool(getattr(args, 'train_dir', None) and getattr(args, 'val_dir', None))
    data_options = (
        {'train_dir': args.train_dir, 'val_dir': args.val_dir}
        if use_separate_dirs else {'data_dir': args.data_dir}
    )
    results = TrainingManager(None).run_full_training(
        **data_options,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=HYPER_PARAMS['weight_decay'],
        save_dir=PATH_CONFIG['model_save_dir'],
        device=device,
        validate_every=getattr(args, 'validate_every', 1),
        resume_from=resume_from,
    )
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Final training accuracy: {results['train_acc'][-1]:.4f}")


def run_testing(args):
    """Executes offline evaluation pipeline on test dataset split."""
    print("\n🧪 Starting Model Evaluation Pipeline...")
    start_time = time.time()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    training_manager = TrainingManager(None)
    _, test_dataset = training_manager.create_dataset(train_dir=args.val_dir, val_dir=args.val_dir)

    classification_manager = DebateClassificationManager(
        model_path=args.model_path,
        device=device
    )
    from .model import shared_model_manager
    shared_model_manager.initialize(device=device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    shared_model_manager.load_checkpoint_weights(checkpoint)

    shared_model_manager.eval_mode()
    classification_manager.classifier.eval()

    all_preds = []
    all_labels = []
    correct_count = 0

    from .config import GNN_MODEL_CONFIG
    from .trainer import load_case_edges  # 👈 导入刚才封装好的解析函数
    
    hardcoded_edges_raw = GNN_MODEL_CONFIG.get('graph_topology', [])
    base_net_path = Path(PATH_CONFIG.get("following_network_dir", "following_networks"))

    correct_case_ids = []
    incorrect_case_ids = []

    print(f"🚀 Starting inference on {len(test_dataset)} test cases.")

    with torch.no_grad():
        for i, (case_data, true_label) in enumerate(test_dataset):
            case_id = case_data.get('_case_id', f"case_{i}")
            text_feats = torch.tensor(case_data['node_features'], dtype=torch.float32).to(device)

            # interactive_rep, edge_attn = shared_model_manager.forward_train(text_feats, hardcoded_edges_tensor)
            case_edges_raw = load_case_edges(case_id, base_net_path, hardcoded_edges_raw)
            case_edges_tensor = torch.tensor(
                case_edges_raw, dtype=torch.long
            ).t().contiguous().to(device)

            interactive_rep, edge_attn = shared_model_manager.forward_train(
                text_feats, case_edges_tensor
            )
            
            logits = classification_manager.classifier(interactive_rep)
            pred_label = torch.argmax(logits, dim=-1).item()

            all_preds.append(pred_label)
            all_labels.append(true_label)

            if pred_label == true_label:
                correct_count += 1
                correct_case_ids.append(case_id)
            else:
                incorrect_case_ids.append(case_id)

            # Export spatial interaction attention weights for interpretability analysis
            if edge_attn is not None:
                
                attn_dir = Path("./test_attention_matrices")
                attn_dir.mkdir(parents=True, exist_ok=True)
                
                case_dump_data = {
                    "case_id": str(case_id),
                    "edges": shared_model_manager.last_attention_edge_index.t().cpu().tolist(),                               
                    "attention": edge_attn.cpu().numpy().tolist(),    
                    "ground_truth": int(true_label),                  
                    "predicted": int(pred_label)                     
                }
                
                with open(attn_dir / f"attn_{case_id}.json", "w", encoding="utf-8") as f:
                    json.dump(case_dump_data, f, ensure_ascii=False, indent=2)
            
            if (i + 1) % 50 == 0:
                print(f"✅ Processed {i+1}/{len(test_dataset)} | Current Acc: {correct_count/(i+1):.4f}")
    
    elapsed_time = time.time() - start_time
    total_samples = len(all_labels)
    
    if total_samples > 0:
        accuracy = correct_count / total_samples
        macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        print(f"\n{'='*60}")
        print("📊 Test Results:")
        print(f"   Total Test Samples: {total_samples}")
        print(f"   Total Correct Predictions: {correct_count}")
        print(f"   Accuracy: {accuracy:.4f}")
        print("   --- Core Metrics (Macro - Fair to Classes) ---")
        print(f"   Macro-Precision: {macro_p:.4f}")
        print(f"   Macro-Recall:    {macro_r:.4f}")
        print(f"   Macro-F1:        {macro_f1:.4f}")
        print("   --- Other F1  ---")
        print(f"   Micro-F1:        {micro_f1:.4f}")
        print(f"   Weighted-F1:     {weighted_f1:.4f}")
        print("\n🧱 Confusion Matrix (0: Buyer, 1: Seller):")
        print("       Pred 0    Pred 1")
        print(f"True 0:  {cm[0][0]:<8} {cm[0][1]}")
        print(f"True 1:  {cm[1][0]:<8} {cm[1][1]}")
        print("   --- Performance Statistics ---")
        print(f"   Total Time: {elapsed_time:.2f}s | Average per Case: {elapsed_time/total_samples:.4f}s")
        print(f"{'='*60}")

        results_file = f"test_results_{total_samples}samples.json"
        save_data = {
            'summary': {
                'accuracy': accuracy,
                'macro_precision': macro_p,
                'macro_recall': macro_r,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'weighted_f1': weighted_f1,
                'confusion_matrix': cm.tolist()
            },
            'case_details': {
                'correct_case_ids': correct_case_ids,
                'incorrect_case_ids': incorrect_case_ids
            },
            'report': classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Buyer', 'Seller'], output_dict=True, zero_division=0)
        }
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Detailed test metrics saved to: {results_file}")

    else:
        print("❌ No evaluation results generated.")


if __name__ == "__main__":
    main()
