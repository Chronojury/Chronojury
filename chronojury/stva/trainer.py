"""
End-to-End Training Module for ChronoJury
Integrates Spatio-Temporal Verdict Aggregation (STVA) and Verdict Generation Head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .classifier import DebateClassifier
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from .config import HYPER_PARAMS, GNN_MODEL_CONFIG, PATH_CONFIG
import matplotlib.pyplot as plt


class CosineDecayLR(optim.lr_scheduler.CosineAnnealingLR):
    """Keep the minimum learning rate after one cosine decay."""

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.optimizer.param_groups]
        return super().get_lr()


def load_case_edges(case_id: str, base_net_path: Path, hardcoded_edges: list) -> list:
    """
    读取指定案例的 following_networks JSON 邻接表，并与 GNN_MODEL_CONFIG 的 graph_topology 合并
    """
    dynamic_edges = []
    case_net_file = base_net_path / f"{case_id}_following_networks.json"
    
    with open(case_net_file, 'r', encoding='utf-8') as f:
        net_data = json.load(f)
    
    if isinstance(net_data, dict):
        # 遍历 {"0": [1, 2], "1": [0, 2], ...}
        for follower_key, followed_ids in net_data.items():
            if follower_key.isdigit() and isinstance(followed_ids, (list, tuple)):
                follower = int(follower_key)
                for followed in followed_ids:
                    dynamic_edges.append([int(followed), follower])
    elif isinstance(net_data, list):
        # 兼容已经是列表的情况
        for edge in net_data:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                dynamic_edges.append([int(edge[0]), int(edge[1])])

    # 合并训练配置中的案件信息节点引证边 (Node 6 -> Nodes 0-5)
    merged_edges = list(dynamic_edges) + list(hardcoded_edges)
    
    # 集合去重并保证顺序稳定
    seen = set()
    final_edges = []
    for edge in merged_edges:
        edge_tuple = (int(edge[0]), int(edge[1]))
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            final_edges.append([edge_tuple[0], edge_tuple[1]])
            
    return final_edges


def collate_graphs(samples):
    """Stack features and offset directed edges on CPU before GPU transfer."""
    cases, labels = zip(*samples)
    features = torch.stack([case['node_features'] for case in cases])
    nodes = features.shape[2]
    edges = torch.cat([case['_edge_index'] + index * nodes
                       for index, case in enumerate(cases)], dim=1)
    return {'node_features': features, 'edge_index': edges,
            'labels': torch.tensor([label[0] if isinstance(label, (list, tuple)) else label
                                    for label in labels], dtype=torch.long),
            'case_ids': [case['_case_id'] for case in cases]}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader_options():
    from .config import DATA_LOADER_CONFIG
    options = dict(num_workers=DATA_LOADER_CONFIG['num_workers'],
                   pin_memory=DATA_LOADER_CONFIG['pin_memory'])
    if options['num_workers']:
        options.update(prefetch_factor=DATA_LOADER_CONFIG['prefetch_factor'], persistent_workers=True)
    return options



class DebateDataset(Dataset):
    """
    Dataset loader for pre-computed multi-stage node features (_for_gnn.json).
    Loads juror deliberation representations and case information across stages.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', test_ratio: float = 0.2):
        self.data_dir = Path(data_dir)
        self.split = split
        self.test_ratio = test_ratio
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        """Loads pre-processed STVA graph feature files."""
        gnn_files = list(self.data_dir.glob("*_for_gnn.json"))
        print(f"🔍 Found {len(gnn_files)} GNN feature files from {self.data_dir}")
        
        all_samples = []
        
        for file_path in gnn_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            case_data = data.get('case_data', {})
            label = case_data.get('whowins')
            
            if label is None:
                continue

            # Node features across T deliberation stages [T, N, D]
            features = torch.as_tensor(data['round_node_features'], dtype=torch.float32)
            if features.dim() == 4 and features.shape[0] == 1:
                features = features.squeeze(0)
            case_data['node_features'] = features
            case_data['edges'] = data.get('edges')
            case_data['_file_path'] = str(file_path)

            case_data['reviewer_buyer_text_raw'] = data.get('reviewer_buyer_text_raw')
            case_data['reviewer_seller_text_raw'] = data.get('reviewer_seller_text_raw')
            case_id = file_path.name.split('_')[0]
            case_data['_case_id'] = case_id
            edges = load_case_edges(case_id, Path(PATH_CONFIG.get('following_network_dir', 'following_networks')),
                                    GNN_MODEL_CONFIG['graph_topology'])
            case_data['_edge_index'] = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
            
            all_samples.append((case_data, label))
        
        if self.test_ratio == 0:
            self.samples = all_samples
            print(f"✅ [{self.split}] Dataset loaded: {len(self.samples)} samples")
        else:
            split_idx = int(len(all_samples) * (1 - self.test_ratio))
            
            if self.split == 'train':
                self.samples = all_samples[:split_idx]
            else:
                self.samples = all_samples[split_idx:]
            
            print(f"✅ [{self.split}] Dataset loaded: {len(self.samples)} samples ")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_data, label = self.samples[idx]
        return case_data, label


class EndToEndTrainer:
    """
    End-to-End Trainer for Spatio-Temporal Verdict Aggregation (STVA).
    Jointly optimizes Intra-Stage Inter-Juror Interaction (ISJI),
    Cross-Stage Opinion Evolution (CSOE), and Verdict Generation Head.
    """
    def __init__(self, 
                 debate_engine_class,
                 classifier_config: Dict[str, Any] = None,
                 device: str = 'cpu',
                 learning_rate: float = None,
                 weight_decay: float = None,
                 save_dir: str = None):
        
        self.debate_engine_class = debate_engine_class
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        from .config import TRAINING_CONFIG
        self.graph_execution = TRAINING_CONFIG['graph_execution']
        
        effective_save_dir = save_dir if save_dir else PATH_CONFIG.get('model_save_dir', 'trained_models/complete')
        self.save_dir = Path(effective_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        import copy
        if classifier_config is None:
            self.classifier_config = copy.deepcopy(GNN_MODEL_CONFIG['classifier'])
        else:
            self.classifier_config = classifier_config

        print(f"🧩 Verdict Classifier Configuration: {self.classifier_config}")
            
        
       # Initialize Shared Model Manager for STVA components
        print("🔧 Initializing Shared Model Manager...")
        from .model import shared_model_manager
        shared_model_manager.initialize(device=device)
        self.shared_model_manager = shared_model_manager
        
        # Verdict Generation Head (MLP)
        self.classifier = DebateClassifier(**self.classifier_config).to(self.device)

        # Parameter collection for multitrack optimization
        print("📦 Collecting trainable parameters...")
        all_params = []

        shared_params = self.shared_model_manager.get_all_trainable_parameters()
        all_params.extend(shared_params)
        print(f"   shared parameters: {sum(p.numel() for p in shared_params):,}")
        
        classifier_params = list(self.classifier.parameters())
        all_params.extend(classifier_params)
        print(f"   classifier parameters: {sum(p.numel() for p in classifier_params):,}")
        
        self.all_params = all_params 
        
        total_trainable = sum(p.numel() for p in all_params)
        print(f"   ✅ Total trainable parameters: {total_trainable:,}")
        
        # Multitrack Learning Rate Scheduler Configuration
        lr_tracks = HYPER_PARAMS.get('multitrack_lr_config', {})
        self.lr_track_config = deepcopy(lr_tracks)
        self.warmup_epochs = HYPER_PARAMS.get('warmup_epochs', 0)
        self.warmup_start_lr = HYPER_PARAMS.get('warmup_start_lr', 1e-7)
        self.lr_schedule_start_epoch = 0

        structural_params = list(self.shared_model_manager.stva_structural.parameters()) if self.shared_model_manager.stva_structural is not None else []
        projection_params = list(self.shared_model_manager.gat_input_projection.parameters()) if self.shared_model_manager.gat_input_projection is not None else []
        temporal_params = list(self.shared_model_manager.stva_temporal.parameters()) if self.shared_model_manager.stva_temporal is not None else []
        if self.shared_model_manager.stva_projection is not None:
            temporal_params += list(self.shared_model_manager.stva_projection.parameters())
            
        classifier_params = list(self.classifier.parameters())

        optim_groups = [
            {'params': structural_params, 'lr': float(lr_tracks['stva_structural']['init_lr']), 'weight_decay': weight_decay, 'name': 'stva_structural'},
            {'params': projection_params, 'lr': float(lr_tracks['gat_input_projection']['init_lr']), 'weight_decay': weight_decay, 'name': 'gat_input_projection'},
            {'params': temporal_params, 'lr': float(lr_tracks['stva_temporal']['init_lr']), 'weight_decay': weight_decay, 'name': 'stva_temporal'},
            {'params': classifier_params, 'lr': float(lr_tracks['classifier']['init_lr']), 'weight_decay': weight_decay, 'name': 'classifier'}
        ]
        
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        self.optimizer = optim.Adam(optim_groups)
        self.all_params = all_params

        self.schedulers = []
        print("   🚀 Multitrack LR Schedulers Activated:")
        
        for group in self.optimizer.param_groups:
            name = group['name']
            cfg = lr_tracks[name]
            mode_type = str(cfg.get('type', 'step')).lower()
            
            scheduler = self._make_lr_scheduler(group, cfg)
                
            # 将调度器和对应的真实组引用打包保存
            self.schedulers.append((scheduler, group))
            print(f"      - Track [{name:<16}] | Init LR: {group['lr']:.8f} | Strategy: {mode_type.upper()}")

        print("✅ Multitrack Learning Rate Engine Initialized.\n")

        # Binary Cross-Entropy Loss for Verdict Prediction
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.failed_cases = []  
        self.val_losses = [] 
        self.val_f1s = []  
        
        base_log_path = Path(PATH_CONFIG['tensorboard_log_dir'])

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        from .config import CURRENT_CONFIG
        config_tag = CURRENT_CONFIG.get('exp_tag', '')
        
        if config_tag and str(config_tag).strip():
            folder_name = f"{timestamp}_{str(config_tag).strip()}"
        else:
            folder_name = timestamp
            
        log_path = base_log_path / folder_name

        self.writer = SummaryWriter(log_dir=str(log_path))
        print(f"📈 TensorBoard Logger initialized at: {log_path}")

    def _make_lr_scheduler(self, group, cfg):
        optimizer = optim.Adam([group])
        kind = str(cfg.get('type', 'step')).lower()
        if kind == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg.get('factor', .5),
                patience=cfg.get('patience', 15), cooldown=cfg.get('cooldown', 5),
                threshold=cfg.get('threshold', 1e-4), threshold_mode='abs',
                min_lr=cfg.get('min_lr', 1e-7))
        if kind == 'cosine':
            scheduler_class = CosineDecayLR if cfg.get('hold_min_lr', False) else optim.lr_scheduler.CosineAnnealingLR
            return scheduler_class(
                optimizer, T_max=cfg.get('T_max', 20), eta_min=cfg.get('eta_min', 1e-7))
        if kind == 'exponential':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.get('gamma', .95))
        if kind == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.get('step_size', 5), gamma=cfg.get('gamma', .8))
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)

    def _restore_lr_schedule(self, training_state, start_epoch):
        from .config import TRAINING_CONFIG
        reset = TRAINING_CONFIG.get('reset_lr_schedule', False)
        saved = (training_state or {}).get('lr_schedule_config')
        if saved and not reset:
            self.lr_track_config = saved['tracks']
            self.warmup_epochs = saved['warmup_epochs']
            self.warmup_start_lr = saved['warmup_start_lr']
            self.lr_schedule_start_epoch = saved['start_epoch']
        if reset or saved:
            self.schedulers = []
            for group in self.optimizer.param_groups:
                cfg = self.lr_track_config[group['name']]
                if reset:
                    group['lr'] = group['initial_lr'] = cfg['init_lr']
                self.schedulers.append((self._make_lr_scheduler(group, cfg), group))
        if reset:
            self.lr_schedule_start_epoch = start_epoch
            print(f"🔄 Reset LR schedule at epoch {start_epoch + 1}; Adam moments retained; "
                  f"warmup {self.warmup_epochs} epochs")
        elif training_state is not None:
            for index, ((scheduler, group), state) in enumerate(zip(self.schedulers, training_state['schedulers'])):
                scheduler.load_state_dict(state)
                group['lr'] = self.optimizer.param_groups[index]['lr']
                scheduler.optimizer.param_groups[0]['lr'] = group['lr']


    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """Loads model weights and evaluation history from checkpoint."""
        print(f"\n🔄 Loading checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found:{checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        training_state = checkpoint if 'checkpoint' in checkpoint else None
        if training_state is not None:
            checkpoint = training_state['checkpoint']

        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.shared_model_manager.load_checkpoint_weights(checkpoint)

        start_epoch = 0
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accs = checkpoint.get('train_accuracies', [])
            self.val_accs = checkpoint.get('val_accuracies', [])
            start_epoch = checkpoint.get('epoch', 0) + 1 
            self._restore_lr_schedule(training_state, start_epoch)
            if training_state is not None:
                self.val_losses = training_state['val_losses']
                self.val_f1s = training_state['val_f1s']
                random.setstate(training_state['python_rng'])
                np.random.set_state(training_state['numpy_rng'])
                torch.set_rng_state(training_state['torch_rng'].cpu())
                if training_state['cuda_rng'] is not None:
                    torch.cuda.set_rng_state_all([s.cpu() for s in training_state['cuda_rng']])
                self._loader_generator.set_state(training_state['loader_rng'].cpu())

        print("\n📊 Checkpoint Evaluation Summary:")
        print(f"   - Saved Timestamp: {checkpoint.get('timestamp', 'unknown')}")
        print(f"   - Historical Best Accuracy: {checkpoint.get('accuracy', 0.0):.4f}")
        
        if 'metrics' in checkpoint:
            m = checkpoint['metrics']
            print("   --- Metrics ---")
            print(f"   Macro-Precision: {m.get('precision', 0.0):.4f}")
            print(f"   Macro-Recall:    {m.get('recall', 0.0):.4f}")
            print(f"   Macro-F1:        {m.get('f1', m.get('macro_f1', 0.0)):.4f}")

            print(f"   Micro-F1:        {m.get('micro_f1', 0.0):.4f}")
            print(f"   Weighted-F1:     {m.get('weighted_f1', 0.0):.4f}")
        
        print(f"\n✅ Checkpoint loaded. Resuming training from Epoch {start_epoch + 1}\n")
        return start_epoch
    

    def _forward_batch(self, batch):
        features = batch['node_features'].to(self.device, non_blocking=True)
        edges = batch['edge_index'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        if self.graph_execution == 'serial':
            nodes = features.shape[2]
            representations = []
            for index, case_features in enumerate(features):
                start = index * nodes
                case_edges = edges[:, (edges[0] >= start) & (edges[0] < start + nodes)] - start
                representation, _ = self.shared_model_manager.forward_train(case_features, case_edges)
                representations.append(representation)
            return torch.cat(representations), labels
        return self.shared_model_manager.forward_batch(features, edges), labels

    def _train_step(self, train_loader, epoch) -> tuple:
        """Executes single training epoch through STVA and computes loss."""
        self.classifier.train()
        self.shared_model_manager.train_mode() 

        # Sync physical learning rates across multitrack schedulers
        if hasattr(self, 'schedulers'):
            for i, (sch, _) in enumerate(self.schedulers):
                warmup_step = epoch - self.lr_schedule_start_epoch + 1
                if 1 <= warmup_step <= self.warmup_epochs:
                    target = self.lr_track_config[self.optimizer.param_groups[i]['name']]['init_lr']
                    sch.optimizer.param_groups[0]['lr'] = self.warmup_start_lr + (
                        target - self.warmup_start_lr) * warmup_step / self.warmup_epochs
                current_sch_lr = sch.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[i]['lr'] = current_sch_lr 

        total_loss = 0
        correct = 0
        total_samples = 0

        step_logits_collector = []
        
        print(f"\n🚀 Training Epoch {epoch+1} - Total Batches: {len(train_loader)}")
        
        for batch_idx, batch in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx
            last_case_id = batch['case_ids'][-1]
            batch_inter_tensor, batch_labels_tensor = self._forward_batch(batch)
            # Predict Seller-Win logits via Verdict Generation Head
            logits = self.classifier(batch_inter_tensor)

            target_one_hot = F.one_hot(batch_labels_tensor, num_classes=2).float()
            cls_loss = self.criterion(logits, target_one_hot)

            step_logits_collector.append(logits.detach().cpu().numpy())

            loss = cls_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('Batch/Cls_Loss', cls_loss.item(), global_step)

            # Gradient norm logging for convergence monitoring
            if (batch_idx + 1) % 20 == 0:
                import logging
                logger = logging.getLogger()
                logger.info(f"\n📊 --- Batch {batch_idx+1} Gradient Norm Monitoring (Case {last_case_id}) ---")

                # 定义要检测的所有模块
                components = {
                    "stva_Structural": self.shared_model_manager.stva_structural,
                    "gat_InputProjection": self.shared_model_manager.gat_input_projection,
                    "stva_Temporal": self.shared_model_manager.stva_temporal,
                    "stva_Projection": self.shared_model_manager.stva_projection,
                    "Classifier": self.classifier
                }

                for comp_name, module in components.items():
                    if module is None: continue
                    logger.info(f"📦 {comp_name}:")
                    for name, param in module.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                norm = param.grad.norm().item()
                                logger.info(f"   - {name:<35} | Grad Norm: {norm:.6f}")
                                self.writer.add_scalar(f'Gradients/{comp_name}/{name}', norm, global_step)
                            else:
                                logger.info(f"   - {name:<35} | Grad Norm: ⚠️ None")
                logger.info(f"{'-'*60}\n")

            batch_size = batch_labels_tensor.numel()
            total_loss += loss.item() * batch_size
            
            preds = torch.argmax(logits, dim=-1)
            batch_correct = (preds == batch_labels_tensor).sum().item()
            correct += batch_correct
            total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
        
        lr_status_dict = {}
        for group in self.optimizer.param_groups:
            group_name = group.get('name', 'unknown')
            current_group_lr = group['lr']
            lr_status_dict[group_name] = current_group_lr
            self.writer.add_scalar(f'Epoch/LR_{group_name}', current_group_lr, epoch)
        
        print(f"📉 Epoch {epoch} Multi Track LR Status: "
              f"[Structural LR: {lr_status_dict.get('stva_structural', 0.0):.8f}] | "
              f"[Temporal LR: {lr_status_dict.get('stva_temporal', 0.0):.8f}] | "
              f"[Cls LR: {lr_status_dict.get('classifier', 0.0):.8f}]")

        avg_epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        
        print(f"\n✅ Epoch Training Complete - Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        return avg_epoch_loss, epoch_acc, step_logits_collector
    

    def _validate_step(self, val_loader, epoch, mode='Val') -> Tuple[float, Dict[str, float]]:
        """Evaluates STVA model performance on validation/test splits."""
        self.classifier.eval()
        self.shared_model_manager.eval_mode()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        total_cls_loss = 0
        total_samples = 0
        skipped_samples = 0
        
        print("\n🔍 Evaluating Model...")

        with torch.no_grad():
            for batch in val_loader:
                batch_inter_tensor, batch_labels_tensor = self._forward_batch(batch)
                logits = self.classifier(batch_inter_tensor)
                
                target_one_hot = F.one_hot(batch_labels_tensor, num_classes=2).float()
                cls_loss = self.criterion(logits, target_one_hot)

                batch_loss = cls_loss

                batch_size = batch_labels_tensor.numel()
                total_loss += batch_loss.item() * batch_size
                total_cls_loss += cls_loss.item() * batch_size
                total_samples += batch_size
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch['labels'].tolist())

        print(f"\n📊 Evaluation Summary: Valid Samples: {total_samples}, Skipped: {skipped_samples}")
        
        if total_samples == 0:
            return 0.0, {"accuracy": 0.0}

        avg_loss = total_loss / total_samples
        avg_cls_loss = total_cls_loss / total_samples
        acc = accuracy_score(all_labels, all_preds)
        
        # Calculate evaluation metrics (Accuracy, Macro/Micro/Weighted F1)
        macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        cm = confusion_matrix(all_labels, all_preds)

        # 打印详细汇报
        print(f"\n{'='*40}")
        print("📊 Epoch Evaluation:")
        print(f"   Average Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f})")
        print(f"   Accuracy: {acc:.4f}")
        print("   --- Macro  ---")
        print(f"   Macro-Precision: {macro_p:.4f}")
        print(f"   Macro-Recall:    {macro_r:.4f}")
        print(f"   Macro-F1:        {macro_f1:.4f}")
        print("   --- Other F1  ---")
        print(f"   Micro-F1:        {micro_f1:.4f}")
        print(f"   Weighted-F1:     {weighted_f1:.4f}")
        print("\n🧱 Confusion Matrix (0: Buyer, 1: Seller):")
        print("       Pred 0    Pred 1")
        print(f"true 0:  {cm[0][0]:<8} {cm[0][1]}")
        print(f"true 1:  {cm[1][0]:<8} {cm[1][1]}")
        print(f"{'='*40}\n")

        metrics = {
            "accuracy": acc,
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "cls_loss": avg_cls_loss
        }

        if total_samples > 0:
            self.writer.add_scalar(f'{mode}/Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{mode}/Cls_Loss', avg_cls_loss, epoch)
            self.writer.add_scalar(f'{mode}/Accuracy', metrics['accuracy'], epoch)
            self.writer.add_scalar(f'{mode}/Macro-F1', metrics['macro_f1'], epoch)
            self.writer.add_scalar(f'{mode}/Weighted-F1', metrics['weighted_f1'], epoch)
        
        return avg_loss, metrics
    
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              test_loader: Optional[DataLoader] = None, 
              num_epochs: int = 20,
              save_best: bool = True,
              validate_every: int = 1,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Runs multi-epoch training loop with validation and checkpoint saving."""
        start_epoch = 0
        best_val_acc = 0.0
        best_epoch = 0
        self._loader_generator = train_loader.generator
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from, resume_training=True)
            if self.val_accs:
                best_val_acc = max(self.val_accs)
                best_epoch = self.val_accs.index(best_val_acc) + 1
                print(f"📈 Historical Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

        print(f"🚀 Starting training pipeline for {num_epochs} epochs")
        if start_epoch > 0:
            print(f"🔄 Resuming training from Epoch {start_epoch + 1}")
        print(f"📊 Training set size: {len(train_loader.dataset)}")
        if val_loader:
            print(f"📊 Validation set size: {len(val_loader.dataset)}")

        if not hasattr(self, 'val_losses'):
            self.val_losses = []

        val_metrics = None
        train_acc = 0.0
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                print(f"\n📅 Epoch {epoch+1}/{num_epochs}")

                self.epoch_logits_pool = []

                train_loss, train_acc, epoch_logits = self._train_step(train_loader, epoch)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)

                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)

                # Logits distribution histogram logging
                if (epoch + 1) % 10 == 0 and len(epoch_logits) > 0:
                    all_logits = np.concatenate(epoch_logits, axis=0).flatten()

                    bins = [-np.inf, -5, -1, -0.5, 0, 0.5, 1, 5, np.inf]
                    labels = ['<-5', '[-5, -1)', '[-1, -0.5)', '[-0.5, 0)', '[0, 0.5)', '[0.5, 1)', '[1, 5)', '>5']

                    counts, _ = np.histogram(all_logits, bins=bins)
                    total_count = len(all_logits)
                    frequencies = counts / total_count 

                    print(f"\n📊 --- Epoch {epoch+1} Logits  Interval Distribution---")
                    for lbl, freq in zip(labels, frequencies):
                        print(f"   - Interval {lbl:<12} | Frequency : {freq*100:.2f}%")
                    print("----------------------------------------------------\n")

                    plt.figure(figsize=(10, 6))

                    bars = plt.bar(labels, frequencies, color='#2b5c8f', edgecolor='black', alpha=0.85, width=0.6)

                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)
                    
                    plt.title(f'Logits Distribution Frequency - Epoch {epoch+1}', fontsize=14, fontweight='bold')
                    plt.xlabel('Logits Range Intervals', fontsize=12)
                    plt.ylabel('Appearance Frequency (Ratio)', fontsize=12)
                    plt.ylim(0, max(frequencies) + 0.1) 
                    plt.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    logits_img_dir = Path(self.save_dir) / "logits_distributions"
                    logits_img_dir.mkdir(parents=True, exist_ok=True)
                    img_save_path = logits_img_dir / f"epoch_{epoch+1}_logits.png"
                    plt.savefig(img_save_path, dpi=200)
                    plt.close() 
                    
                    print(f"💾 Logits distribution plot saved to {img_save_path}")
                
                val_metrics = None
                if val_loader and (epoch + 1) % validate_every == 0:
                    val_loss, val_metrics = self._validate_step(val_loader, epoch, mode='Val')
                    val_acc = val_metrics['accuracy']
                    self.val_accs.append(val_acc)
                    self.val_losses.append(val_loss) 
                    
                    print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"📉 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    print(f"🎯 Val Metrics: P={val_metrics['precision']:.4f}, "
                        f"R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
                    

                    if test_loader is not None:
                        test_loss, test_metrics = self._validate_step(test_loader, epoch, mode='Test')
                        print(f"🧪 [Test Set Evaluation] Test Loss: {test_loss:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, Test Macro-F1: {test_metrics['macro_f1']:.4f}")
                        
                    # =========================================================================

                    if save_best and val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch + 1
                        self._save_checkpoint(epoch, val_acc, val_metrics, is_best=True)
                else:
                    if len(self.val_accs) > 0:
                        self.val_accs.append(self.val_accs[-1])
                        self.val_losses.append(self.val_losses[-1])
                    else:
                        self.val_accs.append(0.0)
                        self.val_losses.append(0.0)
                    print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

                # Step multitrack learning rate schedulers
                lr_tracks = self.lr_track_config
                
                for i, (sch, group) in enumerate(self.schedulers):
                    if epoch - self.lr_schedule_start_epoch >= self.warmup_epochs:
                        if isinstance(sch, optim.lr_scheduler.ReduceLROnPlateau):
                            if val_metrics is not None:
                                sch.step(val_metrics['cls_loss'])
                        else:
                            sch.step()
                    current_sch_lr = sch.optimizer.param_groups[0]['lr']
                    
                    name = group.get('name', 'unknown')
                    cfg = lr_tracks.get(name, {})
                    min_lr_gate = cfg.get('min_lr', 1e-7)

                    if current_sch_lr < min_lr_gate:
                        current_sch_lr = min_lr_gate
                        sch.optimizer.param_groups[0]['lr'] = min_lr_gate

                    self.optimizer.param_groups[i]['lr'] = current_sch_lr

                    group['lr'] = current_sch_lr

                epoch_time = time.time() - epoch_start_time
                self.writer.add_scalar('Runtime/Epoch_Seconds', epoch_time, epoch)
                self.writer.add_scalar('Runtime/Batch_Size', train_loader.batch_size, epoch)
                self.writer.flush()
                self._save_checkpoint(epoch, (val_metrics or {}).get('accuracy', train_acc),
                                      val_metrics or {}, resume=True)
                intermediate_dir = self.save_dir / 'intermediates'
                with (intermediate_dir / 'epochs.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps({'epoch': epoch + 1, 'train_loss': train_loss,
                        'train_accuracy': train_acc, 'validation': val_metrics,
                        'batch_size': train_loader.batch_size, 'seconds': epoch_time,
                        'graph_execution': self.graph_execution,
                        'learning_rates': {g['name']: g['lr'] for g in self.optimizer.param_groups}}) + '\n')
                print(f"⏱️ Epoch Execution Time: {epoch_time:.2f}s")
        finally:
            if hasattr(self, 'writer'):
                self.writer.close()
                print("📉 TensorBoard Writer Closed")

        final_metrics = val_metrics if val_metrics else {"accuracy": train_acc}
        self._save_checkpoint(num_epochs-1, 
                            final_metrics.get('accuracy', train_acc), 
                            final_metrics, 
                            is_best=False, 
                            is_final=True)

        self._save_training_results(best_val_acc, best_epoch, final_metrics)
        
        print("\n✅ Training Complete!")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        if self.failed_cases:
            print(f"\n❌ {len(self.failed_cases)} cases failed during training.")
            print(f"   Failed Case ID List: {sorted(list(set(self.failed_cases)))}")
        else:
            print("\n🎉 All cases processed successfully!")
            
        return {
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "train_loss": self.train_losses,       
            "train_acc": self.train_accs,        
            "val_loss": self.val_losses,         
            "val_acc": self.val_accs,   
            "val_f1": self.val_f1s,             
            "final_metrics": final_metrics
        }
    
    def _save_checkpoint(self, epoch: int, accuracy: float, metrics: Dict[str, float], 
                        is_best: bool = False, is_final: bool = False, resume: bool = False):
        """Saves model checkpoint containing STVA state dicts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'ablation_mode': self.shared_model_manager.ablation_mode,
            'input_dim': self.shared_model_manager.input_dim,
            'num_agents': self.shared_model_manager.num_agents,
            'temporal_layers': self.shared_model_manager.stva_config['temporal']['num_layers'],
            'classifier_state_dict': self.classifier.state_dict(),
            'gat_state_dict': self.shared_model_manager.stva_structural.state_dict(), 
            'gat_input_projection_state_dict': (
                self.shared_model_manager.gat_input_projection.state_dict()
                if self.shared_model_manager.gat_input_projection is not None else None
            ),
            'gat_input_projection_config': GNN_MODEL_CONFIG['stva'].get('gat_input_projection', {}),
            'stva_temporal_state_dict': self.shared_model_manager.stva_temporal.state_dict(), 
            'stva_projection_state_dict': (
                self.shared_model_manager.stva_projection.state_dict()
                if self.shared_model_manager.stva_projection is not None else None
            ),
            'attention_state_dict': self.shared_model_manager.stva_temporal.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'metrics': metrics,
            'model_config': self.classifier_config,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accs,
            'val_accuracies': self.val_accs,
            'timestamp': timestamp
        }
        
        if resume:
            directory = self.save_dir / 'intermediates'
            directory.mkdir(exist_ok=True)
            path = directory / 'latest_resume.pt'
            temporary = path.with_suffix('.tmp')
            torch.save({'checkpoint': checkpoint,
                'schedulers': [s.state_dict() for s, _ in self.schedulers],
                'lr_schedule_config': {'tracks': self.lr_track_config,
                    'warmup_epochs': self.warmup_epochs, 'warmup_start_lr': self.warmup_start_lr,
                    'start_epoch': self.lr_schedule_start_epoch},
                'val_losses': self.val_losses, 'val_f1s': self.val_f1s,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else None,
                'loader_rng': self._loader_generator.get_state()}, temporary)
            temporary.replace(path)
            return

        if is_best:
            filename = f"best_model_{timestamp}.pth"
            print(f"💾 Saving best model: {filename}")
        elif is_final:
            filename = f"final_model_{timestamp}.pth"
            print(f"💾 Saving best model: {filename}")
        else:
            filename = f"checkpoint_epoch_{epoch+1}_{timestamp}.pth"
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            simple_path = self.save_dir / "latest_best.pth"
            torch.save(checkpoint, simple_path)
            print("💾 Saved to latest_best.pth")

    
    def _save_training_results(self, best_val_acc: float, best_epoch: int, final_metrics: Dict[str, float]):
        """Saves structured training execution metrics to JSON."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.save_dir / f"training_results_{timestamp}.json"
        
        training_results = {
            "training_summary": {
                "best_validation_accuracy": best_val_acc,
                "best_epoch": best_epoch,
                "total_epochs": len(self.train_losses),
                "final_train_accuracy": self.train_accs[-1] if self.train_accs else 0.0,
                "final_validation_accuracy": self.val_accs[-1] if self.val_accs else 0.0,
                "final_metrics": final_metrics,
                "timestamp": timestamp
            },
            "epoch_details": [],
            "model_config": self.classifier_config
        }

        for epoch in range(len(self.train_losses)):
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": self.train_losses[epoch],
                "train_accuracy": self.train_accs[epoch] if epoch < len(self.train_accs) else None,
                "validation_accuracy": self.val_accs[epoch] if epoch < len(self.val_accs) else None,
                "is_best_epoch": (epoch + 1) == best_epoch
            }
            training_results["epoch_details"].append(epoch_info)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Detailed training results exported to {results_file}")

        latest_results_file = self.save_dir / "latest_training_results.json"
        with open(latest_results_file, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2)



class TrainingManager:
    """Manager class for initializing datasets, data loaders, and trainers."""

    def __init__(self, debate_engine_class):
        self.debate_engine_class = debate_engine_class
    
    def create_trainer(self, **trainer_kwargs) -> EndToEndTrainer:
        return EndToEndTrainer(
            debate_engine_class=self.debate_engine_class,
            **trainer_kwargs
        )
    
    def create_dataset(self, data_dir: str = None, 
                      train_dir: str = None, 
                      val_dir: str = None,
                      **dataset_kwargs) -> Tuple[DebateDataset, DebateDataset]:
        if train_dir and val_dir:
            print("📁 Using standalone dataset directories:")
            print(f"   Train: {train_dir}")
            print(f"   Val:   {val_dir}")
            
            train_dataset = DebateDataset(train_dir, split='train', test_ratio=0, **dataset_kwargs)
            val_dataset = DebateDataset(val_dir, split='val', test_ratio=0, **dataset_kwargs)
        elif data_dir:
            print(f"📁 Using single directory ratio split: {data_dir}")
            train_dataset = DebateDataset(data_dir, split='train', **dataset_kwargs)
            val_dataset = DebateDataset(data_dir, split='test', **dataset_kwargs)
        else:
            raise ValueError("Must provide data_dir or (train_dir + val_dir)")
        
        return train_dataset, val_dataset

    def create_data_loaders(self, train_dataset, val_dataset, batch_size=None):
        if batch_size is None:
            batch_size = HYPER_PARAMS['batch_size']

        g = torch.Generator()
        g.manual_seed(42)
        
        loader_options = data_loader_options()
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_graphs,
            **loader_options,
            worker_init_fn=seed_worker,
            generator=g,                
            drop_last=len(train_dataset) % batch_size == 1
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_graphs,
            **loader_options,
            drop_last=False
        )
        return train_loader, val_loader
    
    def run_full_training(self, 
                         data_dir: str = None,
                         train_dir: str = None,
                         val_dir: str = None,
                         num_epochs: int = 20,
                         learning_rate: float = 1e-4,
                         weight_decay: float = 1e-3,
                         save_dir: str = None,
                         device: str = 'cpu',
                         validate_every: int = 5,
                         resume_from: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        
        print("🎯 Starting full end-to-end training pipeline...")
        
        if resume_from:
            print(f"🔄 Checkpoint resume mode: Loading from {resume_from}")
        
        print("📊 Creating datasets...")
        train_dataset, val_dataset = self.create_dataset(
            data_dir=data_dir,
            train_dir=train_dir,
            val_dir=val_dir
        )
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
        
        trainer = self.create_trainer(
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_dir=save_dir if save_dir else PATH_CONFIG['model_save_dir']
        )

        test_dir = kwargs.get('test_dir', PATH_CONFIG.get('test_dir'))
        test_loader = None
        if test_dir and Path(test_dir).exists():
            print(f"📁 Load test dataset: {test_dir}")
            test_dataset = DebateDataset(test_dir, split='test', test_ratio=0)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=HYPER_PARAMS['batch_size'], 
                shuffle=False, 
                collate_fn=collate_graphs,
                **data_loader_options(),
                drop_last=False
            )
        else:
            print(f"ℹ️ Default Test Loader: path {test_dir} is not valid. Skipping test loader creation.")
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  
            num_epochs=num_epochs,
            validate_every=validate_every,
            resume_from=resume_from
        )
        
        print("🎉 Training completed!")
        return results
