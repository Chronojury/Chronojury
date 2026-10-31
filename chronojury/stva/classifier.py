
"""
Verdict Generation Module for ChronoJury.
Predicts seller-win probabilities based on aggregated spatio-temporal representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path


class DebateClassifier(nn.Module):
    """
    Verdict Generation Classifier Head.
    Maps the final spatio-temporal verdict vector h_final into decision space (Section 3.2).
    """
    
    def __init__(self, type: str = 'linear', input_dim: int = 512, hidden_dim: int = 128, num_classes: int = 2, dropout_rate: float = 0.0):
        super(DebateClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if str(type).lower() == 'linear':
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, num_classes)
            )
            
        elif str(type).lower() == 'mlp':
            
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  
                nn.GELU(),                   
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError(f"Unsupported classifier architecture: {type}. Choose 'linear' or 'mlp'.")
        
        self._init_weights()

    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, interactive_representation: torch.Tensor) -> torch.Tensor:
        return self.classifier(interactive_representation)
    
    def predict(self, interactive_representation: torch.Tensor) -> Dict[str, Any]:
        """
        Computes decision logits, softmax probability distribution, and predicted outcome.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(interactive_representation)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            result = {
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist(),
                "predicted_class": predicted_class.cpu().numpy().tolist(),
                "confidence": confidence.cpu().numpy().tolist(),
                "class_names": ["买家胜出", "卖家胜出"],
                "raw_logits": logits.cpu(),
                "raw_probabilities": probabilities.cpu()
            }

            if logits.size(0) == 1:
                result["predicted_class"] = result["predicted_class"][0]
                result["confidence"] = result["confidence"][0]
                result["logits"] = result["logits"][0]
                result["probabilities"] = result["probabilities"][0]
            
            return result


class DebateClassificationManager:
    """
    Manager class for initializing, saving, loading, and performing predictions with DebateClassifier.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.classifier = None
        self.model_path = model_path
        
        if model_path:
            self.load_classifier(model_path)
        else:
            self._init_default_classifier()
    
    def _init_default_classifier(self):
        """Initializes default MLP classifier head."""
        print("🔧 Initializes default MLP..")
        self.classifier = DebateClassifier(
            input_dim=512,
            hidden_dim=128,
            num_classes=2,
            dropout_rate=0.3
        ).to(self.device)
        print("✅ Default classifier initialized successfully")
    
    def load_classifier(self, model_path: str):
        """Restore the checkpoint architecture and require every learned weight."""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('classifier_state_dict', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise ValueError("Checkpoint must contain classifier_state_dict or model_state_dict")
        config = dict(checkpoint['model_config'])
        # Older classifier-only checkpoints omit type; infer it from their layers.
        config.setdefault('type', 'mlp' if 'classifier.4.weight' in state_dict else 'linear')
        self.classifier = DebateClassifier(**config).to(self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        print(f"Classifier loaded from: {model_path}")

    def save_classifier(self, save_path: str, accuracy: float = None, metadata: Dict = None):
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.classifier.state_dict(),
            'model_config': {
                'input_dim': self.classifier.input_dim,
                'hidden_dim': self.classifier.hidden_dim,
                'num_classes': self.classifier.num_classes,
                'dropout_rate': self.classifier.dropout_rate
            },
            'timestamp': torch.tensor(0).item(),  
            'device': str(self.device)
        }
        
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, save_path)
        print(f"💾 Classifier saved to {save_path}")
        

    def predict(self, interactive_representation) -> Dict[str, Any]:
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized.")
        
        if not isinstance(interactive_representation, torch.Tensor):
            if isinstance(interactive_representation, (list, np.ndarray)):
                interactive_representation = torch.tensor(interactive_representation, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported input type: {type(interactive_representation)}")

        interactive_representation = interactive_representation.to(self.device)

        result = self.classifier.predict(interactive_representation)

        if isinstance(result["predicted_class"], (int, float)):
            class_name = result["class_names"][int(result["predicted_class"])]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            
            print(f"🏆 Classification result: {class_name}")
            print(f"📊 Confidence: {confidence:.4f}")
            print(f"📈 Probability distribution: Buyer {probabilities[0]:.4f}, Seller {probabilities[1]:.4f}")
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model parameter count and configuration metadata."""
        if self.classifier is None:
            return {"status": "Uninitialized"}
        
        total_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            "status": "Initialized",
            "input_dim": self.classifier.input_dim,
            "hidden_dim": self.classifier.hidden_dim,
            "num_classes": self.classifier.num_classes,
            "dropout_rate": self.classifier.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device)
        }

