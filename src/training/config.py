"""
Configuration management for lung disease classification.
Centralizes hyperparameters and settings.
"""

import yaml
import os

# Default class names for NIH ChestX-ray14 dataset
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Class weights (calculated from dataset statistics)
DEFAULT_POS_WEIGHTS = [
    8.70, 39.39, 7.42, 4.64, 18.39, 16.71, 77.35,
    20.15, 23.02, 47.68, 43.56, 65.50, 32.12, 492.92
]

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'pin_memory': True,
    'use_amp': True,  # Automatic mixed precision
    
    # Model settings
    'model_type': 'attention_densenet',  # densenet, attention_densenet, multiscale, ensemble
    'attention_type': 'cbam',  # cbam, se, ca
    'pretrained': True,
    
    # Loss settings
    'loss_type': 'asl',  # bce, focal, asl, combined
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'asl_gamma_neg': 4,
    'asl_gamma_pos': 1,
    'asl_clip': 0.05,
    
    # Optimizer settings
    'optimizer': 'adam',  # adam, adamw, sgd
    'momentum': 0.9,  # for SGD
    
    # Scheduler settings
    'scheduler': 'cosine',  # cosine, step, plateau, none
    'scheduler_t0': 10,  # for CosineAnnealingWarmRestarts
    'scheduler_t_mult': 2,
    'scheduler_eta_min': 1e-6,
    'step_size': 5,  # for StepLR
    'gamma': 0.1,  # for StepLR
    
    # Early stopping
    'early_stopping': True,
    'patience': 5,
    'min_delta': 0.001,
    
    # Gradient clipping
    'clip_grad_norm': 1.0,
    
    # Data augmentation
    'use_advanced_aug': True,
    'use_mixup': False,
    'mixup_alpha': 0.2,
    'use_cutmix': False,
    'cutmix_alpha': 1.0,
    
    # Image settings
    'image_size': 224,
    
    # Dataset split
    'train_split': 0.8,
    'val_split': 0.2,
    'random_seed': 42,
}

# Evaluation configuration
EVAL_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'use_tta': False,
    'tta_n_augs': 5,
    'viz_method': 'gradcam++',  # gradcam++, scorecam, gradcam
    'num_viz_samples': 10,
    'optimize_threshold': True,
    'threshold_metric': 'f1',  # f1, youden, precision, recall
    'save_visualizations': True,
    'generate_report': True,
}

# Paths
PATHS = {
    'data_dir': './data',
    'csv_file': './data/Data_Entry_2017.csv',
    'models_dir': './models',
    'results_dir': './results',
    'viz_dir': './visualizations',
    'logs_dir': './logs',
}


class Config:
    """Configuration class for easy access and updates."""
    
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        
        self.class_names = config_dict.get('class_names', CLASS_NAMES)
        self.pos_weights = config_dict.get('pos_weights', DEFAULT_POS_WEIGHTS)
        self.train = config_dict.get('train', TRAIN_CONFIG.copy())
        self.eval = config_dict.get('eval', EVAL_CONFIG.copy())
        self.paths = config_dict.get('paths', PATHS.copy())
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
    
    def save(self, path='config.yaml'):
        """Save configuration to YAML file."""
        config_dict = {
            'class_names': self.class_names,
            'pos_weights': self.pos_weights,
            'train': self.train,
            'eval': self.eval,
            'paths': self.paths,
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path='config.yaml'):
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            print(f"Config file {path} not found. Using defaults.")
            return cls()
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"Configuration loaded from {path}")
        return cls(config_dict)
    
    def __repr__(self):
        return f"Config(model={self.train['model_type']}, loss={self.train['loss_type']}, epochs={self.train['num_epochs']})"


def create_directories(config):
    """Create necessary directories based on config."""
    for key, path in config.paths.items():
        if key != 'csv_file' and key != 'data_dir':
            os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Test config
    config = Config()
    print(config)
    print("\nTrain config:", config.train)
    print("\nEval config:", config.eval)
    
    # Save and load
    config.save('test_config.yaml')
    loaded_config = Config.load('test_config.yaml')
    print("\nLoaded config:", loaded_config)
    
    # Clean up
    os.remove('test_config.yaml')
    print("\nConfig system working correctly!")
