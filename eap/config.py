from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import torch


@dataclass
class EAPConfig:
    """Configuration for EAP analysis"""

    # Core EAP parameters
    Method = Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted',
    'EAP-IG-activations']
    method: Method = 'EAP-IG-inputs'
    ig_steps: int = 5
    top_k: int = 50

    # CNP dataset configuration
    cnp_dataset_path: Optional[str] = None
    # intervention_types: List[str] = field(default_factory=lambda: ['role_cross'])
    intervention_type: str ='role_cross'

    semantic_roles: List[str] = field(default_factory=lambda: ['Agent', 'Patient', 'Theme'])

    # Analysis settings
    eap_batch_size: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output settings
    log_dir: Optional[str] = './analysis_results'
    save_graphs: bool = True
    save_json: bool = True
    save_pt: bool = True
    show_plots: bool = False
    plot_circuit_step_graph: bool = True

    track_last_token_only: bool = False # Whether to track only the last token in sequences
    def __post_init__(self):
        """Validate configuration"""
        if self.method not in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'clean-corrupted']:
            raise ValueError(f"Invalid EAP method: {self.method}")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")