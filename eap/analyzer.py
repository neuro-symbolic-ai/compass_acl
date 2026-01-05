import os
from functools import partial
from typing import Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .graph import Graph
from .attribute import attribute
from .evaluate import evaluate_graph, evaluate_baseline

from .config import EAPConfig
from .cnp import CNPDataset, cnp_metric
from .visualization import EAPVisualizer


class EAPAnalyzer:
    """EAP analysis following TRACE conventions"""

    def __init__(self, config: Optional[EAPConfig] = None):
        self.config = config or EAPConfig()
        self.device = self.config.device

        # Initialize visualizer
        self.visualizer = EAPVisualizer(
            self.config.log_dir, self.config
        ) if self.config.log_dir else None

        # Storage for results
        self.cnp_dataset = None

    def load_cnp_dataset(self, dataset_path, tokenizer):
        """Load CNP dataset if path provided"""

        class EAPDataset(Dataset):
            def __init__(self, filepath):
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                elif filepath.endswith('.json'):
                    import json
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.df = pd.DataFrame(data)
                else:
                    raise ValueError(f"Unsupported file format: {filepath}")

                # Validate required columns
                required_cols = ['clean', 'corrupted', 'correct_idx', 'incorrect_idx']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"EAP dataset missing required columns: {missing_cols}")

            def __len__(self):
                return len(self.df)

            def __getitem__(self, index):
                row = self.df.iloc[index]
                return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]

            def to_dataloader(self, batch_size):
                def collate_fn(xs):
                    clean, corrupted, labels = zip(*xs)
                    return list(clean), list(corrupted), torch.tensor(labels)

                return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

        return EAPDataset(dataset_path)

    def analyze(self, model, batch_loader=None, tokenizer=None, model_name: str = "", step: int = 0) -> (Dict[str, Any], Optional[Graph]):
        """Run EAP analysis on model"""

        print(f"Running EAP analysis with method: {self.config.method}")

        if not self.config.cnp_dataset_path:
            print("No CNP dataset path provided - using batch data for EAP analysis")
            return {}, None

        self.cnp_dataset = self.load_cnp_dataset(self.config.cnp_dataset_path, tokenizer)
        if len(self.cnp_dataset) == 0:
            print("EAP dataset is empty - skipping analysis")
            return {}, None

        eap_dataloader = self.cnp_dataset.to_dataloader(self.config.eap_batch_size or 10)

        # Create computational graph
        graph = Graph.from_model(model)

        # Run EAP attribution
        attribute(
            model, graph, eap_dataloader,
            partial(cnp_metric, loss=True, mean=True),
            method=self.config.method,
            ig_steps=self.config.ig_steps,
            track_last_token_only=self.config.track_last_token_only
        )

        # Apply top-k selection
        graph.apply_topn(self.config.top_k, True)

        # ========================================
        # EVALUATE THREE CONDITIONS
        # ========================================

        # 1. Full model performance M(E)
        baseline_performance = evaluate_baseline(
            model, eap_dataloader, partial(cnp_metric, loss=False, mean=False)
        ).mean().item()

        # 2. Circuit performance M(C)
        circuit_performance = evaluate_graph(
            model, graph, eap_dataloader, partial(cnp_metric, loss=False, mean=False)
        ).mean().item()

        # 3. Ablated performance M(∅) - empty graph (everything ablated)
        empty_graph = Graph.from_model(model)
        empty_graph.reset(empty=True)  # Everything disabled
        ablated_performance = evaluate_graph(
            model, empty_graph, eap_dataloader, partial(cnp_metric, loss=False, mean=False),
            skip_clean=False
        ).mean().item()

        # ========================================
        # COMPUTE BOTH FAITHFULNESS METRICS
        # ========================================

        # Paper definition: normalized by discriminative capacity above chance
        if baseline_performance != ablated_performance:
            faithfulness_paper = (circuit_performance - ablated_performance) / \
                                 (baseline_performance - ablated_performance)
        else:
            faithfulness_paper = 0.0
            print("Warning: baseline == ablated, cannot compute paper faithfulness")

        # Simple ratio definition
        faithfulness_simple = circuit_performance / baseline_performance \
            if baseline_performance != 0 else 0.0

        # ========================================
        # RESULTS DICTIONARY
        # ========================================
        results = {
            "method": self.config.method,
            "top_k": self.config.top_k,

            # Raw performances
            "baseline_performance": baseline_performance,
            "circuit_performance": circuit_performance,
            "ablated_performance": ablated_performance,

            # Both faithfulness metrics
            "faithfulness": faithfulness_paper,  # Primary (matches paper)
            "faithfulness_simple": faithfulness_simple,  # Secondary (for comparison -simple baseline)

            # Circuit statistics
            "num_edges_in_circuit": sum(1 for edge in graph.edges.values() if edge.in_graph),
            "num_nodes_in_circuit": sum(1 for node in graph.nodes.values() if node.in_graph),

            # Derived metrics
            "performance_drop": baseline_performance - circuit_performance,
            "performance_drop_pct": (baseline_performance - circuit_performance) / baseline_performance * 100
            if baseline_performance != 0 else 0,
        }

        # Save graphs if requested
        if self.config.save_graphs and self.config.log_dir:
            self._save_graphs(graph, model_name, step)

        if self.config.plot_circuit_step_graph:
            print("Saving circuit graph visualization")
            os.makedirs(os.path.join(self.config.log_dir, 'eap_graphs'), exist_ok=True)
            graph.to_image(os.path.join(self.config.log_dir, f'eap_graphs/{model_name}_graph_step_{step}.png'))

        return results, graph

    def _save_graphs(self, graph, model_name: str, step:int):
        """Save EAP graphs using existing EAP library functions"""
        graph_dir = os.path.join(self.config.log_dir, 'eap_graphs')
        os.makedirs(graph_dir, exist_ok=True)

        if self.config.save_json:
            json_path = os.path.join(graph_dir, f'{model_name}_graph_step_{step}.json')
            graph.to_json(json_path)

        if self.config.save_pt:
            pt_path = os.path.join(graph_dir, f'{model_name}_graph_step_{step}.pt')
            graph.to_pt(pt_path)