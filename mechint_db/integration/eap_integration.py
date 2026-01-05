
import os
from typing import Dict, Any, Optional
from pathlib import Path

from eap.graph import Graph
from eap.config import EAPConfig
from eap.analyzer import EAPAnalyzer
from ..storage.owl_exporter import OWLExporter


class MechIntEAPConfig(EAPConfig):
    """Extended EAP config with OWL export options"""

    def __init__(self, *args, **kwargs):
        # Separate MechInt-specific kwargs from EAPConfig kwargs
        mechint_kwargs = {
            'save_owl': kwargs.pop('save_owl', True),
            'owl_output_dir': kwargs.pop('owl_output_dir', "./data/experiments"),
            'owl_base_uri': kwargs.pop('owl_base_uri', "http://purl.org/mechint"),
            'collect_detailed_metadata': kwargs.pop('collect_detailed_metadata', True),
            'experiment_description': kwargs.pop('experiment_description', None),
            'researcher_name': kwargs.pop('researcher_name', None),
        }

        # Call parent with remaining kwargs
        super().__init__(*args, **kwargs)
        # Ensure log_dir is set if save_owl is True
        if mechint_kwargs['save_owl'] and not self.log_dir:
            self.log_dir = './analysis_results'
            os.makedirs(self.log_dir, exist_ok=True)
        # Set MechInt-specific attributes
        for attr, value in mechint_kwargs.items():
            setattr(self, attr, value)

    def __post_init__(self):
        """Validate extended configuration"""
        super().__post_init__()

        # if self.save_owl and not self.log_dir:
        #     raise ValueError("log_dir must be set when save_owl is enabled")


class MechIntEAPAnalyzer(EAPAnalyzer):
    """Extended EAP analyzer with OWL export capabilities"""

    def __init__(self, config: Optional[EAPConfig] = None,
                 save_owl: bool = True,
                 owl_output_dir: str = "./data/experiments",
                 owl_base_uri: str = "http://purl.org/mechint",
                 experiment_description: Optional[str] = None,
                 researcher_name: Optional[str] = None):

        super().__init__(config)

        # Store OWL export settings
        self.save_owl = save_owl
        self.owl_output_dir = owl_output_dir
        self.owl_base_uri = owl_base_uri
        self.experiment_description = experiment_description
        self.researcher_name = researcher_name

        # Initialize OWL exporter if enabled
        if self.save_owl:
            self.owl_exporter = OWLExporter(
                base_uri=owl_base_uri,
                output_dir=owl_output_dir
            )
        else:
            self.owl_exporter = None

    def analyze(self, model, batch_loader=None, tokenizer=None,
                model_name: str = "", step: int = 0,
                experiment_id: Optional[str] = None) -> tuple[Dict[str, Any], Optional[Graph]]:
        """
        Enhanced analyze method with OWL export

        Args:
            model: The transformer model
            batch_loader: Data loader (for TRACE compatibility, but CNP dataset preferred)
            tokenizer: Tokenizer for CNP dataset
            model_name: Name for result storage
            step: Training step number
            experiment_id: Unique experiment identifier for OWL export

        Returns:
            Tuple of (analysis_results, graph)
        """

        # Run original EAP analysis
        print(f"Running EAP analysis with method: {self.config.method}")

        if not self.config.cnp_dataset_path:
            print("No CNP dataset path provided - using batch data for EAP analysis")
            return {}, None

        # Load dataset and run analysis (same as original)
        self.cnp_dataset = self.load_cnp_dataset(self.config.cnp_dataset_path, tokenizer)
        if len(self.cnp_dataset) == 0:
            print("EAP dataset is empty - skipping analysis")
            return {}, None

        eap_dataloader = self.cnp_dataset.to_dataloader(self.config.eap_batch_size or 10)

        # Create computational graph
        graph = Graph.from_model(model)

        # Run EAP attribution (import here to avoid circular import)
        from eap.attribute import attribute
        from eap.cnp import cnp_metric
        from functools import partial

        attribute(
            model, graph, eap_dataloader,
            partial(cnp_metric, loss=True, mean=True),
            method=self.config.method,
            ig_steps=self.config.ig_steps,
            track_last_token_only=self.config.track_last_token_only,
        )

        # Apply top-k selection
        graph.apply_topn(self.config.top_k, True)

        # Evaluate performance
        from eap.evaluate import evaluate_baseline, evaluate_graph

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

        # Calculate faithfulness
        # faithfulness = circuit_performance / baseline_performance if baseline_performance != 0 else 0
        faithfulness = (circuit_performance - ablated_performance) / \
                                 (baseline_performance - ablated_performance)
        faithfulness_simple = circuit_performance / baseline_performance \
            if baseline_performance != 0 else 0.0

        # Collect analysis results
        results = {
            "method": self.config.method,
            "top_k": self.config.top_k,
            "baseline_performance": baseline_performance,
            "circuit_performance": circuit_performance,
            "faithfulness": faithfulness,
            "faithfulness_simple": faithfulness_simple,
            "num_edges_in_circuit": sum(1 for edge in graph.edges.values() if edge.in_graph),
            "num_nodes_in_circuit": sum(1 for node in graph.nodes.values() if node.in_graph),
            "track_last_token_only": self.config.track_last_token_only,
        }

        # Save original formats if requested
        if self.config.save_graphs and self.config.log_dir:
            self._save_graphs(graph, model_name, step)

        # Generate circuit visualization if requested
        if self.config.plot_circuit_step_graph:
            print("Saving circuit graph visualization")
            os.makedirs(os.path.join(self.config.log_dir, 'eap_graphs'), exist_ok=True)
            graph.to_image(os.path.join(self.config.log_dir, f'eap_graphs/{model_name}_graph_step_{step}.png'))

        # Export to OWL if enabled
        if self.config.save_owl and self.owl_exporter:
            try:
                # Generate experiment_id if not provided
                if experiment_id is None:
                    experiment_id = f"{model_name}_{self.config.method}_{self.config.top_k}"

                # Collect comprehensive metadata
                metadata = self._collect_metadata(model, model_name, step)
                config_dict = self._extract_config_dict(model)

                print(f"Exporting circuit to OWL for experiment: {experiment_id}")
                owl_path = self.owl_exporter.export_circuit(
                    graph=graph,
                    experiment_id=experiment_id,
                    step=step,
                    config=config_dict,
                    metadata=metadata,
                    eap_results=results
                )

                # Add OWL path to results
                results["owl_export_path"] = owl_path
                print(f"Successfully exported to OWL: {owl_path}")

            except Exception as e:
                print(f"Failed to export to OWL: {e}")
                # Don't fail the entire analysis due to OWL export issues
                results["owl_export_error"] = str(e)

        return results, graph

    def _collect_metadata(self, model, model_name: str, step: int) -> Dict[str, Any]:
        """Collect comprehensive metadata for OWL export"""
        metadata = {
            "model_name": model_name,
            "step": step,
            "eap_method": self.config.method,
            "ig_steps": self.config.ig_steps,
            "top_k": self.config.top_k,
        }

        # Add config-specific metadata
        if self.config.experiment_description:
            metadata["description"] = self.config.experiment_description
        if self.config.researcher_name:
            metadata["researcher"] = self.config.researcher_name

        # Add model architecture info if available
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            metadata.update({
                "n_layers": getattr(cfg, 'n_layers', None),
                "d_model": getattr(cfg, 'd_model', None),
                "n_heads": getattr(cfg, 'n_heads', None),
                "model_type": getattr(cfg, 'model_type', None),
            })

        return metadata

    def _extract_config_dict(self, model) -> Dict[str, Any]:
        """Extract configuration dictionary for OWL export"""
        config_dict = {
            "method": self.config.method,
            "ig_steps": self.config.ig_steps,
            "top_k": self.config.top_k,
            "intervention_type": self.config.intervention_type,
        }

        # Add model config if available
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            model_config = {
                "model_type": getattr(cfg, 'model_type', None),
                "n_layers": getattr(cfg, 'n_layers', None),
                "d_model": getattr(cfg, 'd_model', None),
                "n_heads": getattr(cfg, 'n_heads', None),
                "d_ff": getattr(cfg, 'd_ff', None),
            }
            # Only add non-None values
            config_dict.update({k: v for k, v in model_config.items() if v is not None})

        return config_dict

    def export_temporal_evolution(self, experiment_results: Dict[int, tuple],
                                  experiment_id: str) -> Optional[str]:
        """
        Export temporal evolution of circuits across multiple steps

        Args:
            experiment_results: Dict mapping step -> (results, graph) tuples
            experiment_id: Unique experiment identifier

        Returns:
            Path to temporal evolution TTL file
        """
        if not self.config.save_owl or not self.owl_exporter:
            return None

        try:
            # Extract graphs from results
            graphs = {step: graph for step, (results, graph) in experiment_results.items()
                      if graph is not None}

            if not graphs:
                print("No graphs available for temporal evolution export")
                return None

            evolution_path = self.owl_exporter.export_temporal_evolution(graphs, experiment_id)
            print(f"Exported temporal evolution to: {evolution_path}")
            return evolution_path

        except Exception as e:
            print(f"Failed to export temporal evolution: {e}")
            return None


# Convenience functions for easy integration
def create_mechint_eap_analyzer(
        method: str = 'EAP-IG-inputs',
        top_k: int = 50,
        ig_steps: int = 5,
        cnp_dataset_path: Optional[str] = None,
        save_owl: bool = True,
        owl_output_dir: str = "./data/experiments",
        experiment_description: Optional[str] = None,
        **kwargs
) -> MechIntEAPAnalyzer:
    """
    Create a MechInt-enabled EAP analyzer with reasonable defaults

    Args:
        method: EAP attribution method
        top_k: Number of top edges to keep in circuit
        ig_steps: Number of integrated gradients steps
        cnp_dataset_path: Path to CNP dataset
        save_owl: Whether to export to OWL format
        owl_output_dir: Directory for OWL exports
        experiment_description: Description for metadata
        **kwargs: Additional config parameters

    Returns:
        Configured MechIntEAPAnalyzer
    """
    config = MechIntEAPConfig(
        method=method,
        top_k=top_k,
        ig_steps=ig_steps,
        cnp_dataset_path=cnp_dataset_path,
        save_owl=save_owl,
        owl_output_dir=owl_output_dir,
        experiment_description=experiment_description,
        **kwargs
    )

    return MechIntEAPAnalyzer(config)


def analyze_with_owl_export(
        model,
        cnp_dataset_path: str,
        experiment_id: str,
        step: int = 0,
        method: str = 'EAP-IG-inputs',
        top_k: int = 50,
        tokenizer=None,
        **kwargs
) -> tuple[Dict[str, Any], Optional[Graph], Optional[str]]:
    """
    Convenience function to run EAP analysis with OWL export

    Returns:
        Tuple of (results, graph, owl_path)
    """
    analyzer = create_mechint_eap_analyzer(
        method=method,
        top_k=top_k,
        cnp_dataset_path=cnp_dataset_path,
        **kwargs
    )

    results, graph = analyzer.analyze(
        model=model,
        tokenizer=tokenizer,
        experiment_id=experiment_id,
        step=step
    )

    owl_path = results.get("owl_export_path")
    return results, graph, owl_path