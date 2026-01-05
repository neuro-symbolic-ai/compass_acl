import os
from typing import Dict, Optional, Any, Union
from pathlib import Path
import rdflib
from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, PROV
from datetime import datetime

from eap.graph import Graph, Node, Edge
from ..ontology import MECHINT_SCHEMA, TEMPORAL_SCHEMA, PATTERN_SCHEMA


class OWLExporter:
    """Exports EAP circuits to OWL/RDF following MechInt ontology"""

    def __init__(self, base_uri: str = "http://purl.org/mechint",
                 output_dir: str = "./data/experiments"):
        self.base_uri = base_uri
        self.output_dir = Path(output_dir)
        self.schema = MECHINT_SCHEMA
        self.temporal = TEMPORAL_SCHEMA
        self.patterns = PATTERN_SCHEMA

        # Set up namespace bindings
        self.setup_namespaces()

    def setup_namespaces(self):
        """Set up standard namespace bindings"""
        self.namespaces = {
            'mechint': self.schema.namespace,
            'rdf': RDF,
            'rdfs': RDFS,
            'xsd': XSD,
            'prov': PROV
        }

    def export_circuit(self,
                       graph: Graph,
                       experiment_id: str,
                       step: int,
                       config: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       eap_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Export EAP Graph to OWL/RDF format

        Args:
            graph: EAP Graph object
            experiment_id: Unique experiment identifier
            step: Training step number
            config: Training configuration dict
            metadata: Additional metadata
            eap_results: EAP analysis results (faithfulness, performance, etc.)

        Returns:
            Path to saved TTL file
        """

        # Create RDF graph
        rdf_graph = RDFGraph()
        for prefix, namespace in self.namespaces.items():
            rdf_graph.bind(prefix, namespace)

        # Create experiment and run context
        exp_uri = self.schema.namespace[f"experiment_{experiment_id}"]
        run_uri = self.schema.namespace[f"run_{experiment_id}_{step}"]

        # Add experiment and run
        rdf_graph.add((exp_uri, RDF.type, self.schema.classes['Experiment']))
        rdf_graph.add((run_uri, RDF.type, self.schema.classes['Run']))
        rdf_graph.add((run_uri, self.schema.object_properties['ofExperiment'], exp_uri))
        rdf_graph.add((run_uri, self.schema.data_properties['step'],
                       Literal(step, datatype=XSD.integer)))
        rdf_graph.add((run_uri, self.schema.data_properties['timestamp'],
                       Literal(datetime.now(), datatype=XSD.dateTime)))

        # Add configuration metadata
        if config:
            self._add_config_metadata(rdf_graph, run_uri, config)

        # Add EAP results metadata
        if eap_results:
            self._add_eap_results(rdf_graph, run_uri, eap_results)

        # Export nodes
        node_uris = self._export_nodes(rdf_graph, graph, run_uri, experiment_id, step)

        # Export edges
        edge_uris = self._export_edges(rdf_graph, graph, run_uri, experiment_id, step, node_uris)

        # Export circuit as unified entity
        circuit_uri = self._export_circuit_entity(rdf_graph, graph, run_uri,
                                                  experiment_id, step, node_uris, edge_uris)

        # Save to file
        output_path = self._save_rdf(rdf_graph, experiment_id, step)

        return str(output_path)

    def _export_nodes(self, rdf_graph: RDFGraph, graph: Graph,
                      run_uri: URIRef, experiment_id: str, step: int) -> Dict[str, URIRef]:
        """Export graph nodes to RDF"""
        node_uris = {}

        for node_name, node in graph.nodes.items():
            # Create node URI
            safe_node_name = node_name.replace('.', '_')
            node_uri = self.schema.namespace[f"node_{experiment_id}_{step}_{safe_node_name}"]
            node_uris[node_name] = node_uri

            # Add basic node properties
            rdf_graph.add((node_uri, RDF.type, self.schema.classes['Node']))
            rdf_graph.add((node_uri, self.schema.object_properties['inRun'], run_uri))
            rdf_graph.add((node_uri, self.schema.data_properties['address'], Literal(node_name)))
            rdf_graph.add((node_uri, self.schema.data_properties['inGraph'], Literal(node.in_graph)))

            # Add node score if available
            if hasattr(node, 'score') and node.score is not None:
                rdf_graph.add((node_uri, self.schema.data_properties['score'],
                               Literal(float(node.score))))

            # Create and link component
            component_uri = self._create_component(rdf_graph, node_name, experiment_id, node)
            rdf_graph.add((node_uri, self.schema.object_properties['aboutComponent'], component_uri))

        return node_uris

    def _create_component(self, rdf_graph: RDFGraph, node_name: str,
                          experiment_id: str, node: Node) -> URIRef:
        """Create component entity for a node"""
        safe_node_name = node_name.replace('.', '_')
        component_uri = self.schema.namespace[f"component_{experiment_id}_{safe_node_name}"]

        # Get component type from schema
        component_type = self.schema.get_component_type_for_node(node_name)
        rdf_graph.add((component_uri, RDF.type, component_type))

        # Parse and add structural information (layer, head)
        structure_info = self.schema.parse_node_structure(node_name)

        if 'layer' in structure_info:
            rdf_graph.add((component_uri, self.schema.data_properties['layer'],
                           Literal(structure_info['layer'], datatype=XSD.integer)))

        if 'head' in structure_info:
            rdf_graph.add((component_uri, self.schema.data_properties['head'],
                           Literal(structure_info['head'], datatype=XSD.integer)))

        return component_uri

    def _export_edges(self, rdf_graph: RDFGraph, graph: Graph,
                      run_uri: URIRef, experiment_id: str, step: int,
                      node_uris: Dict[str, URIRef]) -> Dict[str, URIRef]:
        """Export graph edges to RDF"""
        edge_uris = {}

        for edge_name, edge in graph.edges.items():
            # Create edge URI (sanitize edge name)
            safe_edge_name = (edge_name.replace('->', '_to_')
                              .replace('<', '_')
                              .replace('>', '')
                              .replace('.', '_'))
            edge_uri = self.schema.namespace[f"edge_{experiment_id}_{step}_{safe_edge_name}"]
            edge_uris[edge_name] = edge_uri

            # Add basic edge properties
            rdf_graph.add((edge_uri, RDF.type, self.schema.classes['CircuitEdge']))
            rdf_graph.add((edge_uri, self.schema.object_properties['inRun'], run_uri))
            rdf_graph.add((edge_uri, self.schema.object_properties['hasSource'],
                           node_uris[edge.parent.name]))
            rdf_graph.add((edge_uri, self.schema.object_properties['hasTarget'],
                           node_uris[edge.child.name]))
            rdf_graph.add((edge_uri, self.schema.data_properties['score'],
                           Literal(float(edge.score))))
            rdf_graph.add((edge_uri, self.schema.data_properties['inCircuit'],
                           Literal(edge.in_graph)))

            # Add edge type (Q/K/V or general flow)
            edge_type_uri = self._get_edge_type(edge)
            rdf_graph.add((edge_uri, self.schema.object_properties['hasEdgeType'], edge_type_uri))

            # Add rank if this edge is in the circuit
            if edge.in_graph:
                rank = self._calculate_edge_rank(edge, graph)
                if rank is not None:
                    rdf_graph.add((edge_uri, self.schema.data_properties['rank'],
                                   Literal(rank, datatype=XSD.integer)))

        return edge_uris

    def _get_edge_type(self, edge: Edge) -> URIRef:
        """Get edge type URI based on edge properties"""
        if edge.qkv:
            return self.schema.edge_types[edge.qkv.upper()]
        else:
            return self.schema.edge_types['Flow']

    def _calculate_edge_rank(self, edge: Edge, graph: Graph) -> Optional[int]:
        """Calculate rank of edge based on score among circuit edges"""
        try:
            # Get all edges in circuit and their scores
            circuit_edges = [(e.score, e) for e in graph.edges.values() if e.in_graph]

            # Sort by score (descending)
            circuit_edges.sort(key=lambda x: x[0], reverse=True)

            # Find rank of this edge
            for rank, (score, e) in enumerate(circuit_edges, 1):
                if e == edge:
                    return rank
            return None
        except:
            return None

    def _export_circuit_entity(self, rdf_graph: RDFGraph, graph: Graph,
                               run_uri: URIRef, experiment_id: str, step: int,
                               node_uris: Dict[str, URIRef],
                               edge_uris: Dict[str, URIRef]) -> URIRef:
        """Export the circuit as a unified entity"""
        circuit_uri = self.schema.namespace[f"circuit_{experiment_id}_{step}"]

        rdf_graph.add((circuit_uri, RDF.type, self.schema.classes['Circuit']))
        rdf_graph.add((circuit_uri, self.schema.object_properties['inRun'], run_uri))

        # Link circuit to its member nodes and edges
        for node_name, node in graph.nodes.items():
            if node.in_graph:
                rdf_graph.add((circuit_uri, self.schema.object_properties['hasMemberNode'],
                               node_uris[node_name]))

        for edge_name, edge in graph.edges.items():
            if edge.in_graph:
                rdf_graph.add((circuit_uri, self.schema.object_properties['hasMemberEdge'],
                               edge_uris[edge_name]))

        # Add circuit-level metrics
        num_edges = sum(1 for edge in graph.edges.values() if edge.in_graph)
        num_nodes = sum(1 for node in graph.nodes.values() if node.in_graph)

        rdf_graph.add((circuit_uri, self.schema.namespace.numEdges,
                       Literal(num_edges, datatype=XSD.integer)))
        rdf_graph.add((circuit_uri, self.schema.namespace.numNodes,
                       Literal(num_nodes, datatype=XSD.integer)))

        return circuit_uri

    def _add_config_metadata(self, rdf_graph: RDFGraph, run_uri: URIRef, config: Dict[str, Any]):
        """Add training configuration as metadata"""
        # Add key config parameters as data properties
        config_mappings = {
            'model_type': self.schema.data_properties['modelType'],
            'method': self.schema.data_properties['eapMethod'],
            'ig_steps': self.schema.namespace.igSteps,
            'top_k': self.schema.namespace.topK,
            'd_model': self.schema.namespace.dModel,
            'n_layers': self.schema.namespace.nLayers,
            'n_heads': self.schema.namespace.nHeads,
        }

        for config_key, property_uri in config_mappings.items():
            if config_key in config and config[config_key] is not None:
                value = config[config_key]
                if isinstance(value, (int, float)):
                    datatype = XSD.integer if isinstance(value, int) else XSD.double
                    rdf_graph.add((run_uri, property_uri, Literal(value, datatype=datatype)))
                else:
                    rdf_graph.add((run_uri, property_uri, Literal(str(value))))

    def _add_eap_results(self, rdf_graph: RDFGraph, run_uri: URIRef, eap_results: Dict[str, Any]):
        """Add EAP analysis results as metadata"""
        results_mappings = {
            'faithfulness': self.schema.data_properties['faithfulness'],
            'faithfulness_simple': self.schema.data_properties['faithfulnessSimple'],  # Add this property
            'baseline_performance': self.schema.data_properties['baselinePerformance'],
            'circuit_performance': self.schema.data_properties['circuitPerformance'],
            'ablated_performance': self.schema.namespace.ablatedPerformance,  # Add this property
            'num_edges_in_circuit': self.schema.namespace.numEdgesInCircuit,
            'num_nodes_in_circuit': self.schema.namespace.numNodesInCircuit,
            'performance_drop': self.schema.namespace.performanceDrop,  # Add this property
        }

        for result_key, property_uri in results_mappings.items():
            if result_key in eap_results and eap_results[result_key] is not None:
                value = eap_results[result_key]
                if isinstance(value, (int, float)):
                    datatype = XSD.integer if isinstance(value, int) else XSD.double
                    rdf_graph.add((run_uri, property_uri, Literal(value, datatype=datatype)))
                else:
                    rdf_graph.add((run_uri, property_uri, Literal(str(value))))

    def _save_rdf(self, rdf_graph: RDFGraph, experiment_id: str, step: int) -> Path:
        """Save RDF graph to TTL file"""
        exp_dir = self.output_dir / experiment_id / "circuits"
        exp_dir.mkdir(parents=True, exist_ok=True)

        output_path = exp_dir / f"step_{step}.ttl"

        # Add some basic prefixes to make TTL more readable
        rdf_graph.serialize(destination=str(output_path), format='turtle')

        # Log export info
        print(f"Exported circuit to OWL: {output_path}")
        print(f"  Triples: {len(rdf_graph)}")
        print(
            f"  Nodes: {len([s for s, p, o in rdf_graph if (s, RDF.type, self.schema.classes['Node']) in rdf_graph])}")
        print(
            f"  Edges: {len([s for s, p, o in rdf_graph if (s, RDF.type, self.schema.classes['CircuitEdge']) in rdf_graph])}")

        return output_path

    def export_temporal_evolution(self, graphs: Dict[int, Graph],
                                  experiment_id: str) -> str:
        """Export temporal evolution of circuits across steps"""
        rdf_graph = RDFGraph()
        for prefix, namespace in self.namespaces.items():
            rdf_graph.bind(prefix, namespace)

        # Create temporal circuit entity
        temporal_circuit_uri = self.temporal.namespace[f"temporal_circuit_{experiment_id}"]
        rdf_graph.add((temporal_circuit_uri, RDF.type, self.temporal.temporal_classes['TemporalCircuit']))

        # Link snapshots and track evolution
        sorted_steps = sorted(graphs.keys())
        prev_step = None

        for step in sorted_steps:
            # Link to snapshot
            circuit_uri = self.schema.namespace[f"circuit_{experiment_id}_{step}"]
            rdf_graph.add((temporal_circuit_uri, self.temporal.temporal_properties['hasSnapshot'],
                           circuit_uri))
            rdf_graph.add((circuit_uri, self.temporal.temporal_properties['snapshotStep'],
                           Literal(step, datatype=XSD.integer)))

            # Track evolution between steps
            if prev_step is not None:
                prev_circuit_uri = self.schema.namespace[f"circuit_{experiment_id}_{prev_step}"]
                rdf_graph.add((prev_circuit_uri, self.temporal.temporal_properties['evolvesTo'],
                               circuit_uri))
                rdf_graph.add((circuit_uri, self.temporal.temporal_properties['evolvesFrom'],
                               prev_circuit_uri))

            prev_step = step

        # Save evolution file
        exp_dir = self.output_dir / experiment_id / "circuits"
        exp_dir.mkdir(parents=True, exist_ok=True)
        evolution_path = exp_dir / "evolution.ttl"
        rdf_graph.serialize(destination=str(evolution_path), format='turtle')

        return str(evolution_path)


# Convenience factory function
def create_owl_exporter(output_dir: str = "./data/experiments") -> OWLExporter:
    """Create an OWL exporter with standard configuration"""
    return OWLExporter(output_dir=output_dir)