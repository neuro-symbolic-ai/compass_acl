from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL, PROV
from typing import Dict, Set, Optional

# Base namespace
MECHINT_BASE = "http://purl.org/mechint"
MECHINT = Namespace(f"{MECHINT_BASE}#")


class MechIntSchema:
    """Central schema definition for MechInt ontology"""

    def __init__(self):
        self.namespace = MECHINT
        self.base_uri = MECHINT_BASE

    # Core Classes
    @property
    def classes(self) -> Dict[str, URIRef]:
        return {
            # Core entities
            'Experiment': self.namespace.Experiment,
            'Model': self.namespace.Model,
            'Checkpoint': self.namespace.Checkpoint,
            'Run': self.namespace.Run,
            'Sequence': self.namespace.Sequence,
            'Token': self.namespace.Token,

            # Semantic concepts
            'SemanticRule': self.namespace.SemanticRule,
            'SemanticRole': self.namespace.SemanticRole,

            # Circuit structures
            'Circuit': self.namespace.Circuit,
            'CircuitEdge': self.namespace.CircuitEdge,
            'CircuitNode': self.namespace.CircuitNode,
            'Node': self.namespace.Node,
            'Edge': self.namespace.Edge,

            # Components (transformer parts)
            'Component': self.namespace.Component,
            'AttnHead': self.namespace.AttnHead,
            'MLP': self.namespace.MLP,
            'LayerNorm': self.namespace.LayerNorm,
            'Residual': self.namespace.Residual,
            'Input': self.namespace.Input,
            'Embed': self.namespace.Embed,
            'PosEmbed': self.namespace.PosEmbed,
            'OutputLogits': self.namespace.OutputLogits,

            # Graph primitives
            'Subpart': self.namespace.Subpart,

            # Interventions
            'Intervention': self.namespace.Intervention,
            'Measure': self.namespace.Measure,

            # Enums
            'EdgeType': self.namespace.EdgeType,
            'InterventionType': self.namespace.InterventionType,

            # Defined classes
            'HighImpactEdge': self.namespace.HighImpactEdge,
            'SupportedEdge': self.namespace.SupportedEdge,
            'WellSupportedCircuit': self.namespace.WellSupportedCircuit,
        }

    # Object Properties
    @property
    def object_properties(self) -> Dict[str, URIRef]:
        return {
            # Model/experiment relationships
            'ofModel': self.namespace.ofModel,
            'hasCheckpoint': self.namespace.hasCheckpoint,
            'ofExperiment': self.namespace.ofExperiment,
            'inRun': self.namespace.inRun,

            # Component relationships
            'aboutComponent': self.namespace.aboutComponent,
            'hasSubpart': self.namespace.hasSubpart,
            'subpartOfComponent': self.namespace.subpartOfComponent,

            # Graph structure
            'hasSource': self.namespace.hasSource,
            'hasTarget': self.namespace.hasTarget,
            'hasEdgeType': self.namespace.hasEdgeType,

            # Semantic relationships
            'hasRole': self.namespace.hasRole,
            'circuitRole': self.namespace.circuitRole,
            'implementsRule': self.namespace.implementsRule,

            # Circuit membership
            'hasMemberEdge': self.namespace.hasMemberEdge,
            'hasMemberNode': self.namespace.hasMemberNode,

            # Interventions
            'appliesToNode': self.namespace.appliesToNode,
            'appliesToEdge': self.namespace.appliesToEdge,
            'patchSource': self.namespace.patchSource,
            'hasMeasure': self.namespace.hasMeasure,

            # Sequences
            'hasSequence': self.namespace.hasSequence,
            'hasToken': self.namespace.hasToken,
        }

    # Data Properties
    @property
    def data_properties(self) -> Dict[str, URIRef]:
        return {
            # Addressing and identification
            'address': self.namespace.address,
            'layer': self.namespace.layer,
            'head': self.namespace.head,

            # Graph state
            'inGraph': self.namespace.inGraph,
            'inCircuit': self.namespace.inCircuit,
            'score': self.namespace.score,
            'rank': self.namespace.rank,

            # Measurements
            'name': self.namespace.name,
            'value': self.namespace.value,
            'unit': self.namespace.unit,
            'scope': self.namespace.scope,

            # Versioning
            'step': self.namespace.step,
            'commit': self.namespace.commit,

            # Temporal and metadata
            'timestamp': self.namespace.timestamp,
            'modelType': self.namespace.modelType,
            'eapMethod': self.namespace.eapMethod,

            # Performance metrics
            'faithfulness': self.namespace.faithfulness,
            'faithfulnessSimple': self.namespace.faithfulnessSimple,
            'baselinePerformance': self.namespace.baselinePerformance,
            'circuitPerformance': self.namespace.circuitPerformance,
            'ablatedPerformance': self.namespace.ablatedPerformance,
            'performanceDrop': self.namespace.performanceDrop,
            'performanceDropPct': self.namespace.performanceDropPct,

            # Circuit statistics
            'numEdges': self.namespace.numEdges,
            'numNodes': self.namespace.numNodes,
            'numEdgesInCircuit': self.namespace.numEdgesInCircuit,
            'numNodesInCircuit': self.namespace.numNodesInCircuit,

            # Configuration parameters
            'igSteps': self.namespace.igSteps,
            'topK': self.namespace.topK,
            'dModel': self.namespace.dModel,
            'nLayers': self.namespace.nLayers,
            'nHeads': self.namespace.nHeads,
            'dFF': self.namespace.dFF,
            'interventionType': self.namespace.interventionType,
        }

    # Enumerated individuals
    @property
    def edge_types(self) -> Dict[str, URIRef]:
        return {
            'Q': self.namespace.Q,
            'K': self.namespace.K,
            'V': self.namespace.V,
            'OV': self.namespace.OV,
            'Flow': self.namespace.Flow,
            'ResidualType': self.namespace.ResidualType,
            'Probe': self.namespace.Probe,
            'InterventionEdgeType': self.namespace.InterventionEdgeType,
        }

    @property
    def intervention_types(self) -> Dict[str, URIRef]:
        return {
            'patch': self.namespace.patch,
            'ablate': self.namespace.ablate,
            'zero': self.namespace.zero,
            'noise': self.namespace.noise,
            'swap': self.namespace.swap,
            'resample': self.namespace.resample,
        }

    # Component type mappings
    @property
    def component_types(self) -> Set[URIRef]:
        return {
            self.namespace.AttnHead,
            self.namespace.MLP,
            self.namespace.LayerNorm,
            self.namespace.Residual,
            self.namespace.Input,
            self.namespace.Embed,
            self.namespace.PosEmbed,
            self.namespace.OutputLogits,
        }

    def get_component_type_for_node(self, node_name: str) -> URIRef:
        """Map node name to appropriate component type"""
        if node_name == 'input':
            return self.namespace.Input
        elif node_name == 'logits':
            return self.namespace.OutputLogits
        elif node_name.startswith('a'):  # attention head pattern: a{layer}.h{head}
            return self.namespace.AttnHead
        elif node_name.startswith('m'):  # MLP pattern: m{layer}
            return self.namespace.MLP
        else:
            return self.namespace.Component

    def parse_node_structure(self, node_name: str) -> Dict[str, Optional[int]]:
        """Extract layer/head information from node names"""
        result = {}

        if node_name.startswith('a'):  # a0.h1 -> layer=0, head=1
            parts = node_name.split('.')
            if len(parts) == 2:
                layer_part = parts[0][1:]  # remove 'a'
                head_part = parts[1][1:]  # remove 'h'
                try:
                    result['layer'] = int(layer_part)
                    result['head'] = int(head_part)
                except ValueError:
                    pass
        elif node_name.startswith('m'):  # m5 -> layer=5
            layer_part = node_name[1:]  # remove 'm'
            try:
                result['layer'] = int(layer_part)
            except ValueError:
                pass

        return result


# Temporal Extensions for Circuit Evolution
class TemporalSchema:
    def __init__(self, base_schema: MechIntSchema):
        self.base = base_schema
        self.namespace = base_schema.namespace

    @property
    def temporal_classes(self) -> Dict[str, URIRef]:
        return {
            'TemporalCircuit': self.namespace.TemporalCircuit,
            'CircuitEvolution': self.namespace.CircuitEvolution,
            'CircuitChange': self.namespace.CircuitChange,
            'VersionedNode': self.namespace.VersionedNode,
            'VersionedEdge': self.namespace.VersionedEdge,
        }

    @property
    def temporal_properties(self) -> Dict[str, URIRef]:
        return {
            # Evolution tracking
            'evolvesFrom': self.namespace.evolvesFrom,
            'evolvesTo': self.namespace.evolvesTo,
            'hasSnapshot': self.namespace.hasSnapshot,
            'snapshotStep': self.namespace.snapshotStep,

            # Change events
            'changedNode': self.namespace.changedNode,
            'changedEdge': self.namespace.changedEdge,
            'changeType': self.namespace.changeType,
            'oldScore': self.namespace.oldScore,
            'newScore': self.namespace.newScore,

            # Versioning
            'version': self.namespace.version,
            'previousVersion': self.namespace.previousVersion,
            'nextVersion': self.namespace.nextVersion,
        }


# Pattern Recognition Schema
class PatternSchema:
    def __init__(self, base_schema: MechIntSchema):
        self.base = base_schema
        self.namespace = base_schema.namespace

    @property
    def pattern_classes(self) -> Dict[str, URIRef]:
        return {
            'CircuitPattern': self.namespace.CircuitPattern,
            'CircuitMotif': self.namespace.CircuitMotif,
            'SemanticPattern': self.namespace.SemanticPattern,
            'LearnedRule': self.namespace.LearnedRule,
            'PatternInstance': self.namespace.PatternInstance,
        }

    @property
    def pattern_properties(self) -> Dict[str, URIRef]:
        return {
            'hasPattern': self.namespace.hasPattern,
            'instanceOf': self.namespace.instanceOf,
            'hasFrequency': self.namespace.hasFrequency,
            'hasConfidence': self.namespace.hasConfidence,
            'coOccursWith': self.namespace.coOccursWith,
            'implies': self.namespace.implies,
            'contradicts': self.namespace.contradicts,
        }


# Convenience instance
MECHINT_SCHEMA = MechIntSchema()
TEMPORAL_SCHEMA = TemporalSchema(MECHINT_SCHEMA)
PATTERN_SCHEMA = PatternSchema(MECHINT_SCHEMA)

# Export commonly used elements
__all__ = [
    'MECHINT', 'MECHINT_BASE', 'MECHINT_SCHEMA',
    'TEMPORAL_SCHEMA', 'PATTERN_SCHEMA',
    'MechIntSchema', 'TemporalSchema', 'PatternSchema'
]