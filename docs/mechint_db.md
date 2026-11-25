# MechInt-DB

MechInt-DB stores circuits found with EAP as OWL/RDF knowledge graphs, so they can be queried and compared across training steps.

- **OWL/RDF export:** circuits are saved as Turtle (`.ttl`) files
- **MechInt ontology:** transformer components, circuit edges, semantic roles and interventions
- **Temporal evolution:** circuit changes across training steps
- **EAP integration:** runs EAP analysis and exports the result in one step
- **Metrics:** faithfulness, ablation results and circuit statistics stored with each circuit

## Basic usage

```python
from mechint_db.integration import MechIntEAPConfig, MechIntEAPAnalyzer

config = MechIntEAPConfig(
    method="EAP-IG-inputs",
    top_k=28,
    ig_steps=5,
    cnp_dataset_path="./data/minimal_pairs.json",
    save_owl=True,
    owl_output_dir="./circuit_ontologies",
    experiment_description="Semantic role circuit analysis",
    researcher_name="Your Name",
)

analyzer = MechIntEAPAnalyzer(config)

results, graph = analyzer.analyze(
    model=hooked_model,
    batch_loader=data_loader,
    tokenizer=tokenizer,
    model_name="my_transformer",
    step=1000,
    experiment_id="semantic_roles_exp1",
)

print(f"Faithfulness: {results['faithfulness']:.4f}")
print(f"Circuit edges: {results['num_edges_in_circuit']}")
```

The minimal-pair file is a JSON or CSV table with the columns `clean`, `corrupted`, `correct_idx` and `incorrect_idx`.

## One-step analysis and export

```python
from mechint_db.integration import analyze_with_owl_export

results, graph, owl_path = analyze_with_owl_export(
    model=hooked_model,
    cnp_dataset_path="./data/minimal_pairs.json",
    experiment_id="semantic_roles_exp1",
    step=1000,
    method="EAP-IG-inputs",
    top_k=28,
    tokenizer=tokenizer,
)
print(f"Circuit saved to: {owl_path}")
```

## Temporal evolution

```python
analyzer.export_temporal_evolution(
    experiment_results=all_step_results,   # {step: (results, graph)}
    experiment_id="semantic_roles_exp1",
)
# Writes evolution.ttl with snapshots and changes
```

## Ontology

The MechInt ontology models:

- **Transformer components:** `AttnHead`, `MLP`, `LayerNorm`, `Residual`, `Embed`
- **Circuit structure:** `Circuit`, `CircuitEdge` and `CircuitNode`, with scores, ranks and edge types (Q/K/V/OV/Flow)
- **Semantic concepts:** `SemanticRole` and `SemanticRule`, linking circuits to linguistic function
- **Temporal evolution:** `CircuitEvolution` and `CircuitChange`
- **Experiments:** `Experiment`, `Run` and `Checkpoint`

Output files are organised by experiment:

```text
circuit_ontologies/
└── semantic_roles_exp1/
    └── circuits/
        ├── step_0.ttl
        ├── step_500.ttl
        ├── step_1000.ttl
        └── evolution.ttl
```
