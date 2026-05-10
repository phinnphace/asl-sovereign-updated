# ASL Sovereign Agent Rules
## Target Repository Structure (asl_sovereign)
- 01_Dataset_Audits/: Baseline evaluations (e.g., SignMNIST), pre-processing audits.
- 02_Vision_and_FSboard/: Gemma-4 multimodal tests, FSboard benchmarks, matrix outputs.
- 03_Validation_and_Ablation/: Spearman/Bootstrap tests, Ollama vs Gemma-4eb ablation logs.

## Delegation Instructions
- Reference the 'asl-gemma4' knowledge base to identify "final" versions of scripts.
- Prioritize .py scripts, .csv pipelines, and .ipynb notebooks.
- Checkpoint: Verify file existence against this structure before staging git commits.