# Research Archive: Methodology & Validation

This directory contains the raw data pipelines (.csv), Python scripts (.py), and baseline notebooks that underpin the ASL Sovereign Decoder Ring. The research evolved from a baseline academic evaluation of standard datasets into a rigorous, multi-phased diagnostic audit.

### Directory Structure

* **`01_Dataset_Audits/`**
  * *Focus:* The origin of the project. Contains baseline evaluations of standard ASL datasets (e.g., SignMNIST). Documents the finding that standard ML "pre-processing" often erases critical spatial/body-anchored syntax, creating models that fail in real-world application.
* **`02_Vision_and_FSboard/`**
  * *Focus:* The discovery phase. Contains the initial tests getting local Gemma-4 to process multimodal inputs, alongside the synthesis of external benchmarks (FSboard) with localized matrix outputs to isolate the specific "fingerprints" of 21-landmark vs. full-body training regimes.
* **`03_Validation_and_Ablation/`**
  * *Focus:* The stress test. Contains the statistical validation protocols (Spearman rank-order, Bootstrap stability tests) proving that failure clusters are stable signatures. This directory also houses the ablation testing logs, documenting the exhaustive attempts to isolate Ollama's runtime effects from Gemma-4eb's native weights, establishing the current limits of model customizability.