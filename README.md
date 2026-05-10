# ASL Sovereign: The Decoder Ring

*A diagnostic instrument for auditing vision model training data provenance from confusion matrix structure alone — without access to model weights, training data, or documentation.*

---

## The Problem: "Don't go to the hardware store for bread."

Computer vision models fail at American Sign Language (ASL) not because the architectures are wrong, but because they are misapplied. A model trained exclusively on 21-hand landmarks cannot learn body-anchored spatial signs no matter how much it is fine-tuned. Standard benchmarks obscure this — published accuracy rates in the high 90s are measured on datasets curated to the point of sanitization, under lab conditions, on hands that do not reflect the full diversity of people who sign.

The actual question — can a machine recognize ASL fingerspelling from real hands, in real conditions, across diverse signers — had not been asked with rigor. This project asks it.

---

## The Finding

Confusion matrix structure is a stable diagnostic fingerprint of a model's training data regime. Specific letter-pair confusions (K→V, D→I, and others) appear at nearly identical rates across two independent datasets — Roboflow ASL (1,185 images, diverse real-world signers) and an Indic Sign Language dataset (ISL, ~14,000 images, different sign language entirely) — with a Mantel cross-dataset correlation of **r = 0.945, p < 0.001**.

This cross-linguistic stability means the confusion structure is a geometric prior in the training data, not a model artifact. It can be read from a confusion matrix alone.

---

## The Decoder Ring

`decoder_ring.py` — the core module. Takes either a confusion matrix or a per-letter accuracy summary and returns a structured diagnostic:

```python
from decoder_ring import decode

result = decode(
    per_letter_accuracy={'K': 0.0, 'D': 0.05, 'C': 0.95, ...},
    top_errors={'K': [('V', 41)], 'D': [('I', 43)]},
    dataset_name='MyDataset'
)
print(result['report'])
# spectrum_position: landmark-only
# deployment_risk:   High — model will fail on body-anchored and two-hand signs
```

Output keys: `spectrum_position`, `spectrum_score`, `firing_pairs`, `families`, `predicted_additional_failures`, `failure_mode`, `training_data_implication`, `deployment_risk`, `report`.

`provenance_decoder.py` — an LLM-agent companion. Takes unstructured failure complaints in plain language and routes them to a diagnosis via local Gemma-4 through Ollama.

---

## Repository Structure

```
decoder_ring.py                          ← core diagnostic module
contractdecoder.py                       ← output schema reference
provenance_decoder.py                    ← LLM-agent precursor (Ollama/Gemma-4)
streamlitasl.py                          ← Streamlit front end (in progress)
DATA_ARCHIVE.md                          ← full data provenance map
comeedyofgemmaerrors.md                  ← narrative: the full technical arc

01_Dataset_Audits/
  asl_mnist_v1.ipynb                     ← KNN/SignMNIST origin (course assignment)
  asl_mnist_final.ipynb                  ← final model training notebook
  copy_of_notebook2_asl_audit_v3_(1).py ← final Colab audit notebook (as exported)
  gwo_claim_audit.py                     ← audits GWO-CTransNet 98.07% claim
  asl_prompt_ab_test.py                  ← prompt variant A/B comparison
  curated_test.py / curated_test (2).py  ← curated standard baseline tests
  crop_grid.py                           ← builds curated_standard/ from ASL poster
  make_manifest.py                       ← generates XAI fine-tuning manifest

02_Vision_and_FSboard/
  import_torch.py                        ← HuggingFace direct load attempt (failed)
  ted_test.py                            ← proof-of-concept: model could not see Ted
  gaussianprobe.py                       ← Gaussian noise probe at incremental sigma
  gaussianprobe_slurm.py                 ← OSU HPC / SLURM variant
  gray-baselinetest.py                   ← zero-input gray baseline (prior test)
  landmark_analysis.py                   ← MediaPipe pairwise distance extraction
  [+ supporting diagnostics and exploration scripts]

03_Validation_and_Ablation/
  decoder_ring.py (root)                 ← see above
  test_decoder_ring.py                   ← 25-case test suite
  validation_protocol_v1.py             ← Spearman/Bootstrap chain, starting point
  final_validation_protocol.py          ← Spearman/Bootstrap/Graph chain, final
  confusion_matrices_normalized.py       ← threshold sensitivity, normalized heatmaps
  diagnostic.py / diagnostic_v2.py      ← HF processor ablation (Ollama vs. native)
  Copy_of_notebook2_asl_audit_v3_(1) (2).ipynb  ← final audit notebook (May 9)
  validation_outputs_final.ipynb         ← final validation outputs (May 9)
```

---

## The Technical Arc

Gemma-4 E4B was released days before this project began. The model card was sparse. Every standard approach was tried: HuggingFace transformers with 4-bit NF4 quantization, explicit `Gemma4Processor`, CUDA layer routing, OSU supercomputer via SLURM. All failed silently — valid `pixel_values`, wrong outputs, every image described as "gray static noise."

The diagnostic image was a neighbor's cat named Ted. Ted confirmed the failure was total: the vision encoder was being silently zeroed by bitsandbytes runtime quantization, because the layer names passed to `llm_int8_skip_modules` did not map to Gemma-4's internal architecture.

Ollama had been running on the machine the entire time. GGUF pre-baked quantization preserves the vision encoder. One test: Ted was described accurately. The entire HuggingFace pipeline was abandoned.

The full account is in `comeedyofgemmaerrors.md`.

---

## Data

All datasets are excluded from git (`.gitignore`). See `DATA_ARCHIVE.md` for full provenance, filenames, and reproducibility instructions.

**Primary datasets:**
- **Roboflow ASL** — 1,185 images, real-world diverse signers, no preprocessing
- **ISL (Biswas 2024)** ⭐ — ~14,000 images, Indic Sign Language, primary cross-linguistic validation set
- **SignMNIST** — 28×28 grayscale baseline, used for benchmark auditing
- **Kaggle ASL Fingerspelling** — MediaPipe landmark sequences

---

## Hardware

Developed and audited locally on an RTX 3070 (8GB VRAM). Inference via Ollama (`gemma4:e4b`, 9.6GB GGUF). No cloud compute required for the decoder ring itself.

---

## Status

The decoder ring is complete and validated. Streamlit front end is in progress. Narrative write-up (`comeedyofgemmaerrors.md`) is ongoing.
