# DATA_ARCHIVE.md — ASL Sovereign Decoder Ring

All datasets listed here are **excluded from Git** via `.gitignore` (`.csv`, `.parquet`, `.zip` patterns)
to maintain repo performance. They are required for full pipeline reproducibility.

---

## 1. Kaggle — ASL Fingerspelling Competition (2023)

**Source:** https://www.kaggle.com/competitions/asl-fingerspelling  
**License:** Competition rules (non-commercial research)

| File | Size | Description |
|---|---|---|
| `111123288.parquet.zip` | ~1.19 GB | Compressed landmark archive |
| `111123288.parquet/111123288.parquet` | — | MediaPipe holistic landmark sequences (1,629 spatial coords × 543 landmarks, float32) |
| `supplemental_metadata.csv` | ~5 MB | Participant/sequence index for the supplemental fingerspelling sentences |
| `character_to_prediction_index.json` | — | 59-character label map (letters + punctuation/digits) |

**Purpose:** Landmark-sequence baseline. The shape-vector format (not raw images) was the
entry point for understanding what MediaPipe extracts vs. what image models actually see.
Used to establish the 21-landmark vs. full-body training regime comparison.

---

## 2. SignMNIST — Static ASL Baseline

**Source:** https://www.kaggle.com/datasets/datamunge/sign-language-mnist  
**License:** CC0 Public Domain

| File | Size | Description |
|---|---|---|
| `sign_mnist_train.csv` | ~83 MB | 27,455 training samples, 28×28 grayscale pixel rows |
| `sign_mnist_test.csv` | ~22 MB | 7,172 test samples, same format |

**Purpose:** Academic benchmark baseline. 26 classes, J and Z excluded (movement gestures).
Used in `01_Dataset_Audits/gwo_claim_audit.py` to stress-test the GWO-CTransNet paper's
claimed 98.07% accuracy against a zero-shot Gemma-4 pipeline (result: 2.40%, gap of 95.67pp).
Demonstrates that 28×28 grayscale discards the spatial anchoring that real-world models require.

---

## 3. Roboflow — Real-World Diverse Signers

**Source:** https://universe.roboflow.com (workspace: `phinns-workspace`, exported April 7, 2026)  
**License:** Roboflow standard export terms

| File / Directory | Description |
|---|---|
| `raw_data/train/[A-Y]/` | 1,185 images, folder-per-letter format, no preprocessing or augmentation |
| `asl_roboflow_simple_prompt.csv` | Gemma-4 inference results, simple prompt |
| `asl_prompt_ab_test.csv` | Prompt variant A/B comparison output (letter A, n=10 per variant) |
| `roboflow_predictions.csv` | Full prediction log |
| `roboflow_confusion_matrix (1).csv` | Confusion matrix — primary decoder ring input |
| `landmark_distances_roboflow.csv` | MediaPipe pairwise landmark distances, 30 samples/letter |

**Purpose:** Real-world image set with diverse signers. No preprocessing means landmark
geometry is preserved. Primary dataset for confusion matrix analysis and decoder ring
calibration. Mantel r=0.945 cross-dataset correlation with ISL confirms structural stability.

---

## 4. FSboard — Multimodal Fingerprint Benchmark

**Source:** Georg et al. (2024), arXiv:2407.15806  
**License:** CC BY 4.0  
**DOI:** https://arxiv.org/abs/2407.15806

| Asset | Description |
|---|---|
| External benchmark | Fingerspelling board dataset used for multimodal confusion signature extraction |

**Purpose:** External reference topology for the decoder ring. FSboard confusion structure
provides the "multimodal pole" of the training regime spectrum (landmark-only ↔ hybrid).
Cited in `decoder_ring.py` knowledge base; Mantel test validates cross-dataset pair stability.
Not stored locally — access via arXiv link above.

---

## 5. ISL — Indic Sign Language Handshape Dataset ⭐ PRIMARY CROSS-LINGUISTIC VALIDATION SET

**Source:** Biswas, Sougatamoy (2024), "ISL Handshape Dataset"  
**Roboflow export tag in repo:** `ISL custom Data/`

| File / Directory | Description |
|---|---|
| `ISL custom Data/ISl_firsthalf/[A–P]/` | Letters A–P, ~550 images/letter |
| `ISL custom Data/[Q–Z]/` | Letters Q–Z, ~550 images/letter |
| `curated_standard/` | Hand-curated reference images (A, B, C) for clean baseline |
| `asl_isl_baseline_full.csv` | Full ASL vs. ISL parallel inference output |
| `isl_confusion_matrix (1).csv` | ISL confusion matrix — decoder ring cross-linguistic input |
| `landmark_distances.csv` | MediaPipe pairwise landmark distances for ISL letters |

**Total:** ~14,000 images across 26 letters (~550/letter), folder-per-letter format.  
No preprocessing. Collected from ISL signers — distinct phonological inventory from ASL.

**Why this dataset is the headline finding:**  
The decoder ring pairs (K→V, D→I, etc.) fire at nearly identical rates on ISL and Roboflow ASL
(Mantel r=0.945, p<0.001), despite different signers, different sign language, and different
visual style. This cross-linguistic stability is the core claim of the sovereign project:
confusion structure is a geometric prior in the training data, not a model artifact.
ISL is the independent replication that makes the finding publishable.

---

## Derived Pipeline Outputs (also gitignored)

These CSVs are generated by the methodology scripts and excluded from git. They can be
fully reproduced by running the pipeline in order.

| File | Generated by |
|---|---|
| `asl_hallucination_baseline.csv` | Early hallucination baseline run |
| `gaussian_noise_results.csv` | `02_Vision_and_FSboard/gaussianprobe.py` |
| `gray_baseline_results.csv` | `02_Vision_and_FSboard/gray-baselinetest.py` |
| `gwo_audit_results.csv` | `01_Dataset_Audits/gwo_claim_audit.py` |

---

## Reproducibility Note

To reconstruct the full dataset locally:

1. **Kaggle:** Download via `kaggle competitions download -c asl-fingerspelling`
2. **SignMNIST:** Download via `kaggle datasets download -d datamunge/sign-language-mnist`
3. **Roboflow:** Re-export from `phinns-workspace` or contact project owner
4. **ISL:** Cite Biswas (2024) and request dataset; or re-export from Roboflow ISL project
5. **FSboard:** Access via arXiv:2407.15806 supplemental materials
