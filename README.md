# ASL Sovereign: The Provenance Decoder Ring 🔍

*A diagnostic routing agent utilizing local Gemma-4 to analyze failure modes in ASL recognition, mapping structural errors to opaque training data to prevent wasted compute.*

## The Problem: "Don't go to the hardware store for bread."
Computer vision models fail at American Sign Language (ASL) not necessarily because the architectures are flawed, but because they are misapplied due to opaque training data provenance. Applying standard fine-tuning to a model trained exclusively on 21-hand landmarks to fix body-anchored spatial errors is mathematically impossible and wastes massive compute resources. 

## The Solution
This tool uses a locally hosted **Gemma 4** model via **Ollama** to act as a diagnostic agent. By evaluating unstructured user complaints or confusion matrix fingerprints, Gemma maps the specific failure clusters to the original training data structure. This framework routes the task to the correct model variant rather than scapegoating the technology.

## Local Execution & Hardware Profile
This framework prioritizes epistemic agency and data privacy. It was designed to run entirely offline on consumer hardware (developed and audited locally on an 8GB VRAM RTX 3070).

**Requirements:**
* Python 3.10+
* Local installation of [Ollama](https://ollama.com/) running a Gemma variant. 

**Run Instructions:**
```bash
git clone [https://github.com/phinnphace/asl-sovereign.git](https://github.com/phinnphace/asl-sovereign.git)
cd asl-sovereign
pip install -r requirements.txt
ollama run gemma
streamlit run app.py
