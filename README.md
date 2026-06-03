# ASL Sovereign: The Decoder Ring

Ya know how systems and structures say "just be yourself?" And that does not really work out so well in my experience, possibly in yours as well.

Upfront, I do not sign. This happened because an ASL dataset was/is being used to teach something it should not be and I think that is not ok. In summary.

This is the result of that. This dashboard will get better by you being you. You can explore how in the library. There is no reason to take my word for it.

Talk to it like you would a friend, casually, unafraid of being judged. Why? Because your actual experience, in your real words, teaches this chatbot model so that the next person who uses words similar to yours can be recognized. I hope that makes sense. This does not work if you adjust to it — it needs to adjust to you.

If you sign ASL and a recognition tool has ever gotten you wrong — this is for you.

Upload a confusion matrix if you have one, or a CSV, or nothing at all. If you are training vision models and want to know which model is optimal for your use case, this tool works for that too.

A transparent pipeline on training data is better for everyone. The right tool, in the right application, is cheaper, less error prone, causes less harm and ultimately we can stop going to the hardware store for bread.

The model on the other end was trained to meet you where you are, not where a dataset decided you should be.

That inversion is the whole point.

---

## The Problem

ASL signers have been doing the adapting for a long time. Tools get built, benchmarks get published with accuracy rates in the high 90s, and the numbers are real — for the lab conditions, the curated datasets, and the hands that happened to be in the room when the data was collected. Real signers in real conditions are a different story.

The failure is almost never the architecture. It's what the model was trained on. A model trained on 21-point hand skeleton data cannot learn body-anchored spatial signs no matter how long you fine-tune it. The data didn't include what it needed to learn. That gets baked in, and it shows up as the same letter pairs failing over and over — K going to V, D going to I, M and N and T collapsing into each other.

Those patterns are a fingerprint. They tell you exactly what regime the model was trained on. This project reads that fingerprint.

---

## The Decoder Ring

You describe what's going wrong. The model diagnoses it.

Plain language works. "K always comes back as V." "It can't tell M from N." "The whole middle cluster is wrong." That's enough. The backend — a fine-tuned Gemma 2 Instruct model — was trained specifically to recognize the failure patterns that correspond to different training data regimes and return a structured diagnosis.

You can also upload a CSV of your results, a confusion matrix image, or a dataset. Whatever you have. Or nothing — just talk.

The ongoing conversation between users and this tool is itself training data. Quality over quantity. Every real complaint from a real signer is more valuable than a synthetic example. The model gets better at meeting you as more of you show up.

---

## The Finding (for the technical readers)

Confusion matrix structure is a stable diagnostic fingerprint of a model's training data regime. Specific letter-pair confusions (K→V, D→I, M/N/T cluster) appear at nearly identical rates across two independent datasets — Roboflow ASL (1,185 images, diverse real-world signers) and an Indic Sign Language dataset (ISL, ~14,000 images, different sign language entirely) — with a Mantel cross-dataset correlation of **r = 0.945, p < 0.001**.

This cross-linguistic stability means the confusion structure is a geometric prior in the training data, not a model artifact. It can be read from a confusion matrix alone. That's the `decoder_ring.py` module. The chatbot is the interface.

---

## The Chatbot Model

The diagnostic chatbot runs **Gemma 2 9B Instruct** (GGUF, Q4_K_M quantization), fine-tuned on ASL training data failure modes using Unsloth/Alpaca format. It is not Gemma 4. Do not confuse the two.

- **Research/audit model:** Gemma 4 E4B (via Ollama) — used for the original confusion matrix analysis
- **Chatbot/diagnostic model:** Gemma 2 9B Instruct (fine-tuned) — what you're talking to

The distinction matters. Gemma 4 E4B was the subject of study. Gemma 2 Instruct was selected specifically because it was trained to respond naturally to the way people actually describe problems — not to produce structured research output.

---

## The Data

There is no large labeled dataset of "models with known training regimes" in the literature. That set doesn't exist yet. So the absolute accuracy of this diagnostic instrument is accumulating in real time, through real use.

The construct validity is established: bootstrap persistence at 1,000 iterations, Gaussian noise probe confirming weight-encoded confusion structure, Mantel r=0.945 cross-dataset. The instrument is measuring something real and stable. The number for how often it gets the diagnosis right will be built honestly, from actual submissions — not made up.

---

## Stack

- **Frontend:** React + Tailwind, retro comic/newsprint aesthetic
- **Backend:** Modal serverless GPU (L4), FastAPI + llama-cpp-python
- **Model:** `gemma-2-9b-it.Q4_K_M.gguf` on Modal Volume `asl-model-volume`
- **Data logging:** Google Sheets (`diagnoses` worksheet) — all submissions logged for model improvement
- **Endpoint:** `https://phinnphace--asl-decoder-cloud-fastapi-app.modal.run`

---

## Repository Structure

```
decoder_ring.py                    ← core diagnostic module
provenance_decoder.py              ← LLM-agent companion (Gemma 2 via llama-cpp)
chatbot/                           ← React frontend + Modal backend
  src/Home.jsx                     ← main UI
  src/Library.jsx                  ← Ted's Library (technical docs)
  modal_api.py                     ← Modal serverless GPU deployment

01_Dataset_Audits/                 ← baseline evaluations, pre-processing audits
02_Vision_and_FSboard/             ← Gemma 4 multimodal tests, FSboard benchmarks
03_Validation_and_Ablation/        ← Spearman/Bootstrap, Ollama vs Gemma-4eb ablation

DATA_ARCHIVE.md                    ← full data provenance map
```

---

## On the data plan

User submissions go to a Google Sheet. Nothing is manipulated. A real complaint from someone who signs is better data than a hundred synthetic examples. The world will catch up — this is not waiting for it.

---

*Certified by Ted, stoop tabby.*
