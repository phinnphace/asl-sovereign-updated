## Section 1: Getting the Pipeline Working: Sharing the epic fails in details to save you the trouble 

**Gemma 4 was brand new when this project began — released only days prior and the model card, well,  truly a "card".
 https://ai.google.dev/gemma/docs/core/model_card_4 

The model card described a multimodal vision-language model available in several sizes, with the E4B (4 billion parameter, instruction-tuned) variant as the practical choice for 8GB VRAM. The 26B A4B MoE variant was briefly considered and immediately ruled out — it would not fit even with aggressive quantization. E2B was considered and set aside as less capable on vision tasks. E4B it was.

The model card specified sampling parameters: temperature 1.0, top_p 0.95, top_k 64. These were followed faithfully in early runs, and later interrogated through prompt A/B testing — more on that later.

Initially in preparation for this, got set up and booked time on the HPC at OSU, but after numerous 'handshake fails" via HF decided the path of least resistance was migrating this local and this was the next phase in the plan.
The initial conda environment was called `asl_sovereign`. It was built by installing the standard HuggingFace stack — transformers, accelerate, bitsandbytes — alongside unsloth, which had been installed for a different project. This was the first mistake.

I am going to keep the within notebook top level bried for flow, and the detailed failout will be in the attached document for anyone who wants it. Not everyone will need the blow by blow technical failures. Some folks might just be here for the research part.
Unsloth ships with a custom PyTorch build. The version that landed in `asl_sovereign` was `torch 2.11.0+cu130` — a non-standard build targeting CUDA 13.0, a version that was cutting-edge at the time. bitsandbytes 0.49.2 was not built against CUDA 13.0. The conflict was silent. No error was thrown at install time. The environment appeared functional.

It was not.

**AutoModelForMultimodalLM: a class that does not exist**

The first script used `AutoModelForMultimodalLM` to load the model. This class does not exist for Gemma 4. The correct class is `AutoModelForImageTextToText`. This caused the first cascade of errors and confused several hours of debugging because the error messages were not always clear about the root cause.

**AutoProcessor: resolves to nothing**


The second problem: `AutoProcessor.from_pretrained()` threw a `ValueError: Unrecognized processing class` when pointed at the Gemma 4 model repository. The initial assumption was a HuggingFace authentication issue — the model was gated behind a license agreement. The HuggingFace CLI login was performed, the license was accepted on the web interface, the token was configured. The error persisted.

The actual cause: transformers 4.52.0, which was the stable release at the time, only supported up to `Gemma3Processor`. Gemma 4 support had not yet been merged into the stable branch. The fix required upgrading to transformers 5.5.3 (a dev/nightly build), and even then `AutoProcessor` could not auto-resolve the Gemma 4 processor configuration. The explicit class `Gemma4Processor` had to be imported directly.

This was confirmed by running:
```python
[x for x in dir(transformers) if 'gemma4' in x.lower()]
```
In 4.52.0 the list was empty. In 5.5.3 it included `Gemma4Processor`, `Gemma4ForConditionalGeneration`, and a full suite of supporting classes.
**The token explosion**

With the processor finally loading, the first inference attempts produced a peculiar output in the diagnostic script: the decoded prompt showed hundreds of repeated `<|image|>` tokens instead of a single bounded vision token block. The model output described every image as "placeholder canvas" or "gray static noise."

The cause was a double injection bug. The original script manually constructed a raw prompt string containing the special tokens `<|turn>`, `<|image|>`, and `<bos>` — formatting copied and adapted from documentation examples. At the same time, the image was passed to `processor()`, which performed its own token injection. The processor saw the manually placed `<|image|>` token and then also injected image tokens for the actual pixel data, resulting in the explosion.

The fix was conceptually simple: never manually write special tokens. Let `apply_chat_template()` handle all token injection by passing a structured `messages` dict instead of a raw string. In practice, arriving at this fix required significant back-and-forth because the manual approach had been adopted specifically to try to fix an earlier problem — a debugging spiral where each attempted fix introduced a new issue.

The diagnostic confirmed this: the chat template output showed exactly one `<|image|>` placeholder. The native processor injection showed a clean, bounded vision token block. 256 decoded image tokens for a 224x224 image at the default token budget. This was correct behavior — it had appeared broken because of the explosion, but the underlying mechanism was sound.
*The clean environment: asl_gemma4**

A new conda environment was created from scratch: `asl_gemma4`, Python 3.11, `torch 2.5.1+cu121` (official PyTorch with CUDA 12.1, confirmed stable), bitsandbytes 0.49.2, transformers 5.5.3. The `asl_sovereign` environment was abandoned entirely and should not be used.

The critical rule: always confirm `(asl_gemma4)` in the PowerShell prompt before running anything. The old environment's torch version will silently corrupt quantization and produce meaningless results.

---

## Phase 3: The Vision Blindness Problem

With the environment resolved, the pipeline appeared functional. The diagnostic showed valid pixel_values — shape `[1, 2520, 768]`, dtype float32, min/max 0.0/1.0. The image was reaching the model. And yet: every single image was described as "gray static noise," "placeholder canvas," or "uniform gray background." Letters were not identified. Hands were not described.

The first hypothesis was dataset distribution bias — that the model's vision encoder had been trained on curated white-hand benchmark images and could not generalize to the real-world diverse Roboflow dataset. This was a reasonable hypothesis and, as it turned out, directionally interesting but wrong about the immediate cause.

**Ted the cat**
In that place of frustration that borders on sheer manic glee I thought well if the model can't see a 1.2 MB picture of the cat I am lo-fi trying to woo... So Ted became the diagnostic.
Ted, a 4000x3000 JPEG, 1.2MB, a tabby cat lying in a patio area with bricks and wood and a window. This image was passed to the model with a simple prompt: "Describe what you see."

The response: "This image is a solid, uniform gray color. There are no discernible objects, scenes, or features within the frame."
/content/ted.jpg

The model could not see Ted. The model could not see anything. This was not a dataset distribution problem. This was a complete vision encoder failure affecting all images regardless of content.


### What a model card should have told us

The pipeline that worked: **Gemma 4 via Ollama**. Pre-baked GGUF quantization preserves the vision encoder. HuggingFace runtime quantization with bitsandbytes silently corrupts it — producing gray noise despite valid pixel values. Pipeline validation was confirmed using a photograph of Ted, a neighbor's cat of assigned name and gender, whose visibility to the model established correct vision encoder function.

Once the pipeline was stable, the first finding was immediate: the model could describe hands in sophisticated kinematic detail — finger orientation, palm facing, joint angles — and still output the wrong letter. The failure was not perceptual. It was somewhere between perception and classification.

---
## Section 2: Prompt Engineering — What the Model Wanted

*The root cause was the 4-bit NF4 quantization applied at runtime through bitsandbytes. The llm_int8_skip_modules parameter was intended to protect the vision encoder from quantization — the idea being that the language model layers could be quantized for memory efficiency while the vision encoder maintained full precision. In practice, the layer names passed to llm_int8_skip_modules ("vision_tower", "vision_model", "image_encoder") did not correctly map to Gemma 4's actual internal architecture. The parameter was silently ignored and the vision encoder was quantized anyway, zeroing out its outputs and producing the gray noise behavior.

Multiple attempts were made to fix this — trying different layer name variations, adjusting the dtype parameter, restructuring the quantization config — but without reliable documentation of Gemma 4's internal layer naming convention, this was guesswork.

Ollama: the solution hiding in plain sight

Ollama had been installed on the machine from the start, visible in Task Manager throughout the project. The model had already been pulled: gemma4:e4b, 9.6GB, sitting cached and ready.

Ollama uses GGUF format with llama.cpp for inference. Unlike HuggingFace's runtime quantization, Ollama pre-bakes quantization at model conversion time, during which the vision encoder weights are handled correctly. The vision encoder is preserved. The language model is quantized efficiently. There is no layer name mapping to worry about.

A single test confirmed it: Ted was described in precise, accurate, affectionate detail. The model could see the tabby cat, his peaceful demeanor , and the stoop.

![Ted](https://raw.githubusercontent.com/phinnphace/asl-sovereign/main/ted.jpg)

The entire HuggingFace pipeline was abandoned. The new architecture was simple: PIL for image loading, base64 encoding, an HTTP POST to http://localhost:11434/api/chat, pandas for results. No torch, no transformers, no bitsandbytes, no GPU setup. Just requests and a locally running Ollama server. The model card is explicitly non-declaratory on this. Perhaps this was too simplistic for me to grasp and that is on me https://ai.google.dev/gemma/docs/core/model_card_4#1_sampling_parameters I need a protocol that states something to the effect of " this model operates with Ollama. Hard stop" Guidance is suggestive to me not a protcol.







Unrecognized processing class` when pointed at the Gemma 4 model repository. The initial assumption was a HuggingFace authentication issue — the model was gated behind a license agreement. The HuggingFace CLI login was performed, the license was accepted on the web interface, the token was configured. The error persisted.

