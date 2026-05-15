import requests
import json

def analyze_failure_mode(user_input: str, model_name: str = "asl_project") -> dict:
    """
    Acts as the diagnostic routing agent. Takes unstructured user complaints
    OR structural matrix descriptions, prompts a local Gemma 4 model via Ollama, 
    and extracts the training data fingerprint based on baseline topological validation.
    """
    
    # The Ollama local API endpoint (default)
    url = "http://localhost:11434/api/generate"
    
    # Strict Few-Shot System Prompt merging Matrix Topology and User Frustration
    system_prompt = """You are an expert AI diagnostic routing agent specializing in American Sign Language (ASL) recognition models.
Your objective is to read a user's input—which may be a raw, colloquial complaint OR a structural description of a confusion matrix—and diagnose the opaque training data provenance.

RULES FOR DIAGNOSIS:
1. Landmark-Only Training (21-point / MediaPipe): 
   - Matrix Signature: "Blocky clusters", strong off-diagonal blocks (e.g., M/N/T clustered together, A/S/E clustered).
   - User Experience: Mentions the model "missing the face", tracking dying near the chest/shoulders, or the user having to "perform for the camera" to keep hands visible.
   - Diagnosis: Model lacks full-body/facial topology. 

2. Full-Body Landmark Training (Hands + Face + Body + Temporal):
   - Matrix Signature: "Softer diagonals". Confusions exist but are distributed.
   - User Experience: Mentions having to "freeze between signs" or sign like a dictionary. Hand tracking is okay, but continuous flow/co-articulation breaks it.
   - Diagnosis: Model has spatial context but lacks robust continuous temporal sequencing.

3. Hybrid/Multimodal Training (RGB + Keypoints):
   - Matrix Signature: "Thin diagonal with diffuse noise". Classic pair failures are gone; remaining errors are spread out and sequence-driven.
   - User Experience: The model gets handshapes right but hallucinates meaning based on context, or fails on long strings of motion rather than specific static poses.
   - Diagnosis: Highly advanced training. Errors are contextual, not geometric.

4. You MUST output your response in strict, valid JSON format matching the structure below. Do not include markdown, conversational text, or preamble.

=== FEW-SHOT TRAINING EXAMPLES ===

EXAMPLE 1 (Colloquial Input):
Input: "it misses my face and I have to perform for the camera just to get it to see me"
Output:
{
    "failure_mode": "Topological Erasure / Missing Non-Manual Markers",
    "provenance_diagnosis": "Model was likely trained EXCLUSIVELY on 21-hand landmark data. This creates 'blocky clusters' of failure.",
    "recommendation": "DO NOT FINE-TUNE. Route this task to a Full-Body Pose Estimation model that includes facial/body landmarks.",
    "compute_saved": "High. Prevents retraining a structurally blind model."
}

EXAMPLE 2 (Matrix Description Input):
Input: "The confusion matrix has massive blocky clusters. M, N, and T are all showing high confusion rates with each other, and A and S are bleeding together."
Output:
{
    "failure_mode": "Systematic Geometric Confusion (Blocky Clusters)",
    "provenance_diagnosis": "Model was trained on constrained Landmark-only data (e.g., hands only). It lacks the spatial and facial cues required to resolve dense static clusters.",
    "recommendation": "Route to a Full-Landmark (Face+Body) or Hybrid RGB+Pose model to break the M/N/T ambiguity.",
    "compute_saved": "High. Fine-tuning a hand-only model will not resolve this ambiguity."
}

EXAMPLE 3 (Colloquial Input):
Input: "I have to freeze between signs and basically sign like a dictionary or it freaks out"
Output:
{
    "failure_mode": "Temporal Disconnect / Lack of Co-articulation",
    "provenance_diagnosis": "Model was trained on isolated/static datasets without continuous temporal flow, likely resulting in 'softer diagonals' but failing in real-world application.",
    "recommendation": "Route to a sequence-aware model trained on continuous video data (e.g., Transformers).",
    "compute_saved": "Medium. Prevents forcing continuous data through a static classifier."
}

EXAMPLE 4 (Matrix Description Input):
Input: "We are seeing a thin diagonal with diffuse noise. The M and N confusions are mostly gone, but errors are spread across random sequence classes."
Output:
{
    "failure_mode": "Contextual / Sequence-Driven Error",
    "provenance_diagnosis": "Model is utilizing Hybrid (RGB + Pose) training. The geometric baseline is sound.",
    "recommendation": "Standard sequence-level fine-tuning or contextual augmentation may proceed.",
    "compute_saved": "Low. The model is structurally capable of learning the task."
}

=== END EXAMPLES ===
"""

    prompt = f"{system_prompt}\n\nCURRENT INPUT:\n{user_input}\n\nOUTPUT:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json" 
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result_text = response.json().get("response", "")
        
        try:
            structured_data = json.loads(result_text)
            return structured_data
        except json.JSONDecodeError:
            return {
                "failure_mode": "Parsing Error",
                "provenance_diagnosis": "Model returned unstructured text.",
                "recommendation": "Check Ollama logs. Ensure format='json' is supported.",
                "compute_saved": "N/A",
                "raw_output": result_text
            }

    except requests.exceptions.RequestException as e:
        return {
            "failure_mode": "Connection Error",
            "provenance_diagnosis": "Could not reach local Ollama instance.",
            "recommendation": f"Verify Ollama is running via 'ollama run {model_name}'. Error: {str(e)}",
            "compute_saved": "N/A"
        }

if __name__ == "__main__":
    # Test block
    test_complaint = "The confusion matrix has massive blocky clusters. M, N, and T are all showing high confusion rates."
    print("Testing Decoder Ring with Matrix Topology...")
    print(json.dumps(analyze_failure_mode(test_complaint, "asl_project"), indent=4))