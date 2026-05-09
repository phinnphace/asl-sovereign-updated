import requests
import json

def analyze_failure_mode(user_input: str, model_name: str = "gemma") -> dict:
    """
    Acts as the diagnostic routing agent. Takes unstructured user complaints,
    prompts a local Gemma 4 model via Ollama, and extracts the training data fingerprint.
    """
    
    # The Ollama local API endpoint (default)
    url = "http://localhost:11434/api/generate"
    
    # Strict Few-Shot System Prompt to force Gemma into a diagnostic state
    system_prompt = """You are an expert AI diagnostic routing agent specializing in American Sign Language (ASL) recognition models.
Your objective is to read a user's unstructured complaint or error log and determine what training data the failing model was likely trained on, based on its failure mode (confusion matrix fingerprint).

RULES:
1. If the failure involves spatial depth, body-anchored signs, touching the chest/shoulders/face, or two hands crossing, the model was likely trained exclusively on limited landmark data (e.g., 21-hand landmarks).
2. If the failure is purely lexical (confusing similar handshapes like 'M' and 'N' but spatial signs are fine), the model was likely trained on robust image data but needs standard fine-tuning.
3. You MUST output your response in strict, valid JSON format matching the structure below. Do not include any markdown formatting, preamble, or conversational text. Just the JSON object.

JSON STRUCTURE:
{
    "failure_mode": "Brief classification of the error",
    "provenance_diagnosis": "Your diagnosis of the training data",
    "recommendation": "Specific routing or fine-tuning advice",
    "compute_saved": "Estimation of compute saved by not fine-tuning a structurally flawed model"
}

EXAMPLE 1:
Input: "The app works fine for spelling my name, but it completely breaks when I try to sign 'tired' or 'hungry' where my hands touch my chest."
Output:
{
    "failure_mode": "Spatial / Body-Anchored Occlusion",
    "provenance_diagnosis": "Model was likely trained EXCLUSIVELY on 21-hand landmark data. It lacks full-body topology.",
    "recommendation": "DO NOT FINE-TUNE. Route this task to a Full-Body Pose Estimation or Hybrid model.",
    "compute_saved": "High. Prevents wasted retraining cycles on an impossible structural task."
}
"""

    prompt = f"{system_prompt}\n\nCURRENT INPUT:\n{user_input}\n\nOUTPUT:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Forces Ollama to return structured JSON
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result_text = response.json().get("response", "")
        
        # Parse the JSON string returned by Gemma
        try:
            structured_data = json.loads(result_text)
            return structured_data
        except json.JSONDecodeError:
            # Fallback if Gemma disobeys the JSON constraint
            return {
                "failure_mode": "Parsing Error",
                "provenance_diagnosis": "Model returned unstructured text.",
                "recommendation": "Check Ollama logs. Ensure format='json' is supported by your local build.",
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

# Quick test execution block
if __name__ == "__main__":
    test_complaint = "It gets hand shapes right but fails whenever the sign touches the shoulder."
    print("Testing Decoder Ring...")
    # Change "gemma" to your exact local model tag (e.g., "gemma:7b" or "gemma-4")
    print(json.dumps(analyze_failure_mode(test_complaint, "gemma"), indent=4))
