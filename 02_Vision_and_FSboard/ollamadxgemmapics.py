# Quick diagnostic — run in asl_gemma4
import requests, base64, json
from PIL import Image
import io

# Create a test image at known size
img = Image.new('RGB', (224, 224), color=(128, 128, 128))
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

resp = requests.post('http://localhost:11434/api/chat', json={
    "model": "gemma4:e4b",
    "messages": [{"role": "user", "content": "What size is this image?", "images": [img_b64]}],
    "options": {"temperature": 0.1},
    "stream": False,
    "verbose": True  # asks Ollama to return timing/token info
})
data = resp.json()
print(json.dumps(data, indent=2))