import requests, base64, json
from PIL import Image
import io

img = Image.new('RGB', (300, 248), color=(100, 150, 200))
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

for ctx in [2048, 8192, 32768]:
    resp = requests.post('http://localhost:11434/api/chat', json={
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": "Describe this image in one word.", "images": [img_b64]}],
        "options": {"temperature": 0.1, "num_ctx": ctx},
        "stream": False
    })
    data = resp.json()
    print(f"num_ctx={ctx}: prompt_eval_count={data.get('prompt_eval_count')} eval_count={data.get('eval_count')}")
    print(f"  response: {data['message']['content'][:80]}")