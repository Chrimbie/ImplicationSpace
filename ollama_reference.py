"""Ground truth: what does the real CodeLlama 70B say after
'hello'? Ask Ollama (llama.cpp) directly — raw completion,
greedy (temperature 0, fixed seed). My from-space stack
predicted token 29875 ('i') after [BOS, hello]; this is the
reference that prediction is judged against."""
import json
import urllib.request

body = {
    "model": "codellama:70b",
    "prompt": "hello",
    "raw": True,
    "stream": False,
    "options": {"temperature": 0.0, "seed": 42, "num_predict": 20},
}
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    out = json.loads(r.read())
print("RESPONSE:", repr(out.get("response")))
print("eval_count:", out.get("eval_count"),
      "eval_duration_s:", (out.get("eval_duration") or 0) / 1e9,
      "load_duration_s:", (out.get("load_duration") or 0) / 1e9)
open("ollama_reference_result.json", "w").write(
    json.dumps(out, indent=1))
print("saved ollama_reference_result.json")
