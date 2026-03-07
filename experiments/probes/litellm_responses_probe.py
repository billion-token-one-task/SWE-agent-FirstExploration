import os
import json
import litellm

base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

try:
    r = litellm.responses(
        model="gpt-5.3-codex",
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "Reply with OK only."}]}
        ],
        api_base=base,
        api_key=key,
        max_output_tokens=32,
    )
    print("TYPE", type(r))
    if hasattr(r, "model_dump"):
        d = r.model_dump()
    elif isinstance(r, dict):
        d = r
    else:
        d = {"repr": repr(r)}
    print("KEYS", list(d.keys())[:60])
    print("SNIP", json.dumps(d, ensure_ascii=False)[:1600])
except Exception as e:
    print("ERR", type(e).__name__, str(e))
