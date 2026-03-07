import os
import json
import litellm

base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

try:
    stream = litellm.responses(
        model="gpt-5.3-codex",
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "Reply with OK only."}]}
        ],
        api_base=base,
        api_key=key,
        max_output_tokens=32,
        stream=True,
    )
    print("stream_type", type(stream))
    count=0
    for ev in stream:
        count += 1
        d = ev.model_dump() if hasattr(ev, 'model_dump') else (ev if isinstance(ev, dict) else {'repr':repr(ev)})
        print("EVENT", count, json.dumps(d, ensure_ascii=False)[:400])
        if count>=20:
            break
except Exception as e:
    print("ERR", type(e).__name__, str(e))
