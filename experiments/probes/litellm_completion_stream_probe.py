import os, json
import litellm

base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

try:
    stream = litellm.completion(
        model="gpt-5.3-codex",
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=32,
        api_base=base,
        api_key=key,
        stream=True,
    )
    print("stream_type", type(stream))
    for i,chunk in enumerate(stream, 1):
        d = chunk.model_dump() if hasattr(chunk, 'model_dump') else (chunk if isinstance(chunk, dict) else {'repr':repr(chunk)})
        print("CHUNK", i, json.dumps(d, ensure_ascii=False)[:500])
        if i>=20:
            break
except Exception as e:
    print("ERR", type(e).__name__, str(e))
