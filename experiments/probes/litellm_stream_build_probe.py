import os, json
import litellm

base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

stream = litellm.completion(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "Reply with OK only."}],
    max_tokens=32,
    api_base=base,
    api_key=key,
    stream=True,
)
chunks=[]
for c in stream:
    chunks.append(c)
print('chunks', len(chunks))
resp = litellm.stream_chunk_builder(chunks=chunks)
print('resp_type', type(resp))
d = resp.model_dump() if hasattr(resp,'model_dump') else (resp if isinstance(resp,dict) else {'repr':repr(resp)})
print('keys', list(d.keys())[:40])
print('usage', d.get('usage'))
print('choice0', d.get('choices',[{}])[0].get('message'))
