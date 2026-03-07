import os, json
import litellm

base = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

tools = [{
    "type": "function",
    "function": {
        "name": "echo_tool",
        "description": "Echoes text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}]

stream = litellm.completion(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "Call echo_tool with text='hello'."}],
    tools=tools,
    stream=True,
    max_tokens=80,
    api_base=base,
    api_key=key,
)
chunks = list(stream)
resp = litellm.stream_chunk_builder(chunks=chunks)
d = resp.model_dump() if hasattr(resp, 'model_dump') else resp
print(json.dumps(d, ensure_ascii=False)[:2000])
