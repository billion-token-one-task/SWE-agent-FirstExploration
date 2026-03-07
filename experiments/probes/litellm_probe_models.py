import os
import litellm

key = os.getenv("OPENAI_API_KEY")
base = "https://api.tabcode.cc/openai"
models = ["gpt-5.3-codex", "gpt-5", "gpt-5-mini", "o4-mini", "gpt-4o-mini"]
for m in models:
    try:
        r = litellm.completion(
            model=m,
            messages=[{"role": "user", "content": "reply ok"}],
            max_tokens=16,
            timeout=30,
            api_base=base,
            api_key=key,
        )
        print(m, "=> OK", repr(r.choices[0].message.content))
    except Exception as e:
        print(m, "=> ERR", type(e).__name__, str(e)[:260])
