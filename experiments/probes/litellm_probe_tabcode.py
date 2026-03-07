import os
import litellm

key = os.getenv("OPENAI_API_KEY")
for base in [
    "https://api.tabcode.cc/openai",
    "https://api.tabcode.cc/openai/v1",
    "https://api.tabcode.cc/v1",
    "https://api.tabcode.cc",
]:
    try:
        r = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "reply ok"}],
            max_tokens=6,
            timeout=20,
            api_base=base,
            api_key=key,
        )
        print(base, "=> OK", repr(r.choices[0].message.content))
    except Exception as e:
        print(base, "=> ERR", type(e).__name__, str(e)[:240])
